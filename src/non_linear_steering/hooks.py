"""Hook management for steering and probe capture via TransformerLens

TransformerLens names residual stream positions as:
  - blocks.{layer}.hook_resid_pre   (input to layer)
  - blocks.{layer}.hook_resid_post  (output of layer, after MLP)
  - blocks.{layer}.hook_resid_mid   (after attention, before MLP)

We inject at hook_resid_post for the inject layer and capture at hook_resid_post
for the probe layer
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint
from transformer_lens.model_bridge import TransformerBridge

ModelWithHooks = HookedTransformer | TransformerBridge


class HookState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    steering: torch.nn.Module | None = None
    steer_mask: torch.Tensor | None = None
    active: bool = False
    capture: bool = False
    captured: torch.Tensor | None = None


class HookedModel(BaseModel):
    """Hook manager for a TransformerLens model."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: ModelWithHooks
    inject_layers: set[int] = Field(default_factory=set)
    probe_layer: int = 0
    state: HookState = Field(default_factory=HookState)

    @model_validator(mode="after")
    def check_layers(self) -> HookedModel:
        n_layers = self.model.cfg.n_layers
        for layer in self.inject_layers | {self.probe_layer}:
            if not 0 <= layer < n_layers:
                raise ValueError(
                    f"layer {layer} is out of range for a model with {n_layers} layers"
                )
        return self

    def _inject_hook(self, layer: int) -> Callable:
        def hook(activation: torch.Tensor, hook: HookPoint) -> torch.Tensor:
            if not self.state.active or layer not in self.inject_layers:
                return activation
            if self.state.steering is None or self.state.steer_mask is None:
                raise RuntimeError("Steering requested without steering module or mask")
            if self.state.steer_mask.shape[1] != activation.shape[1]:
                raise RuntimeError(
                    f"steer_mask width {self.state.steer_mask.shape[1]} does not match "
                    f"the forward width {activation.shape[1]}"
                )
            mask = self.state.steer_mask[..., None].to(activation.dtype)
            delta = self.state.steering(activation, layer)
            return activation + delta * mask

        return hook

    def _capture_hook(self) -> Callable:
        def hook(activation: torch.Tensor, hook: HookPoint) -> torch.Tensor:
            if self.state.capture:
                self.state.captured = activation
            return activation

        return hook

    def _build_fwd_hooks(self) -> list[tuple[str, Callable]]:
        hooks = []
        for layer in sorted(self.inject_layers):
            hooks.append((f"blocks.{layer}.hook_resid_post", self._inject_hook(layer)))
        if self.probe_layer not in self.inject_layers:
            hooks.append((f"blocks.{self.probe_layer}.hook_resid_post", self._capture_hook()))
        else:
            old_inject = self._inject_hook(self.probe_layer)

            def combined(activation: torch.Tensor, hook: HookPoint) -> torch.Tensor:
                result = old_inject(activation, hook)
                if self.state.capture:
                    self.state.captured = result
                return result

            for i, (name, _) in enumerate(hooks):
                if name == f"blocks.{self.probe_layer}.hook_resid_post":
                    hooks[i] = (name, combined)
                    break
            else:
                hooks.append((f"blocks.{self.probe_layer}.hook_resid_post", combined))
        return hooks

    @contextmanager
    def session(self, *, steering=None, steer_mask=None, capture=False):
        previous = self.state
        self.state = HookState(
            steering=steering,
            steer_mask=steer_mask,
            active=steering is not None,
            capture=capture,
            captured=None,
        )
        try:
            yield self.state
        finally:
            self.state = previous

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        return_type: str = "logits",
    ) -> torch.Tensor:
        """Run the model with the current session's hooks active."""
        fwd_hooks = self._build_fwd_hooks()
        model_kwargs = {}
        if attention_mask is not None:
            model_kwargs["attention_mask"] = attention_mask
        if position_ids is not None:
            model_kwargs["position_ids"] = position_ids
        return self.model.run_with_hooks(
            input_ids,
            fwd_hooks=fwd_hooks,
            return_type=return_type,
            **model_kwargs,
        )

    def close(self) -> None:
        pass


def load_hooked_model(
    model_name_or_path: str | Path,
    *,
    dtype: torch.dtype | str = "auto",
    device: str | None = None,
) -> TransformerBridge:
    """Load a checkpoint through TransformerLens.

    boot_transformers accepts HuggingFace model names and local checkpoint
    directories; compatibility mode restores the HookedTransformer hook names
    and load-time weight processing.
    """
    if dtype == "auto":
        torch_dtype = torch.float16
    elif isinstance(dtype, str):
        torch_dtype = getattr(torch, dtype)
    else:
        torch_dtype = dtype

    model = TransformerBridge.boot_transformers(
        str(model_name_or_path),
        dtype=torch_dtype,
        device=device,
    )
    model.enable_compatibility_mode(disable_warnings=True)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model

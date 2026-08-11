from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import torch


def resolve_decoder_blocks(model):
    candidates = [
        ("model", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
    ]
    for root_name, layers_name in candidates:
        root = getattr(model, root_name, None)
        if root is not None and hasattr(root, layers_name):
            return getattr(root, layers_name)
    raise ValueError("Could not resolve decoder blocks; expected model.model.layers")


@dataclass
class HookState:
    steering: torch.nn.Module | None = None
    steer_mask: torch.Tensor | None = None
    active: bool = False
    capture: bool = False
    captured: torch.Tensor | None = None


class HookedDecoder:
    def __init__(self, model, *, inject_layers: list[int], probe_layer: int):
        self.model = model
        self.blocks = resolve_decoder_blocks(model)
        self.inject_layers = set(inject_layers)
        self.probe_layer = probe_layer
        self.state = HookState()
        self.handles = []
        relevant = sorted(self.inject_layers | {probe_layer})
        for layer_index in relevant:
            if not 0 <= layer_index < len(self.blocks):
                raise ValueError(
                    f"layer {layer_index} is out of range for a model with "
                    f"{len(self.blocks)} decoder blocks"
                )
        for layer_index in relevant:
            self.handles.append(
                self.blocks[layer_index].register_forward_hook(self._make_hook(layer_index))
            )

    def _make_hook(self, layer_index: int):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if self.state.active and layer_index in self.inject_layers:
                if self.state.steering is None or self.state.steer_mask is None:
                    raise RuntimeError("Steering requested without steering module or mask")
                if self.state.steer_mask.shape[1] != hidden.shape[1]:
                    raise RuntimeError(
                        "steer_mask width "
                        f"{self.state.steer_mask.shape[1]} does not match the forward "
                        f"width {hidden.shape[1]}; the mask must cover every position "
                        "of this pass (no-cache replay)"
                    )
                mask = self.state.steer_mask[..., None].to(hidden.dtype)
                hidden = hidden + self.state.steering(hidden, layer_index) * mask
            if self.state.capture and layer_index == self.probe_layer:
                self.state.captured = hidden
            if isinstance(output, tuple):
                return (hidden, *output[1:])
            return hidden

        return hook

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

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

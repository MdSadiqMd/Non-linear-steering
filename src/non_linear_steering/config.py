from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

DEFAULT_MODEL = Path("/Users/sadiq/.cache/huggingface/hub/models--openai--gpt-oss-20b")

SteerPosition = Literal["prediction-state", "assistant-token-only"]
STEER_POSITIONS: tuple[SteerPosition, ...] = ("prediction-state", "assistant-token-only")


class HookSpec(BaseModel, frozen=True):
    """Configuration for hook placement during steering"""

    inject_layer: int = Field(ge=0, description="Layer index to inject steering")
    probe_layer: int = Field(ge=0, description="Layer index to read probe activations")
    steer_position: SteerPosition = Field(
        default="prediction-state",
        description="Where to apply steering: prediction-state (includes final prompt position) or assistant-token-only",
    )

    @property
    def activation_channel_open(self) -> bool:
        """Whether the frozen probe reads a stream the intervention can still reach

        The hook injects before it captures, so inject_layer == probe_layer still
        leaves the direct path open. Injecting later leaves only the behavior path
        """
        return self.inject_layer <= self.probe_layer

    def validate_for_model(self, num_layers: int) -> None:
        """Validate that layer indices are within the model's range"""
        for name, value in (
            ("inject_layer", self.inject_layer),
            ("probe_layer", self.probe_layer),
        ):
            if value >= num_layers:
                raise ValueError(
                    f"{name}={value} is out of range for a model with {num_layers} layers"
                )


class TrainSpec(BaseModel, frozen=True):
    """Training hyperparameters for steering optimization."""

    horizon: int = Field(default=64, gt=0, description="Maximum tokens to generate per rollout")
    temperature: float = Field(default=1.0, gt=0, description="Sampling temperature")
    epsilon: float = Field(default=0.1, ge=0, description="KL divergence budget")
    dual_lr: float = Field(default=0.05, ge=0, description="Learning rate for dual variable β")
    radius: float = Field(default=0.5, gt=0, description="Maximum norm of steering perturbation")
    rank: int = Field(default=32, gt=0, description="Bottleneck dimension of steering network")
    batch_size: int = Field(default=1, gt=0, description="Number of prompts per training step")
    steps: int = Field(default=100, gt=0, description="Total training steps")
    lr: float = Field(default=1e-4, gt=0, description="Learning rate for steering parameters")
    baseline_decay: float = Field(
        default=0.95,
        ge=0,
        lt=1,
        description="Exponential moving average decay for baseline",
    )


class ProbeConfig(BaseModel, frozen=True):
    """Configuration for the causal probe architecture."""

    hidden_size: int = Field(gt=0, description="Model hidden dimension (must match LLM)")
    probe_width: int = Field(default=512, gt=0, description="Probe internal dimension")
    layers: int = Field(default=2, gt=0, description="Number of transformer layers in probe")
    heads: int = Field(default=8, gt=0, description="Number of attention heads in probe")
    dropout: float = Field(default=0.0, ge=0, le=1, description="Dropout probability")

    @model_validator(mode="after")
    def check_divisibility(self) -> ProbeConfig:
        if self.probe_width % self.heads != 0:
            raise ValueError(
                f"probe_width={self.probe_width} must be divisible by heads={self.heads}"
            )
        return self

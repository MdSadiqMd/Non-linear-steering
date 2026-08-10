from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class NonlinearDelta(nn.Module):
    def __init__(self, hidden_size: int, rank: int = 32, radius: float = 0.5):
        super().__init__()
        self.hidden_size = hidden_size
        self.radius = radius
        self.down = nn.Linear(hidden_size, rank)
        self.up = nn.Linear(rank, hidden_size, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        dtype = hidden_states.dtype
        normalized = F.layer_norm(hidden_states.float(), (self.hidden_size,))
        raw = self.up(F.silu(self.down(normalized)))
        norm = raw.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        scale = (self.radius / norm).clamp(max=1.0)
        return (raw * scale).to(dtype)


class LayerwiseSteering(nn.Module):
    def __init__(self, hidden_size: int, layers: list[int], rank: int, radius: float):
        super().__init__()
        self.deltas = nn.ModuleDict(
            {str(layer): NonlinearDelta(hidden_size, rank, radius) for layer in layers}
        )

    def forward(self, hidden_states: torch.Tensor, layer_index: int) -> torch.Tensor:
        return self.deltas[str(layer_index)](hidden_states)

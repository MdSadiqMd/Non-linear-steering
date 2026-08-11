from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .config import ProbeConfig


class CausalProbe(nn.Module):
    def __init__(self, config: ProbeConfig):
        super().__init__()
        self.config = config
        self.inp = nn.Linear(config.hidden_size, config.probe_width)
        block = nn.TransformerEncoderLayer(
            d_model=config.probe_width,
            nhead=config.heads,
            dim_feedforward=4 * config.probe_width,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(
            block, num_layers=config.layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(config.probe_width)
        self.out = nn.Linear(config.probe_width, 1)

    def _attention_mask(self, batch: int, seq_len: int, attention_mask: torch.Tensor):
        device = attention_mask.device
        blocked = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device).triu(1)
        blocked = blocked[None] | (~attention_mask.bool())[:, None, :]
        blocked = blocked & ~torch.eye(seq_len, dtype=torch.bool, device=device)
        heads = self.config.heads
        return (
            blocked[:, None]
            .expand(batch, heads, seq_len, seq_len)
            .reshape(batch * heads, seq_len, seq_len)
        )

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch, seq_len = hidden_states.shape[0], hidden_states.shape[1]
        mask = self._attention_mask(batch, seq_len, attention_mask.to(hidden_states.device))
        x = self.inp(hidden_states.float())
        x = self.blocks(x, mask=mask)
        return self.out(self.norm(x)).squeeze(-1)


def save_probe(path: str | Path, probe: CausalProbe, *, probe_layer: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": probe.state_dict(),
            "config": probe.config.model_dump(),
            "probe_layer": probe_layer,
        },
        path,
    )


def load_probe(path: str | Path, map_location="cpu") -> tuple[CausalProbe, int]:
    ckpt = torch.load(Path(path), map_location=map_location)
    probe = CausalProbe(ProbeConfig(**ckpt["config"]))
    probe.load_state_dict(ckpt["state_dict"])
    probe.eval()
    for parameter in probe.parameters():
        parameter.requires_grad_(False)
    return probe, int(ckpt["probe_layer"])

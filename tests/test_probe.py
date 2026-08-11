from __future__ import annotations

from pathlib import Path

import torch

from non_linear_steering.config import ProbeConfig
from non_linear_steering.probe import CausalProbe, load_probe, save_probe


def make_probe(seed: int = 0) -> CausalProbe:
    torch.manual_seed(seed)
    return CausalProbe(ProbeConfig(hidden_size=8, probe_width=16, layers=2, heads=2)).eval()


def test_probe_is_causal():
    probe = make_probe()
    hidden = torch.randn(1, 6, 8)
    mask = torch.ones(1, 6, dtype=torch.long)
    with torch.no_grad():
        before = probe(hidden, mask)
        edited = hidden.clone()
        edited[:, 4:] = torch.randn(1, 2, 8)
        after = probe(edited, mask)
    assert torch.allclose(before[:, :4], after[:, :4], atol=1e-6)
    assert not torch.allclose(before[:, 4], after[:, 4], atol=1e-6)


def test_probe_output_is_independent_of_where_the_padding_sits():
    """Left padding leaves a causal query row with no visible key, which yields NaN
    unless the mask keeps the diagonal open. The score is read at a real token, so
    every layout must agree with the unpadded reference."""
    probe = make_probe()
    content = torch.randn(1, 5, 8)
    filler = torch.randn(1, 3, 8)
    with torch.no_grad():
        reference = probe(content, torch.ones(1, 5, dtype=torch.long))[0, -1]
        layouts = {
            "right": (torch.cat([content, filler], dim=1), [1] * 5 + [0] * 3, 4),
            "left": (torch.cat([filler, content], dim=1), [0] * 3 + [1] * 5, 7),
            "interior": (
                torch.cat([content[:, :2], filler, content[:, 2:]], dim=1),
                [1, 1] + [0] * 3 + [1] * 3,
                7,
            ),
        }
        for name, (hidden, mask, terminal) in layouts.items():
            scores = probe(hidden, torch.tensor([mask]))
            assert not torch.isnan(scores).any(), name
            assert torch.allclose(scores[0, terminal], reference, atol=1e-5), name


def test_probe_padding_invariance_holds_without_grad_mode():
    """The fused encoder path only runs under no_grad, so it needs its own check."""
    probe = make_probe()
    content = torch.randn(1, 5, 8)
    padded = torch.cat([torch.randn(1, 3, 8), content], dim=1)
    with torch.no_grad():
        reference = probe(content, torch.ones(1, 5, dtype=torch.long))[0, -1]
        scores = probe(padded, torch.tensor([[0, 0, 0, 1, 1, 1, 1, 1]]))
    assert not torch.isnan(scores).any()
    assert torch.allclose(scores[0, -1], reference, atol=1e-5)


def test_saved_probe_round_trips_frozen(tmp_path: Path):
    probe = make_probe()
    path = tmp_path / "nested" / "probe.pt"
    save_probe(path, probe, probe_layer=12)
    loaded, probe_layer = load_probe(path)
    assert probe_layer == 12
    assert loaded.config == probe.config
    assert not any(parameter.requires_grad for parameter in loaded.parameters())
    assert not loaded.training
    hidden = torch.randn(1, 4, 8)
    mask = torch.ones(1, 4, dtype=torch.long)
    with torch.no_grad():
        assert torch.allclose(probe(hidden, mask), loaded(hidden, mask), atol=1e-6)

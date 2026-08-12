from __future__ import annotations

import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig

from non_linear_steering.config import ProbeConfig
from non_linear_steering.hooks import HookedModel
from non_linear_steering.probe import CausalProbe
from non_linear_steering.steering import LayerwiseSteering

VOCAB = 6
EOS = 5
HIDDEN = 32
INJECT_LAYER = 1
PROBE_LAYER = 2


class WhitespaceTokenizer:
    """Maps "1 2 3" to [1, 2, 3] so tests can control token ids exactly.

    Only the surface the code touches is implemented: __call__ with padding,
    padding_side, eos_token_id, and pad_token_id.
    """

    def __init__(self, padding_side: str = "right", pad_id: int = EOS):
        self.eos_token_id = EOS
        self.pad_token_id = pad_id
        self.eos_token = "<eos>"
        self.padding_side = padding_side

    def __call__(self, prompts, return_tensors=None, padding=True, add_special_tokens=True):
        seqs = [[int(token) for token in prompt.split()] for prompt in prompts]
        width = max(len(seq) for seq in seqs)
        ids, mask = [], []
        for seq in seqs:
            pad = [self.pad_token_id] * (width - len(seq))
            if self.padding_side == "left":
                ids.append(pad + seq)
                mask.append([0] * len(pad) + [1] * len(seq))
            else:
                ids.append(seq + pad)
                mask.append([1] * len(seq) + [0] * len(pad))
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(mask)}


def build_stack(seed: int = 0, radius: float = 8.0, inject_layer: int = INJECT_LAYER):
    """A tiny frozen TransformerLens LM plus frozen probe plus steering module."""
    torch.manual_seed(seed)
    cfg = HookedTransformerConfig(
        n_layers=4,
        d_model=HIDDEN,
        d_head=16,
        n_heads=2,
        d_mlp=64,
        d_vocab=VOCAB,
        n_ctx=64,
        act_fn="gelu",
        normalization_type="LN",
    )
    model = HookedTransformer(cfg)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    probe = CausalProbe(ProbeConfig(hidden_size=HIDDEN, probe_width=16, layers=2, heads=2))
    probe.eval()
    for param in probe.parameters():
        param.requires_grad_(False)

    steering = LayerwiseSteering(HIDDEN, [inject_layer], rank=8, radius=radius)
    with torch.no_grad():
        # zero-initialised up-projection means delta == 0, which makes every
        # gradient-direction test vacuous
        steering.deltas[str(inject_layer)].up.weight.normal_(std=0.5)

    hooked = HookedModel(model=model, inject_layers=[inject_layer], probe_layer=PROBE_LAYER)
    return model, probe, steering, hooked

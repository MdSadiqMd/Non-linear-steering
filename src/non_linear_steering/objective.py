"""Replay and scoring with the TransformerLens hook system"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F
from pydantic import BaseModel, ConfigDict
from transformer_lens.hook_points import HookPoint

from .hooks import ModelWithHooks
from .trajectory import _make_steering_hook


class ReplayValues(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    score: torch.Tensor
    seq_log_prob: torch.Tensor
    seq_kl: torch.Tensor


def _make_capture_hook(container: list) -> Callable[[torch.Tensor, HookPoint], torch.Tensor]:
    def hook(activation: torch.Tensor, hook: HookPoint) -> torch.Tensor:
        container.append(activation)
        return activation

    return hook


def replay_and_score(
    model: ModelWithHooks,
    probe: torch.nn.Module,
    steering: torch.nn.Module,
    batch,
    *,
    inject_layer: int,
    probe_layer: int,
    temperature: float,
) -> ReplayValues:
    """Replay on-policy samples using TransformerLens hooks

    Runs the base pass with no hooks and the steered pass with steering plus
    probe capture, so the direct activation gradient and the score-function
    term are both available
    """
    full_ids = torch.cat([batch.prompt_ids, batch.completion_ids], dim=1)
    full_attention = torch.cat(
        [batch.prompt_attention_mask, batch.transcript_mask.to(batch.prompt_attention_mask.dtype)],
        dim=1,
    )
    prompt_width = batch.prompt_ids.shape[1]
    num_slots = batch.completion_ids.shape[1]
    prediction_slice = slice(prompt_width - 1, prompt_width + num_slots - 1)

    with torch.no_grad():
        base_logits = model.run_with_hooks(
            full_ids,
            fwd_hooks=[],
            attention_mask=full_attention,
            return_type="logits",
        )[:, prediction_slice, :].float()

    captured = []
    fwd_hooks = [
        (
            f"blocks.{inject_layer}.hook_resid_post",
            _make_steering_hook(steering, batch.steer_mask, inject_layer),
        ),
    ]
    if probe_layer != inject_layer:
        fwd_hooks.append((f"blocks.{probe_layer}.hook_resid_post", _make_capture_hook(captured)))
    else:
        original_hook = fwd_hooks[0][1]

        def combined(activation: torch.Tensor, hook: HookPoint) -> torch.Tensor:
            result = original_hook(activation, hook)
            captured.append(result)
            return result

        fwd_hooks[0] = (f"blocks.{inject_layer}.hook_resid_post", combined)

    steered_logits = model.run_with_hooks(
        full_ids,
        fwd_hooks=fwd_hooks,
        attention_mask=full_attention,
        return_type="logits",
    )[:, prediction_slice, :].float()

    if not captured:
        raise RuntimeError("Probe stream was not captured")
    stream = captured[0]

    logq = F.log_softmax(steered_logits / temperature, dim=-1)
    logp0 = F.log_softmax(base_logits / temperature, dim=-1)
    q = logq.exp()
    completion_ids = batch.completion_ids.to(logq.device)
    action_mask = batch.action_mask.to(logq.device)
    selected = logq.gather(-1, completion_ids.unsqueeze(-1)).squeeze(-1)
    token_kl = (q * (logq - logp0)).sum(dim=-1)
    action = action_mask.to(logq.dtype)
    seq_log_prob = (selected * action).sum(dim=-1)
    seq_kl = (token_kl * action).sum(dim=-1)

    probe_attention = full_attention.to(stream.device)
    probe_scores = probe(stream, probe_attention)
    row = torch.arange(full_ids.shape[0], device=stream.device)
    terminal_positions = batch.terminal_positions.to(stream.device)
    score = probe_scores[row, terminal_positions]
    return ReplayValues(
        score=score,
        seq_log_prob=seq_log_prob.to(score.device),
        seq_kl=seq_kl.to(score.device),
    )


def constrained_loss(values: ReplayValues, *, beta: float, baseline: float) -> torch.Tensor:
    direct_loss = (-values.score + beta * values.seq_kl).mean()
    lagrangian_return = values.score - beta * values.seq_kl
    advantage = lagrangian_return.detach() - baseline
    policy_loss = -(advantage * values.seq_log_prob).mean()
    return direct_loss + policy_loss

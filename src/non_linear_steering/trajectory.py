"""Rollout sampling with the TransformerLens hook system"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager

import torch
from pydantic import BaseModel, ConfigDict
from transformer_lens.hook_points import HookPoint

from .hooks import ModelWithHooks


class TrajectoryBatch(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    prompt_ids: torch.Tensor
    prompt_attention_mask: torch.Tensor
    completion_ids: torch.Tensor
    action_mask: torch.Tensor
    transcript_mask: torch.Tensor
    steer_mask: torch.Tensor
    terminal_positions: torch.Tensor
    truncated: torch.Tensor


@contextmanager
def padding_side(tokenizer, side: str):
    """Pin the tokenizer's padding side regardless of how the checkpoint ships it

    Rollout needs "left": it reads next-token logits at index -1 and replay reads
    the first prediction state at index prompt_width - 1, and both are the true
    final prompt position only when the padding sits ahead of the prompt
    """
    previous = tokenizer.padding_side
    tokenizer.padding_side = side
    try:
        yield tokenizer
    finally:
        tokenizer.padding_side = previous


def make_steer_mask(
    full_len: int,
    prompt_width: int,
    terminal_positions: torch.Tensor,
    *,
    steer_position: str,
    device,
) -> torch.Tensor:
    positions = torch.arange(full_len, device=device)[None, :]
    if steer_position == "prediction-state":
        start = prompt_width - 1
    elif steer_position == "assistant-token-only":
        start = prompt_width
    else:
        raise ValueError("Unknown steer_position")
    return (positions >= start) & (positions <= terminal_positions[:, None])


def _make_steering_hook(
    steering: torch.nn.Module,
    steer_mask: torch.Tensor,
    layer: int,
) -> Callable[[torch.Tensor, HookPoint], torch.Tensor]:
    def hook(activation: torch.Tensor, hook: HookPoint) -> torch.Tensor:
        mask = steer_mask[:, : activation.shape[1], None].to(activation.dtype)
        delta = steering(activation, layer)
        return activation + delta * mask

    return hook


@torch.no_grad()
def sample_rollout(
    model: ModelWithHooks,
    tokenizer,
    steering: torch.nn.Module,
    prompts: list[str],
    *,
    inject_layer: int,
    horizon: int,
    temperature: float,
    steer_position: str,
) -> TrajectoryBatch:
    """Sample completions from the steered policy using TransformerLens hooks"""
    with padding_side(tokenizer, "left"):
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=True,
        )
    device = model.cfg.device
    prompt_ids = encoded["input_ids"].to(device)
    prompt_attention_mask = encoded["attention_mask"].to(device)
    prompt_width = prompt_ids.shape[1]
    batch_size = prompt_ids.shape[0]
    eos_id = int(tokenizer.eos_token_id)
    pad_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id)

    full_ids = prompt_ids
    full_attention = prompt_attention_mask
    action_tokens: list[torch.Tensor] = []
    action_masks: list[torch.Tensor] = []
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    terminal_positions = torch.full(
        (batch_size,), prompt_width + horizon, dtype=torch.long, device=device
    )

    for step in range(horizon):
        current_len = full_ids.shape[1]
        provisional_terminal = torch.full(
            (batch_size,), current_len - 1, dtype=torch.long, device=device
        )
        steer_mask = make_steer_mask(
            current_len,
            prompt_width,
            provisional_terminal,
            steer_position=steer_position,
            device=device,
        )
        fwd_hooks = [
            (
                f"blocks.{inject_layer}.hook_resid_post",
                _make_steering_hook(steering, steer_mask, inject_layer),
            )
        ]
        logits = (
            model.run_with_hooks(
                full_ids,
                fwd_hooks=fwd_hooks,
                attention_mask=full_attention,
                return_type="logits",
            )[:, -1, :].float()
            / temperature
        )
        next_token = torch.distributions.Categorical(logits=logits).sample()
        active = ~finished
        next_token = torch.where(active, next_token, torch.full_like(next_token, pad_id))
        action_tokens.append(next_token)
        action_masks.append(active)
        full_ids = torch.cat([full_ids, next_token[:, None]], dim=1)
        full_attention = torch.cat(
            [full_attention, active[:, None].to(full_attention.dtype)], dim=1
        )
        emitted = active & next_token.eq(eos_id)
        terminal_positions = torch.where(
            emitted, torch.full_like(terminal_positions, prompt_width + step), terminal_positions
        )
        finished = finished | emitted
        if finished.all():
            break

    completion_ids = torch.stack(action_tokens, dim=1)
    action_mask = torch.stack(action_masks, dim=1)
    truncated = ~finished

    if truncated.any():
        forced_eos = torch.full((batch_size, 1), eos_id, dtype=torch.long, device=device)
        completion_ids = torch.cat([completion_ids, forced_eos], dim=1)
        forced_action = torch.zeros((batch_size, 1), dtype=torch.bool, device=device)
        action_mask = torch.cat([action_mask, forced_action], dim=1)
        terminal_positions = torch.where(
            truncated,
            torch.full_like(terminal_positions, prompt_width + completion_ids.shape[1] - 1),
            terminal_positions,
        )
    positions = torch.arange(action_mask.shape[1], device=device)[None, :] + prompt_width
    transcript_mask = positions <= terminal_positions[:, None]
    full_len = prompt_width + completion_ids.shape[1]
    steer_mask = make_steer_mask(
        full_len,
        prompt_width,
        terminal_positions,
        steer_position=steer_position,
        device=device,
    )
    return TrajectoryBatch(
        prompt_ids=prompt_ids,
        prompt_attention_mask=prompt_attention_mask,
        completion_ids=completion_ids,
        action_mask=action_mask,
        transcript_mask=transcript_mask,
        steer_mask=steer_mask,
        terminal_positions=terminal_positions,
        truncated=truncated,
    )

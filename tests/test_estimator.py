"""Ground-truth checks on the gradient estimator

Vocabulary and horizon are small enough to enumerate every completion, so
E_{y~q}[.] and its exact gradient are available in closed form and the Monte
Carlo estimator used by the training loop can be scored against them
"""

from __future__ import annotations

import itertools

import pytest
import torch
import torch.nn.functional as F
from tiny_stack import EOS, INJECT_LAYER, PROBE_LAYER, VOCAB, build_stack

from non_linear_steering.objective import constrained_loss, replay_and_score
from non_linear_steering.trajectory import (
    TrajectoryBatch,
    _make_steering_hook,
    make_steer_mask,
)

pytest.importorskip("transformer_lens")

PROMPT = [1, 2, 3]
HORIZON = 3
PROMPT_WIDTH = len(PROMPT)
STEER_POSITION = "prediction-state"


def make_batch(completions: torch.Tensor) -> TrajectoryBatch:
    """A fixed-length rollout: HORIZON sampled actions plus one forced EOS

    This is the shape sample_rollout produces for a truncated trajectory, which is
    every trajectory here because no completion is allowed to stop early
    """
    rows = completions.shape[0]
    prompt_ids = torch.tensor([PROMPT]).repeat(rows, 1)
    forced_eos = torch.full((rows, 1), EOS, dtype=torch.long)
    completion_ids = torch.cat([completions, forced_eos], dim=1)
    action_mask = torch.cat(
        [
            torch.ones(rows, HORIZON, dtype=torch.bool),
            torch.zeros(rows, 1, dtype=torch.bool),
        ],
        dim=1,
    )
    terminal = torch.full((rows,), PROMPT_WIDTH + HORIZON, dtype=torch.long)
    return TrajectoryBatch(
        prompt_ids=prompt_ids,
        prompt_attention_mask=torch.ones_like(prompt_ids),
        completion_ids=completion_ids,
        action_mask=action_mask,
        transcript_mask=torch.ones(rows, HORIZON + 1, dtype=torch.bool),
        steer_mask=make_steer_mask(
            PROMPT_WIDTH + HORIZON + 1,
            PROMPT_WIDTH,
            terminal,
            steer_position=STEER_POSITION,
            device=prompt_ids.device,
        ),
        terminal_positions=terminal,
        truncated=torch.ones(rows, dtype=torch.bool),
    )


def all_completions() -> torch.Tensor:
    return torch.tensor(list(itertools.product(range(VOCAB), repeat=HORIZON)))


@torch.no_grad()
def sample_completions(model, steering, rows, generator) -> torch.Tensor:
    """Same protocol as make_batch: HORIZON actions, no early stopping"""
    ids = torch.tensor([PROMPT]).repeat(rows, 1)
    for _ in range(HORIZON):
        width = ids.shape[1]
        attention = torch.ones_like(ids)
        steer_mask = make_steer_mask(
            width,
            PROMPT_WIDTH,
            torch.full((rows,), width - 1, dtype=torch.long),
            steer_position=STEER_POSITION,
            device=ids.device,
        )
        fwd_hooks = [
            (
                f"blocks.{INJECT_LAYER}.hook_resid_post",
                _make_steering_hook(steering, steer_mask, INJECT_LAYER),
            )
        ]
        logits = model.run_with_hooks(
            ids,
            fwd_hooks=fwd_hooks,
            attention_mask=attention,
            return_type="logits",
        )[:, -1, :].float()
        ids = torch.cat([ids, torch.multinomial(logits.softmax(-1), 1, generator=generator)], dim=1)
    return ids[:, PROMPT_WIDTH:]


def flat_grad(params) -> torch.Tensor:
    return torch.cat(
        [(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in params]
    )


def zero_grads(params) -> None:
    for param in params:
        param.grad = None


def exact_expectations(model, probe, steering):
    batch = make_batch(all_completions())
    values = replay_and_score(
        model,
        probe,
        steering,
        batch,
        inject_layer=INJECT_LAYER,
        probe_layer=PROBE_LAYER,
        temperature=1.0,
    )
    probability = values.seq_log_prob.exp()
    return probability, values


def test_enumerated_completions_are_a_distribution(stack):
    model, probe, steering, _ = stack
    probability, _ = exact_expectations(model, probe, steering)
    assert probability.shape[0] == VOCAB**HORIZON
    assert abs(probability.sum().item() - 1.0) < 1e-4


def test_teacher_forced_replay_matches_incremental_decode():
    """The intervention must be position-local, or the replayed log q is not the
    probability that produced the stored tokens"""
    _, _, steering, hooked = build_stack()
    batch = make_batch(all_completions()[:8])
    full_ids = torch.cat([batch.prompt_ids, batch.completion_ids], dim=1)
    attention = torch.ones_like(full_ids)
    with torch.no_grad():
        with hooked.session(steering=steering, steer_mask=batch.steer_mask):
            teacher_forced = hooked.forward(full_ids, attention_mask=attention)
        worst = 0.0
        for width in range(PROMPT_WIDTH, full_ids.shape[1]):
            prefix = full_ids[:, :width]
            prefix_attention = torch.ones_like(prefix)
            steer_mask = make_steer_mask(
                width,
                PROMPT_WIDTH,
                torch.full((prefix.shape[0],), width - 1, dtype=torch.long),
                steer_position=STEER_POSITION,
                device=prefix.device,
            )
            with hooked.session(steering=steering, steer_mask=steer_mask):
                incremental = hooked.forward(prefix, attention_mask=prefix_attention)
            worst = max(
                worst, (incremental[:, -1] - teacher_forced[:, width - 1]).abs().max().item()
            )
    assert worst < 1e-4, worst


def exact_gradient(which: str):
    model, probe, steering, _ = build_stack()
    params = list(steering.parameters())
    probability, values = exact_expectations(model, probe, steering)
    target = values.score if which == "score" else values.seq_kl
    expectation = (probability * target).sum()
    zero_grads(params)
    expectation.backward()
    gradient = flat_grad(params).clone()
    zero_grads(params)
    return gradient, expectation.item()


def monte_carlo_gradient(which: str, *, total=4096, chunk=1024, pathwise=True, reinforce=True):
    model, probe, steering, _ = build_stack()
    params = list(steering.parameters())
    generator = torch.Generator().manual_seed(1234)
    zero_grads(params)
    for _ in range(total // chunk):
        completions = sample_completions(model, steering, chunk, generator)
        values = replay_and_score(
            model,
            probe,
            steering,
            make_batch(completions),
            inject_layer=INJECT_LAYER,
            probe_layer=PROBE_LAYER,
            temperature=1.0,
        )
        target = values.score if which == "score" else values.seq_kl
        surrogate = target.new_zeros(())
        if pathwise:
            surrogate = surrogate + target.mean()
        if reinforce:
            coefficient = target.detach()
            surrogate = (
                surrogate + ((coefficient - coefficient.mean()) * values.seq_log_prob).mean()
            )
        (surrogate * (chunk / total)).backward()
    gradient = flat_grad(params).clone()
    zero_grads(params)
    return gradient


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a, b, dim=0).item()


def test_both_channels_recover_the_exact_score_gradient():
    reference, _ = exact_gradient("score")
    assert reference.norm() > 1e-3
    assert cosine(monte_carlo_gradient("score"), reference) > 0.99


def test_both_channels_recover_the_exact_kl_gradient():
    """grad E[C] needs the score-function term too: the prefixes are sampled from q"""
    reference, _ = exact_gradient("kl")
    assert reference.norm() > 1e-6
    assert cosine(monte_carlo_gradient("kl"), reference) > 0.99


def test_score_function_only_is_not_the_full_gradient():
    """Guards the direct channel: if replay silently ran under no_grad the pathwise
    term would vanish and this estimator would look correct"""
    reference, _ = exact_gradient("score")
    assert cosine(monte_carlo_gradient("score", pathwise=False), reference) < 0.5


def test_constrained_loss_gradient_matches_the_lagrangian(stack):
    """loss must differentiate to -grad(F - beta K) with the two-term estimator"""
    model, probe, steering, _ = stack
    params = list(steering.parameters())
    beta, baseline = 0.7, 0.1
    batch = make_batch(all_completions()[:16])
    values = replay_and_score(
        model,
        probe,
        steering,
        batch,
        inject_layer=INJECT_LAYER,
        probe_layer=PROBE_LAYER,
        temperature=1.0,
    )

    zero_grads(params)
    constrained_loss(values, beta=beta, baseline=baseline).backward(retain_graph=True)
    produced = flat_grad(params).clone()

    zero_grads(params)
    direct = (-values.score + beta * values.seq_kl).mean()
    advantage = (values.score - beta * values.seq_kl).detach() - baseline
    (direct - (advantage * values.seq_log_prob).mean()).backward()
    expected = flat_grad(params).clone()
    zero_grads(params)

    assert torch.allclose(produced, expected, atol=1e-6)

from __future__ import annotations

import torch
from tiny_stack import EOS, INJECT_LAYER, PROBE_LAYER, WhitespaceTokenizer, build_stack

from non_linear_steering.objective import replay_and_score
from non_linear_steering.trajectory import (
    TrajectoryBatch,
    make_steer_mask,
    padding_side,
    sample_rollout,
)

HORIZON = 6
STEER_POSITION = "prediction-state"


def constant_head(model, token: int) -> None:
    """Make one token certain so the terminal bookkeeping can be asserted exactly."""
    with torch.no_grad():
        model.unembed.W_U.zero_()
        model.unembed.b_U.zero_()
        model.unembed.b_U[token] = 30.0


def roll(model, steering, tokenizer, prompts, seed=0, horizon=HORIZON):
    torch.manual_seed(seed)
    return sample_rollout(
        model,
        tokenizer,
        steering,
        prompts,
        inject_layer=INJECT_LAYER,
        horizon=horizon,
        temperature=1.0,
        steer_position=STEER_POSITION,
    )


def assert_bookkeeping(batch: TrajectoryBatch, horizon: int) -> None:
    full_ids = torch.cat([batch.prompt_ids, batch.completion_ids], dim=1)
    prompt_width = batch.prompt_ids.shape[1]
    for row in range(full_ids.shape[0]):
        terminal = int(batch.terminal_positions[row])
        slots = batch.completion_ids.shape[1]
        assert int(full_ids[row, terminal]) == EOS
        expected_transcript = torch.tensor(
            [prompt_width + slot <= terminal for slot in range(slots)]
        )
        assert torch.equal(batch.transcript_mask[row].cpu(), expected_transcript)
        expected_steer = torch.tensor(
            [prompt_width - 1 <= pos <= terminal for pos in range(full_ids.shape[1])]
        )
        assert torch.equal(batch.steer_mask[row].cpu(), expected_steer)
        actions = int(batch.action_mask[row].sum())
        if bool(batch.truncated[row]):
            # the appended end-of-turn token is a transcript token, never an action
            assert actions == horizon
            assert not bool(batch.action_mask[row, -1])
            assert terminal == prompt_width + slots - 1
        else:
            assert actions == terminal - prompt_width + 1
            assert torch.equal(
                batch.action_mask[row].cpu(),
                torch.tensor([prompt_width + slot <= terminal for slot in range(slots)]),
            )


def test_natural_eos_is_an_action_and_sets_the_terminal():
    model, _, steering, _ = build_stack()
    constant_head(model, EOS)
    batch = roll(model, steering, WhitespaceTokenizer(), ["1 2 3", "1 2 3 4"])
    assert not batch.truncated.any()
    assert batch.completion_ids.shape[1] == 1
    assert batch.action_mask.all()
    assert_bookkeeping(batch, HORIZON)


def test_truncated_rollout_appends_a_non_action_eos():
    model, _, steering, _ = build_stack()
    constant_head(model, 0)
    batch = roll(model, steering, WhitespaceTokenizer(), ["1 2 3", "1 2 3 4"])
    assert batch.truncated.all()
    assert batch.completion_ids.shape[1] == HORIZON + 1
    assert int(batch.action_mask.sum()) == 2 * HORIZON
    assert_bookkeeping(batch, HORIZON)


def test_mixed_termination_bookkeeping_holds_across_seeds():
    model, _, steering, _ = build_stack()
    tokenizer = WhitespaceTokenizer()
    seen = set()
    for seed in range(12):
        batch = roll(model, steering, tokenizer, ["1 2 3", "1 2 3 4"], seed=seed)
        assert_bookkeeping(batch, HORIZON)
        seen.update(bool(flag) for flag in batch.truncated)
    assert seen == {True, False}, "seeds did not cover both termination paths"


def test_rollout_left_pads_and_restores_the_tokenizer():
    model, _, steering, _ = build_stack()
    tokenizer = WhitespaceTokenizer(padding_side="right")
    batch = roll(model, steering, tokenizer, ["1 2", "1 2 3 4"])
    assert tokenizer.padding_side == "right"
    # left padding is what makes index -1 and prompt_width - 1 the real final
    # prompt position for every row
    assert batch.prompt_attention_mask[0].tolist() == [0, 0, 1, 1]
    assert batch.prompt_ids[0, -2:].tolist() == [1, 2]


def test_padding_side_context_restores_on_error():
    tokenizer = WhitespaceTokenizer(padding_side="right")
    try:
        with padding_side(tokenizer, "left"):
            assert tokenizer.padding_side == "left"
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert tokenizer.padding_side == "right"


def unpad_row(batch: TrajectoryBatch, row: int) -> TrajectoryBatch:
    """Rebuild a single-row batch with the prompt padding removed."""
    keep = batch.prompt_attention_mask[row].bool()
    prompt_ids = batch.prompt_ids[row][keep][None]
    pads = int((~keep).sum())
    terminal = batch.terminal_positions[row][None] - pads
    prompt_width = prompt_ids.shape[1]
    slots = batch.completion_ids.shape[1]
    return TrajectoryBatch(
        prompt_ids=prompt_ids,
        prompt_attention_mask=torch.ones_like(prompt_ids),
        completion_ids=batch.completion_ids[row][None],
        action_mask=batch.action_mask[row][None],
        transcript_mask=batch.transcript_mask[row][None],
        steer_mask=make_steer_mask(
            prompt_width + slots,
            prompt_width,
            terminal,
            steer_position=STEER_POSITION,
            device=prompt_ids.device,
        ),
        terminal_positions=terminal,
        truncated=batch.truncated[row][None],
    )


def test_replay_values_do_not_depend_on_prompt_padding():
    """A short prompt batched with a long one must score exactly as it does alone.

    This is the check that fails when prompts are right-padded: the sampler then
    reads logits at a pad position and the transcript is conditioned on padding.
    """
    model, probe, steering, _ = build_stack()
    batch = roll(model, steering, WhitespaceTokenizer(), ["1 2", "1 2 3 4 5"], seed=5)
    assert batch.prompt_attention_mask[0].sum() < batch.prompt_attention_mask[1].sum()
    batched = replay_and_score(
        model,
        probe,
        steering,
        batch,
        inject_layer=INJECT_LAYER,
        probe_layer=PROBE_LAYER,
        temperature=1.0,
    )
    for row in range(2):
        alone = replay_and_score(
            model,
            probe,
            steering,
            unpad_row(batch, row),
            inject_layer=INJECT_LAYER,
            probe_layer=PROBE_LAYER,
            temperature=1.0,
        )
        assert torch.allclose(alone.score, batched.score[row][None], atol=1e-4)
        assert torch.allclose(alone.seq_log_prob, batched.seq_log_prob[row][None], atol=1e-4)
        assert torch.allclose(alone.seq_kl, batched.seq_kl[row][None], atol=1e-4)


def test_rollout_log_prob_matches_the_sampling_distribution():
    """Replay must reproduce the probabilities the sampler actually used."""
    model, probe, steering, hooked = build_stack()
    tokenizer = WhitespaceTokenizer()
    batch = roll(model, steering, tokenizer, ["1 2 3"], seed=7)
    values = replay_and_score(
        model,
        probe,
        steering,
        batch,
        inject_layer=INJECT_LAYER,
        probe_layer=PROBE_LAYER,
        temperature=1.0,
    )

    prompt_width = batch.prompt_ids.shape[1]
    ids = batch.prompt_ids
    total = 0.0
    with torch.no_grad():
        for slot in range(batch.completion_ids.shape[1]):
            if not bool(batch.action_mask[0, slot]):
                continue
            attention = torch.ones_like(ids)
            steer_mask = make_steer_mask(
                ids.shape[1],
                prompt_width,
                torch.tensor([ids.shape[1] - 1]),
                steer_position=STEER_POSITION,
                device=ids.device,
            )
            with hooked.session(steering=steering, steer_mask=steer_mask):
                logits = hooked.forward(ids, attention_mask=attention)[:, -1, :].float()
            token = batch.completion_ids[0, slot]
            total += float(logits.log_softmax(-1)[0, token])
            ids = torch.cat([ids, token[None, None]], dim=1)
    assert abs(total - float(values.seq_log_prob[0].detach())) < 1e-3


def test_steer_mask_width_mismatch_is_rejected():
    _, _, steering, hooked = build_stack()
    ids = torch.tensor([[1, 2, 3]])
    bad_mask = torch.ones(1, 5, dtype=torch.bool)
    with hooked.session(steering=steering, steer_mask=bad_mask):
        try:
            hooked.forward(ids)
        except RuntimeError as error:
            assert "steer_mask width" in str(error)
        else:
            raise AssertionError("expected a steer_mask width mismatch to be rejected")

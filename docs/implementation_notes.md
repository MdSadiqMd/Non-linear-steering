# Implementation Notes

All model loading and hook management goes through **TransformerLens**:

- Models load via `TransformerBridge.boot_transformers` (HuggingFace names or local
  checkpoint directories) with compatibility mode enabled, which restores the
  `HookedTransformer` hook names and load-time weight processing.
- The CLI resolves `--device auto` (the default) to CUDA when available, else CPU,
  and resolves `--dtype auto` to `float16` **before** calling the bridge, which
  rejects an `"auto"` device string.
- Steering injects at `blocks.{inject-layer}.hook_resid_post` and the probe reads at
  `blocks.{probe-layer}.hook_resid_post` with `run_with_hooks`.

The source code follows these correctness conditions from the [[docs/problem_statement.md]]:

- The language model and probe are frozen during steering training.
- Only the nonlinear intervention parameters are optimized.
- Rollout uses discrete on-policy sampling with gradients disabled.
- Replay uses teacher forcing with autograd enabled.
- The probe score supplies the direct activation gradient.
- The selected-token log probability supplies the behavior gradient.
- Forward KL is computed as full-vocabulary conditional KL along sampled prefixes.
- The KL expectation also receives a score-function term through the same Lagrangian loss.
- Natural EOS tokens are actions; forced EOS tokens are transcript tokens only.
- Padding, repeated tokens after termination, and forced EOS are excluded from action log probability and KL.
- The first completion token is steered by default using `--steer-position prediction-state`, which applies steering at the final prompt prediction state.
- The baseline and the dual multiplier are both updated after the current gradient is formed, so the baseline never depends on the sample it scores and $\beta$ stays the multiplier the loss actually used.

The implementation intentionally uses no-cache teacher-forced replay for the optimization pass. This is slower for a 20B model but avoids base/steered KV-cache contamination and keeps the probability calculation faithful to the objective.

## Padding And Position Contract

Rollout reads next-token logits at index `-1` and replay reads the first prediction state at index `prompt_width - 1`. Both are the true final prompt position only when prompts are **left**-padded, so `sample_rollout` pins `tokenizer.padding_side = "left"` for the duration of the call and restores it afterwards. Probe training pins `"right"` instead, because it reads the label at `attention_mask.sum() - 1`.

TransformerLens derives positional ids from the attention mask (`cumsum - 1`) internally, so a padded prefix never consumes real position slots and the distance between prompt and completion is independent of the longest prompt in the batch.

`CausalProbe` builds one merged causal-plus-padding attention mask and keeps the diagonal open. A left-padded position has no visible key under a causal mask, and a fully masked softmax row produces `NaN` that spreads to every later position, including the terminal read. Keeping the diagonal open is a no-op for unpadded positions and makes the probe score identical whether padding sits on the left, on the right, or between prompt and completion.

## Hook Order And The Activation Channel

Steering and probe capture share one forward hook per decoder block, and the hook injects before it captures. So `--inject-layer == --probe-layer` still leaves the direct activation path open, and only `--inject-layer > --probe-layer` closes it. The steering command warns when the injection sits downstream of the probe read, because in that configuration the probe score is constant in the steering parameters and only the score-function term trains.

## Verification

`uv run pytest` runs the checks that back the derivation, on a tiny randomly initialized Llama where the full completion support can be enumerated:

- Enumerated completions form a distribution: the replayed sequence probabilities sum to 1.
- Teacher-forced replay logits match incremental decode to ~1e-8, so the intervention is position-local and the replayed $\log q$ is the probability that produced the stored tokens.
- The direct-plus-score-function estimator recovers the exact enumerated $\nabla \mathbb{E}[S]$ and $\nabla \mathbb{E}[C]$ (cosine > 0.99); the score-function term alone does not.
- `constrained_loss` differentiates to the Lagrangian gradient.
- Rollout bookkeeping: the terminal position always holds an EOS token, the forced EOS is in `transcript_mask` but not `action_mask`, and a short prompt batched with a long one scores exactly as it does alone.
- The full `preflight -> train-probe -> train-steering` CLI path runs against a locally saved tiny checkpoint.

Beyond the tests, the pipeline was exercised end-to-end on the cached `gpt2`: probe
training at layer 6 saves a checkpoint, and steering training at inject-layer 4 /
probe-layer 6 keeps the observed KL below `epsilon` (so the dual stays at 0) while the
baseline EMA drifts and the forced-EOS path activates on truncated rollouts.
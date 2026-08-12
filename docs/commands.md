# Commands

Use the local GPT-OSS path supplied by the user:

```bash
MODEL=/Users/sadiq/.cache/huggingface/hub/models--openai--gpt-oss-20b
```

## 1. Install Runtime Dependencies

```bash
cd /Users/sadiq/Developer/non-linear-steering
uv sync
```

## 2. Check The Local Model

```bash
cd /Users/sadiq/Developer/non-linear-steering
uv run nls preflight --model /Users/sadiq/.cache/huggingface/hub/models--openai--gpt-oss-20b
```

Current expected result: `NOT TRAINABLE YET`, because the specified cache path contains tokenizer/config files but no model weight shards.

## 3. Run The Checks

These need no model weights; they build a tiny randomly initialized Llama in-process.

```bash
cd /Users/sadiq/Developer/non-linear-steering
uv run pytest
```

## 4. Train The Probe

This requires `data/probe_train.jsonl` with `prompt`, `completion`, and `label` fields.

```bash
cd /Users/sadiq/Developer/non-linear-steering
uv run nls train-probe \
  --model /Users/sadiq/.cache/huggingface/hub/models--openai--gpt-oss-20b \
  --dataset data/probe_train.jsonl \
  --probe-layer 12 \
  --batch-size 1 \
  --epochs 1 \
  --seed 0 \
  --output checkpoints/probe.pt
```

## 5. Train Nonlinear Steering

This requires the saved probe checkpoint and `data/prompts.jsonl`.

```bash
cd /Users/sadiq/Developer/non-linear-steering
uv run nls train-steering \
  --model /Users/sadiq/.cache/huggingface/hub/models--openai--gpt-oss-20b \
  --prompts data/prompts.jsonl \
  --probe checkpoints/probe.pt \
  --inject-layer 8 \
  --probe-layer 12 \
  --steer-position prediction-state \
  --horizon 64 \
  --temperature 1.0 \
  --epsilon 0.1 \
  --radius 0.5 \
  --rank 32 \
  --batch-size 1 \
  --device cpu \
  --steps 100 \
  --seed 0 \
  --output checkpoints/steering.pt
```

Steering runs on TransformerLens: the model loads through
`TransformerBridge.boot_transformers` and the intervention hooks the residual stream at
`blocks.{inject-layer}.hook_resid_post`. `--device` selects the compute device (default:
CUDA when available, else CPU).

`--inject-layer` must be at or before `--probe-layer`, otherwise the probe score has no
gradient in the steering parameters and only the score-function term trains. The command
warns rather than failing, because shielded injection is a valid separate experiment.

`--batch-size` above 1 is supported: prompts are left-padded and every per-row read is
indexed by that row's own terminal position.

## Why Probe Training Is Mandatory

The base GPT-OSS model is not a lie detector. The steering objective needs $s_\phi$, a differentiable probe score. Therefore the correct command order is:

```bash
uv run nls preflight --model /Users/sadiq/.cache/huggingface/hub/models--openai--gpt-oss-20b
uv run nls train-probe --model /Users/sadiq/.cache/huggingface/hub/models--openai--gpt-oss-20b --dataset data/probe_train.jsonl --probe-layer 12 --output checkpoints/probe.pt
uv run nls train-steering --model /Users/sadiq/.cache/huggingface/hub/models--openai--gpt-oss-20b --prompts data/prompts.jsonl --probe checkpoints/probe.pt --inject-layer 8 --probe-layer 12 --output checkpoints/steering.pt
```

## Verified Runs On The Cached gpt2

The same commands were exercised end-to-end against the locally cached `gpt2` (12 layers, model weights present):

- `preflight --model gpt2` reports `status: HuggingFace model name (will download)`; pointed at the HF cache directory
  (`/Users/sadiq/.cache/huggingface/hub/models--openai-community--gpt2`) it reports `status: OK`.
- `train-probe --probe-layer 6` trains one epoch over the 8 example transcripts in ~10 s on CPU and saves the checkpoint.
- `train-steering --inject-layer 4 --probe-layer 6 --horizon 16 --steps 10` completes in ~3.5 min on CPU (~19 s per step); the full `--steps 100` run takes roughly 30-40 min on CPU.

Layer mapping for the two documented models:

| Model | Layers | `--probe-layer` | `--inject-layer` |
|-------|--------|----------------|------------------|
| gpt2 (cached) | 12 | 6 | 4|
| GPT-OSS 20B (local cache) | 24 | 12 | 8 |

Both training commands accept `--device auto` (the default): it resolves to CUDA when available, else CPU, before the model bridge boots.
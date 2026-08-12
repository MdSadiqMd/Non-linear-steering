# Data Formats

## Probe Training Data

`train-probe` requires labeled transcripts. Each JSONL row must contain:

```json
{"prompt": "...", "completion": "...", "label": 1}
```

`label` must be `0` or `1`, where `1` means the transcript should receive a high lie/deception probe score.

Example:

```jsonl
{"prompt":"User: Did you alter the logs?\nAssistant:","completion":"No, I did not alter anything.","label":1}
{"prompt":"User: Did you alter the logs?\nAssistant:","completion":"Yes, I changed them yesterday.","label":0}
```

The code appends the tokenizer EOS token before reading the probe label position, so the
probe learns its readout at the same end-of-turn position the steering stage reads. Rows are
shuffled each epoch from `--seed`, so file order does not matter.

## Steering Prompt Data

`train-steering` requires prompts only. Each JSONL row must contain:

```json
{"prompt": "..."}
```

Example:

```jsonl
{"prompt":"User: Did you alter the logs?\nAssistant:"}
{"prompt":"User: Are you hiding anything from me?\nAssistant:"}
```

The steering stage samples completions from the steered policy and optimizes the saved probe score under the KL constraint.
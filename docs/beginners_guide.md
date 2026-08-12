# Beginner's Guide to Non-linear Steering

This guide explains what this project does, why it matters, and how it works — assuming you're new to mechanistic interpretability.

## What Problem Are We Solving?

Large language models (LLMs) like GPT can generate text that sounds helpful but might be deceptive. We want to:

1. **Detect** when the model is being deceptive (using a "probe")
2. **Steer** the model to be more deceptive (for research purposes)
3. **Control** how much we change the model's behavior (using a "KL constraint")

This is useful for AI safety research: if we can make a model more deceptive in a controlled way, we can study what deception looks like internally and build better detectors.

## Key Concepts

### 1. Residual Stream

When a language model processes text, information flows through layers. At each layer, there's a "residual stream" — a vector representation of everything the model knows so far. Think of it as the model's working memory at that point.

```
Input → [Layer 1] → residual → [Layer 2] → residual → ... → [Layer N] → Output
                       ↑                       ↑
                   We can read            We can modify
                   this (probe)           this (steering)
```

### 2. Probe

A probe is a small neural network that reads the residual stream and predicts something about the text. In our case, it predicts: "Is this response deceptive?"

We train the probe on labeled examples:
- "No, I didn't change anything" (deceptive = 1)
- "Yes, I modified the logs" (honest = 0)

After training, the probe can score any text on how deceptive it seems.

### 3. Steering Intervention

Instead of just reading the residual stream, we can also modify it. A "steering intervention" adds a small vector to the residual stream to push the model's behavior in a direction we want.

$$
\begin{aligned}
\text{Original residual: } & h \\
\text{Steering vector: } & \delta(h) \\
\text{Modified residual: } & h + \delta(h)
\end{aligned}
$$

Our steering is "nonlinear" — it depends on what the model is currently thinking ($h$), not just a fixed direction.

### 4. KL Divergence Constraint

If we steer too aggressively, the model might output gibberish. The KL divergence measures how different the steered model's outputs are from the original model.

We set a budget ($\epsilon = 0.1$ by default): "The steered model can be different, but not too different."

$$
D_{\text{KL}}(\pi_{\text{steered}} \| \pi_{\text{original}}) \le \epsilon
$$

This keeps the outputs coherent while still allowing meaningful behavioral changes.

## How It Works

### Phase 1: Train the Probe

```
┌─────────────────────────────────────────────────────────┐
│  INPUT: Labeled transcripts                             │
│  "Did you alter the logs?" → "No, I didn't" (label=1)   │
│  "Did you alter the logs?" → "Yes, I did"   (label=0)   │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  FROZEN LLM: Run text through the model                 │
│  Extract residual stream at layer 6                     │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  PROBE: Small transformer that reads residuals          │
│  Learns to predict: deceptive (1) vs honest (0)         │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  OUTPUT: Saved probe checkpoint                         │
│  Can now score any text on "deceptiveness"              │
└─────────────────────────────────────────────────────────┘
```

### Phase 2: Train the Steering

```
┌─────────────────────────────────────────────────────────┐
│  INPUT: Prompts like "Did you alter the logs?"          │
└─────────────────────────────────────────────────────────┘
                           │
            ┌──────────────┴──────────────┐
            ▼                             ▼
┌─────────────────────┐       ┌─────────────────────────┐
│  ROLLOUT (no grad)  │       │  REPLAY (with grad)     │
│  Sample completions │       │  Compute exact probs    │
│  from steered model │       │  and probe scores       │
└─────────────────────┘       └─────────────────────────┘
            │                             │
            └──────────────┬──────────────┘
                           ▼
┌─────────────────────────────────────────────────────────┐
│  LOSS = -probe_score + β × KL_divergence                │
│                                                         │
│  Two gradient paths:                                    │
│  1. Direct: steering → residual → probe → score         │
│  2. Behavior: steering → what tokens are sampled        │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  UPDATE: Optimize steering network parameters           │
│  Also update β (Lagrange multiplier) to enforce KL ≤ ε  │
└─────────────────────────────────────────────────────────┘
```

## The Two Gradient Paths

This is the trickiest part. When we sample text from a language model, we pick tokens randomly. You can't backpropagate through random sampling. So how do we train?

**Solution: Two-pass approach**

1. **Rollout**: Sample a completion with gradients disabled. Store the tokens.
2. **Replay**: Run the same tokens again with gradients enabled. Now everything is differentiable.

But there's a subtlety: changing the steering changes *which* tokens get sampled. This is captured by the "score function" gradient (also called REINFORCE):

$$
\nabla \mathbb{E}[\text{score}] = \mathbb{E}[\nabla \text{score}] + \mathbb{E}[\text{score} \times \nabla \log p(\text{tokens})]
$$

Both paths are needed for an unbiased gradient.

## What the Code Produces

After training, you have:

1. **Probe checkpoint** (`probe.pt`): A detector that scores text on the target property
2. **Steering checkpoint** (`steering.pt`): A network that modifies the residual stream

You can use these to:
- Generate text from the steered model
- Compare probe scores with/without steering
- Study what the steering intervention does to internal representations

## Example Workflow

```bash
# 1. Check model is accessible
uv run nls preflight --model gpt2

# 2. Train probe on labeled data
uv run nls train-probe \
  --model gpt2 \
  --dataset data/probe_train.jsonl \
  --probe-layer 6 \
  --output checkpoints/probe.pt

# 3. Train steering to maximize probe score under KL constraint
uv run nls train-steering \
  --model gpt2 \
  --prompts data/prompts.jsonl \
  --probe checkpoints/probe.pt \
  --inject-layer 4 \
  --probe-layer 6 \
  --epsilon 0.1 \
  --output checkpoints/steering.pt

# 4. Use the trained steering for inference
```

## Glossary

| Term | Meaning |
|------|---------|
| **Residual stream** | The vector representation at a layer in the transformer |
| **Probe** | A small network that reads the residual stream and predicts a property |
| **Steering** | Modifying the residual stream to change model behavior |
| **KL divergence** | A measure of how different two probability distributions are |
| **Rollout** | Sampling tokens from the model |
| **Replay** | Re-running the same tokens with gradients enabled |
| **REINFORCE** | A technique for computing gradients through sampling |
| **Lagrange multiplier (β)** | A parameter that enforces the KL constraint |

## Further Reading

- [[docs/math.md]] — Full mathematical derivation
- [[docs/implementation_notes.md]] — Code architecture details
- Anthropic's work on probes: "Discovering Language Model Behaviors with Model-Written Evaluations"
- "Steering Language Models With Activation Engineering" (Turner et al.)
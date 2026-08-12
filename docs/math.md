# Mathematical Implementation

Sampling token IDs is discrete, so gradients do not pass through multinomial sampling. The
implementation therefore uses two passes:

1. **Rollout**: sample $y \sim q_\theta(\cdot | x)$ with steering active and gradients disabled.
2. **Replay**: teacher-force the same fixed transcript $(x, y)$ with steering active and
   autograd enabled.

For a fixed sampled transcript, both terms are differentiable in the steering parameters
$\theta$:

$$
S_\theta(x, y), \qquad \log q_\theta(y | x) = \sum_t \log q_\theta(y_t | x, y_{<t})
$$

## Score Gradient

The exact gradient of the expected probe score is

$$
\nabla_\theta \mathbb{E}[S_\theta] = \mathbb{E}\left[\nabla_\theta S_\theta + S_\theta \nabla_\theta \log q_\theta(y | x)\right].
$$

The first term is the **activation path** (direct / pathwise):

$$
\theta \to \delta_\theta \to h_\theta \to \text{frozen probe} \to S_\theta.
$$

This path is open only when the intervention layer is at or before the probe read layer. When
the intervention sits downstream, the probe never sees the steered residuals for a fixed token
sequence and its score is constant in $\theta$, so only the second term trains.

The second term is the **behavior path** (score-function / REINFORCE). It accounts for changing
which discrete completion is sampled. The coefficient $S_\theta$ is detached: the derivative of
$S_\theta \log q_\theta$ with respect to $\theta$ already counts the direct term, and counting it
again would double it.

## Forward KL Constraint

At each sampled prefix $u_t = (x, y_{<t})$, the full-vocabulary conditional forward KL is

$$
k_t = \sum_v q_\theta(v | u_t) \left[\log q_\theta(v | u_t) - \log p_0(v | u_t)\right].
$$

The sequence-level forward KL is an expectation under $q_\theta$ because the prefixes are
themselves sampled from the steered policy:

$$
K(\theta) = \mathbb{E}_{y \sim q_\theta}[C_\theta(y)], \qquad
C_\theta(y) = \sum_t k_t.
$$

Its gradient also has two terms — the direct term from differentiating the per-position KL
and the score-function term from differentiating the distribution over prefixes:

$$
\nabla_\theta K = \mathbb{E}\left[\nabla_\theta C_\theta + C_\theta \nabla_\theta \log q_\theta(y | x)\right].
$$

Computing only $\nabla_\theta C_\theta$ treats the sampled prefixes as fixed and is biased.
Including the score-function term is what makes the KL constraint exact, not just a heuristic
penalty.

## Lagrangian Loss

The constrained problem is

$$
\max_\theta \mathbb{E}[S_\theta] \quad \text{subject to} \quad K(\theta) \le \epsilon.
$$

For a fixed nonnegative multiplier $\beta$, the primal Lagrangian is $F - \beta (K - \epsilon)$.
A minimization surrogate whose expected gradient is exactly $-\nabla_\theta(F - \beta K)$ is

$$
\begin{aligned}
\mathcal{L}_{\text{direct}} &= \text{mean}(-S_\theta + \beta C_\theta) \\
\mathcal{L}_{\text{policy}} &= -\text{mean}\left(\text{stopgrad}(S_\theta - \beta C_\theta - \text{baseline}) \log q_\theta(y | x)\right) \\
\mathcal{L} &= \mathcal{L}_{\text{direct}} + \mathcal{L}_{\text{policy}}
\end{aligned}
$$

The `stopgrad` (detach) is essential: without it, the direct derivatives of $S_\theta$ and
$C_\theta$ are counted a second time through the policy coefficient.

The baseline must be independent of the current sample's action given the prompt. A moving
average from previous batches or a learned value $b(x)$ is valid. The current batch mean is
not exactly unbiased for finite batches when each sample contributes to its own baseline; it
multiplies the expected score-function gradient by $(B-1)/B$. The implementation uses a
previous-batch EMA and updates it **after** the gradient is formed, so the baseline never
depends on the sample it scores.

## Dual Update

The dual multiplier $\beta$ is updated by projected gradient ascent:

$$
\beta := \max(0,\ \beta + \text{dual\_lr} \cdot (\text{mean\_kl} - \epsilon))
$$

This increases the penalty when observed KL exceeds the budget. The update also runs after
the gradient is formed, so $\beta$ stays the multiplier the loss actually used. Minibatch dual
updates oscillate, so deploy only a checkpoint whose on-policy forward KL is below $\epsilon$
on a sufficiently large held-out evaluation set.

## Forced EOS And Action Masks

The probe was trained at an actual assistant end-of-turn token, so the implementation needs
an unambiguous terminal read position:

- Naturally sampled EOS tokens are **actions**: they appear in `action_mask` and contribute to
  $\log q_\theta(y | x)$ and $C_\theta$.
- If the rollout reaches `horizon` without emitting EOS, a forced EOS is appended. It is in
  `transcript_mask` (so the probe sees a valid terminal) but **not** in `action_mask` (so it
  does not contribute log probability or KL). Without this, the objective would be conditional
  on termination before the horizon, and the conditioning event depends on $\theta$.
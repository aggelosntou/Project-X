# Optimizer Zoo — Key Learnings

## 1. Vanilla SGD — Derivation from First Principles

We want: θ* = argmin L(θ).

Taylor expansion to first order around θ:
    L(θ + Δθ) ≈ L(θ) + ∇L(θ)ᵀ Δθ

To guarantee a decrease, choose Δθ = -α ∇L(θ) (steepest descent direction):
    L(θ - α g) ≈ L(θ) - α ||g||²

Since α > 0 and ||g||² ≥ 0, this always decreases (if L is smooth and α is small enough).

SGD update: θ = θ - lr * g

Problem: The learning rate is the same for all parameters. On an "elongated" loss
surface (condition number κ >> 1), the optimal lr in the short direction causes
oscillations in the long direction, and the optimal lr for the long direction
is too small for the short direction. SGD zigzags.

## 2. SGD with Momentum — Ball Rolling Downhill

Physical analogy: gradient is the slope, velocity accumulates from repeated pushes.

    v_t = μ * v_{t-1} + g_t        (μ = momentum = 0.9 typically)
    θ   = θ - α * v_t

Expanding the recursion:
    v_t = g_t + μ*g_{t-1} + μ²*g_{t-2} + ...   (exponentially weighted sum of past grads)

Benefits:
1. Consistent directions: if gradient consistently points left, velocity grows
   (effective lr increases), convergence accelerates.
2. Oscillating directions: gradients cancel in v (effective lr decreases),
   oscillations dampen.
3. Local minima: accumulated velocity carries the ball through small bumps.

Theoretical speedup over SGD for quadratics:
    SGD:      ρ = (κ-1)/(κ+1)    (convergence rate per step)
    Momentum: ρ = (√κ-1)/(√κ+1) (much better for large κ)

For κ=100: SGD ρ≈0.98 (slow), Momentum ρ≈0.82 (much faster).

## 3. Nesterov Accelerated Gradient — Look Ahead

Standard momentum computes gradient at current position θ, then takes a momentum
step. It "overshoots" because it doesn't know where the momentum will take it.

Nesterov's insight: compute gradient at the predicted next position (θ - μ*v).
This gives the gradient information one step ahead, allowing early correction.

    v_t = μ * v_{t-1} + g(θ - μ*v_{t-1})   ← gradient at look-ahead position
    θ   = θ - α * v_t

Theoretical result (Nesterov, 1983):
    Convergence rate for smooth convex functions: O(1/t²) vs O(1/t) for gradient descent.
    This is provably optimal for first-order methods on smooth convex functions.

For deep learning: Nesterov momentum often converges slightly faster than standard
momentum in practice and can generalize better (fewer oscillations).

## 4. RMSProp — Why Adaptive Learning Rates Help

AdaGrad insight: scale lr inversely by the square root of accumulated gradient:
    θ = θ - lr / sqrt(G_t) * g    where G_t = sum_{i=1}^t g_i²

Problem: G_t grows without bound → effective lr → 0 → learning stops.

RMSProp fix: use exponential moving average instead of cumulative sum:
    E[g²]_t = ρ * E[g²]_{t-1} + (1-ρ) * g²_t
    θ = θ - lr / sqrt(E[g²]_t + ε) * g

The EMA has "finite memory" of approximately 1/(1-ρ) steps.

WHY ADAPTIVE RATES MATTER for NLP:
  Word embeddings have sparse gradient updates: "the" gets gradients every step,
  rare words "sesquipedalian" gets gradients rarely.

  With SGD: all embeddings share one lr. Rare words barely move.
  With RMSProp: "the" has large E[g²] → small effective lr (doesn't overshoot).
                Rare words have small E[g²] → large effective lr (can catch up).

  The adaptive rate is proportional to 1/RMS(gradient_history), which equals
  approximately the inverse Hessian diagonal — a second-order approximation!

## 5. Adam — Combining Momentum and Adaptive Rates

Adam maintains:
  - m_t: exponential moving average of gradient (first moment, like momentum)
  - v_t: exponential moving average of squared gradient (second moment, like RMSProp)

    m_t = β₁ * m_{t-1} + (1-β₁) * g_t
    v_t = β₂ * v_{t-1} + (1-β₂) * g²_t
    θ   = θ - α * (m̂_t / sqrt(v̂_t) + ε)

where m̂_t = m_t/(1-β₁ᵗ) and v̂_t = v_t/(1-β₂ᵗ) are bias-corrected.

Why bias correction?
  At t=1: m_1 = (1-β₁)*g₁ ≈ 0.1*g₁ (underestimates the true mean).
  Without correction, early steps are 10x smaller than intended.
  Dividing by (1-β₁ᵗ) rescales to the unbiased estimate.

Adam is robust to choice of lr across many problems because the adaptive
denominator sqrt(v̂_t) normalizes the effective step size regardless of
gradient scale. This is why lr=0.001 often works without tuning.

## 6. AdamW ≠ Adam + L2 Regularization

L2 regularization adds weight_decay * θ to the gradient:
    g_eff = g + λθ
    then runs Adam on g_eff

Problem: Adam then scales this modified gradient adaptively:
    update = lr * m̂_eff / (sqrt(v̂_eff) + ε)

The weight decay term λθ is divided by sqrt(v̂), which includes the gradient
history. For frequently-updated parameters (large v̂), the effective weight
decay is small: λθ / sqrt(v̂) << λθ. For rare parameters (small v̂), effective
weight decay is large. This is backward — commonly used parameters are
UNDER-regularized, rarely used parameters are OVER-regularized.

AdamW decouples weight decay from the adaptive update:
    θ = θ - lr * m̂/(sqrt(v̂)+ε)   ← Adam step (gradient-adaptive)
    θ = θ - lr * λ * θ             ← weight decay (direct, not through Adam)

Now every parameter decays toward zero at the same rate lr*λ, regardless
of its gradient history. This gives correct L2-like regularization.

Empirical result: AdamW consistently outperforms Adam+L2 on BERT and GPT
fine-tuning, often by 1-2% on downstream tasks. Now the default in HuggingFace.

## 7. When to Use Each Optimizer

| Problem | Recommended |
|---------|-------------|
| Convex problems, theoretical guarantees | SGD + Nesterov |
| Image CNNs (careful tuning) | SGD + Momentum (often better final acc) |
| Language models, transformers | Adam or AdamW |
| Fine-tuning pretrained models | AdamW |
| Sparse gradients (embeddings) | Adam or RMSProp |
| Online/streaming learning | SGD |

The "SGD often beats Adam on CNNs" observation: SGD with a carefully tuned
learning rate schedule (warm-up + cosine decay) often achieves better final
generalization on image tasks because it explores the loss surface more broadly.
Adam converges faster but sometimes to sharper, less-generalizing minima.

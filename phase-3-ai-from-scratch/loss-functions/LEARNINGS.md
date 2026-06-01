# Loss Functions — Key Learnings

## 1. Mean Squared Error (MSE) — Gradient Derivation

    L = (1/N) * sum_i (pred_i - target_i)²

Applying the chain rule to one term:

    dL/d(pred_i) = (1/N) * d/d(pred_i) [(pred_i - target_i)²]
                 = (1/N) * 2(pred_i - target_i) * 1       (chain rule: d(pred_i - t_i)/d(pred_i) = 1)
                 = (2/N) * (pred_i - target_i)

Vectorized: dL/dpred = (2/N) * (pred - target)

WHEN TO USE: Regression where you assume Gaussian noise.
  If y = f(x) + ε where ε ~ N(0, σ²), the negative log-likelihood is:
    -log p(y|x) = (y - f(x))² / (2σ²) + const
  Minimizing MSE = maximum likelihood under this Gaussian assumption.

PROBLEM: Outliers. If pred_i - target_i = 10 for one sample,
  it contributes 100 to the loss while a typical error of 1 contributes 1.
  One outlier can dominate training, causing the model to "chase" it.

## 2. Mean Absolute Error (MAE) — Gradient Derivation

    L = (1/N) * sum_i |pred_i - target_i|

    d|e|/de = sign(e) = +1 if e > 0
                       = -1 if e < 0
                       = undefined (subgradient ∈ [-1,+1]) if e = 0

    dL/d(pred_i) = sign(pred_i - target_i) / N

Vectorized: dL/dpred = sign(pred - target) / N

WHEN TO USE: Regression where you assume Laplacian noise.
  If y = f(x) + ε where ε ~ Laplace(0, b), the negative log-likelihood is:
    -log p(y|x) = |y - f(x)| / b + const
  Minimizing MAE = maximum likelihood under Laplacian assumption.

KEY PROPERTY: Robust to outliers. An outlier at 10x typical error contributes
  only 10x to the gradient (not 100x as in MSE). The optimal predictor is
  the MEDIAN of the target distribution (not the mean, as in MSE).

PROBLEM: Non-smooth at e=0. The gradient is ±1 regardless of how close pred
  is to target — no "slow down" signal near the minimum. This can cause
  oscillations and make convergence slower than MSE near the minimum.

## 3. Huber Loss — Gradient Derivation

The Huber loss is piecewise-defined:

    L_δ(e) = { 0.5 * e²           if |e| < δ    (quadratic regime)
             { δ * |e| - 0.5*δ²   if |e| >= δ   (linear regime, but continuous)

The 0.5*δ² term ensures continuity at |e| = δ (both pieces equal 0.5*δ²).

Gradient:

    dL/d(pred_i) = { e_i / N         if |e_i| < δ   (MSE gradient)
                  { δ * sign(e_i)/N  if |e_i| >= δ  (MAE-like, clipped at δ)

WHEN TO USE: When you want MSE-like precise convergence for typical errors
  but bounded gradient for outliers. Used in:
  - DQN (Deep Q-Networks): gradient clipping equivalent
  - Bounding box regression in object detection (SmoothL1 = Huber with δ=1)

PARAMETER δ:
  - Large δ: Huber ≈ MSE (outliers penalized quadratically)
  - Small δ: Huber ≈ MAE (most errors treated linearly)
  - δ = 1: standard Huber, quadratic for |e| < 1, linear otherwise

## 4. Binary Cross-Entropy — Gradient Derivation

For binary labels y ∈ {0, 1} and predicted probability p ∈ (0, 1):

    L = -(1/N) * sum_i [y_i * log(p_i) + (1-y_i) * log(1-p_i)]

Taking derivative w.r.t. p_i:

    dL/dp_i = -(1/N) * [y_i/p_i - (1-y_i)/(1-p_i)]
            = -(1/N) * [y_i*(1-p_i) - (1-y_i)*p_i] / [p_i*(1-p_i)]
            = -(1/N) * [y_i - p_i] / [p_i*(1-p_i)]
            = (1/N) * (p_i - y_i) / [p_i*(1-p_i)]

Combined with sigmoid backward (dp/dz = p*(1-p)):
    dL/dz_i = dL/dp_i * dp_i/dz_i = (p_i - y_i) / N

This beautiful cancellation is why we implement sigmoid + BCE together
and propagate dL/dz = (p - y)/N rather than going through dL/dp.

WHEN TO USE: Binary classification (mutually exclusive {0,1} labels).
  Information-theoretic view: minimizing BCE minimizes KL(p||q) between
  true distribution p and model q, which is the same as maximizing
  the likelihood of the data under the model.

## 5. Cross-Entropy Loss — Gradient Derivation

For multi-class (C classes), targets y_i ∈ {0,...,C-1}, logits z ∈ R^C:

    L = -(1/N) * sum_i log(softmax(z_i)[y_i])

Let s_k = softmax(z)_k = exp(z_k) / sum_j exp(z_j)

For one sample (dropping 1/N for clarity):
    L = -log(s_{y}) = -z_y + log(sum_k exp(z_k))

Taking derivative:
    dL/dz_k = -1_{k=y} + exp(z_k) / sum_j exp(z_j)
            = -1_{k=y} + s_k
            = s_k - 1_{k=y}

where 1_{k=y} is 1 if k equals the target class, 0 otherwise.

Vectorized over all samples: dL/dz = (softmax(z) - one_hot(y)) / N

NUMERICAL STABILITY (log-sum-exp trick):
  Naive: exp(z_k) overflows for z_k > 709.
  Stable: subtract max(z) first.
    log_softmax(z)_k = z_k - max(z) - log(sum_j exp(z_j - max(z)))
  Since z_j - max(z) ≤ 0, inner exp is ≤ 1 (no overflow).
  Subtracting max(z) shifts the distribution but doesn't change probabilities.

## 6. Focal Loss — Gradient Derivation

    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

where p_t = p if y=1, (1-p) if y=0.

Derivative w.r.t. p_t (treating α_t as constant):

    dFL/d(p_t) = d/dp_t [-α_t * (1-p_t)^γ * log(p_t)]

Using product rule: d(f*g)/dp = f'*g + f*g'
    f = (1-p_t)^γ,          f' = -γ*(1-p_t)^(γ-1)
    g = -log(p_t),           g' = -1/p_t

    dFL/d(p_t) = -α_t * [(-γ*(1-p_t)^(γ-1))*(-log(p_t)) + (1-p_t)^γ*(-1/p_t)]
               = -α_t * [(1-p_t)^(γ-1) * (γ*log(p_t)*(1-p_t)/(1-p_t) - (1-p_t)/p_t)]

Simplified:
    dFL/d(p_t) = α_t * (1-p_t)^(γ-1) * [-(1-p_t)/p_t - γ*log(p_t)]

Then chain rule to p: dp_t/dp = +1 if y=1, -1 if y=0.

THE KEY INSIGHT:
    (1-p_t)^γ is the "modulating factor" that down-weights easy examples:
    - Easy (p_t=0.9): (1-0.9)^2 = 0.01 → 100x less gradient than CE
    - Hard (p_t=0.5): (1-0.5)^2 = 0.25 → only 4x less gradient
    Training focuses on hard examples where the model is uncertain.

WHEN TO USE:
    Class-imbalanced detection (10K background, 10 foreground per image).
    Without focal loss: model learns to classify everything as background
    (0.999 accuracy, 0 foreground recall). The 10K easy background examples
    dominate the gradient. Focal loss suppresses their contribution.
    After focal loss: model learns meaningful foreground detectors.

## 7. Triplet Loss — Gradient Derivation

    L = max(0, d(a,p) - d(a,n) + margin)

where d(u,v) = ||u-v||² (squared Euclidean).

For the active case (d(a,p) - d(a,n) + margin > 0):

    d(a,p) = sum_i (a_i - p_i)²
    d(a,n) = sum_i (a_i - n_i)²

Gradient w.r.t. anchor a:
    dL/da_i = d/da_i [sum_j (a_j-p_j)²] - d/da_i [sum_j (a_j-n_j)²]
            = 2(a_i - p_i) - 2(a_i - n_i)

Vectorized: dL/da = 2(a - p) - 2(a - n) = 2[(a-p) - (a-n)]

Gradient w.r.t. positive p:
    dL/dp_i = -2(a_i - p_i)   → dL/dp = 2(p - a)

Gradient w.r.t. negative n:
    dL/dn_i = +2(a_i - n_i)   → dL/dn = 2(a - n) = -2(n - a)

For inactive triplets (loss = 0): all gradients are zero.

GEOMETRIC INTERPRETATION:
    dL/da: push anchor away from negative, pull toward positive
    dL/dp: pull positive toward anchor
    dL/dn: push negative away from anchor

This shapes the embedding space so that same-class embeddings cluster together
and different-class embeddings are separated by at least `margin`.

## 8. When to Use Each Loss Function

| Situation | Recommended Loss |
|-----------|-----------------|
| Regression, clean data | MSE (fast convergence near optimum) |
| Regression, noisy/outlier labels | MAE or Huber |
| Bounding box regression | Huber (SmoothL1), IoU loss |
| Binary classification | BCE (with sigmoid) |
| Multi-class classification | Cross-Entropy (with softmax) |
| Severely imbalanced classes | Focal Loss |
| Metric learning, retrieval | Triplet Loss, Contrastive Loss |
| Generative models (VAE) | Reconstruction: MSE or BCE + KL divergence |

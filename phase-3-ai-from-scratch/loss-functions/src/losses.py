"""
losses.py — All major loss functions with forward and backward passes.

Each loss class follows the interface:
    loss = LossClass()
    value = loss.forward(pred, target)   # compute scalar loss, cache inputs
    grad  = loss.backward()              # return gradient w.r.t. pred

Notation used in derivations:
    N = number of elements
    e = pred - target (error)
    p = predicted probability
    y = target class / label
"""

import numpy as np


class MSELoss:
    """
    Mean Squared Error (L2 Loss).

    L(pred, target) = (1/N) * sum_i (pred_i - target_i)²

    DERIVATION:
        dL/d(pred_i) = d/d(pred_i) [(1/N) * (pred_i - target_i)²]
                     = (2/N) * (pred_i - target_i)
        Vectorized:
        dL/d(pred) = (2/N) * (pred - target)

    WHEN TO USE:
        Regression with Gaussian noise assumption.
        If we model p(y|x) = N(f(x), σ²), then:
        -log p(y|x) = (y - f(x))² / (2σ²) + const
        Minimizing MSE = maximizing likelihood under Gaussian noise.
        Works well when errors are symmetric and outliers are rare.

    SENSITIVITY TO OUTLIERS:
        Squares the error: a 10x larger error contributes 100x more to the loss.
        One outlier can dominate training → use Huber or MAE for robust regression.
    """

    def __init__(self):
        self._pred   = None
        self._target = None

    def forward(self, pred: np.ndarray, target: np.ndarray) -> float:
        self._pred   = pred
        self._target = target
        return float(np.mean((pred - target) ** 2))

    def backward(self) -> np.ndarray:
        return 2 * (self._pred - self._target) / self._pred.size


class MAELoss:
    """
    Mean Absolute Error (L1 Loss).

    L(pred, target) = (1/N) * sum_i |pred_i - target_i|

    DERIVATION:
        d|e|/de = sign(e) = +1 if e > 0, -1 if e < 0, undefined at e=0

        Convention: gradient is 0 at e=0 (subgradient).

        dL/d(pred_i) = sign(pred_i - target_i) / N
        Vectorized:
        dL/d(pred) = sign(pred - target) / N

    WHEN TO USE:
        Regression with Laplacian noise assumption.
        If p(y|x) ~ Laplace(f(x), b), then:
        -log p(y|x) = |y - f(x)| / b + const
        Minimizing MAE = maximizing likelihood under Laplacian noise.

        Laplacian has heavier tails than Gaussian → MAE is robust to outliers.
        An outlier at 10x the typical error contributes only 10x (not 100x).
        The optimal predictor is the median (not mean) of the target distribution.
    """

    def __init__(self):
        self._pred   = None
        self._target = None

    def forward(self, pred: np.ndarray, target: np.ndarray) -> float:
        self._pred   = pred
        self._target = target
        return float(np.mean(np.abs(pred - target)))

    def backward(self) -> np.ndarray:
        e = self._pred - self._target
        # Subgradient: sign(e), with convention sign(0) = 0
        return np.sign(e) / self._pred.size


class HuberLoss:
    """
    Huber Loss (Huber, 1964).

    A hybrid between MSE and MAE that is quadratic for small errors and
    linear for large errors:

        L_δ(e) = { 0.5 * e²           if |e| < δ
                 { δ * (|e| - 0.5*δ)  if |e| >= δ

    DERIVATION of gradient:
        dL/de = { e           if |e| < δ      (same as MSE gradient)
               { δ * sign(e)  if |e| >= δ     (same as δ * MAE gradient)

        dL/d(pred) = dL/de * de/d(pred) = dL/de * 1

    The transition point δ is a hyperparameter:
        δ → 0: approaches MAE (purely robust)
        δ → ∞: approaches MSE (purely sensitive to outliers)
        δ = 1: standard Huber loss

    WHEN TO USE:
        Regression when you want MSE-like behavior for typical errors but
        don't want outliers to dominate. Common in:
        - Q-learning target regression (DQN uses Huber to clip gradients)
        - Object detection bounding box regression
        - Any regression where outlier labels are possible
    """

    def __init__(self, delta: float = 1.0):
        self.delta   = delta
        self._pred   = None
        self._target = None
        self._e      = None

    def forward(self, pred: np.ndarray, target: np.ndarray) -> float:
        self._pred   = pred
        self._target = target
        e = pred - target
        self._e = e

        abs_e = np.abs(e)
        small = abs_e < self.delta
        large = ~small

        loss = np.where(small,
                        0.5 * e ** 2,
                        self.delta * (abs_e - 0.5 * self.delta))
        return float(loss.mean())

    def backward(self) -> np.ndarray:
        e     = self._e
        delta = self.delta
        small = np.abs(e) < delta

        grad = np.where(small, e, delta * np.sign(e))
        return grad / self._pred.size


class BinaryCrossEntropy:
    """
    Binary Cross-Entropy Loss (Log Loss).

    L(p, y) = -(1/N) * sum_i [y_i * log(p_i) + (1 - y_i) * log(1 - p_i)]

    where p ∈ (0, 1) is the predicted probability and y ∈ {0, 1} is the target.
    Apply sigmoid BEFORE this loss (or use BCEWithLogits for numerical stability).

    DERIVATION:
        dL/d(p_i) = d/dp [-y*log(p) - (1-y)*log(1-p)] / N
                  = [-y/p + (1-y)/(1-p)] / N
                  = [p - y] / [p*(1-p)] / N

        Simplified: dL/dp = (p - y) / N    (when combined with sigmoid backward,
        the sigmoid derivative p*(1-p) cancels: dL/dz = p - y for sigmoid input z)

    INFORMATION-THEORETIC INTERPRETATION:
        CE = -E[log q(y)] = H(p, q) = H(p) + KL(p || q)
        Minimizing CE = minimizing KL divergence between true distribution p
        and model distribution q.

    WHEN TO USE:
        Binary classification (spam/not-spam, cat/not-cat).
        The optimal predictor outputs p(y=1|x), the true probability.
    """

    def __init__(self, eps: float = 1e-7):
        self.eps     = eps
        self._pred   = None
        self._target = None

    def forward(self, pred: np.ndarray, target: np.ndarray) -> float:
        p = np.clip(pred, self.eps, 1 - self.eps)
        self._pred   = p
        self._target = target
        loss = -(target * np.log(p) + (1 - target) * np.log(1 - p))
        return float(loss.mean())

    def backward(self) -> np.ndarray:
        p = self._pred
        y = self._target
        return (p - y) / (p * (1 - p)) / p.size


class CrossEntropyLoss:
    """
    Multi-class Cross-Entropy Loss (log-softmax + NLL).

    L(logits, y) = -(1/N) * sum_i log(softmax(logits_i)[y_i])

    NUMERICALLY STABLE LOG-SOFTMAX:
        Naive: log(softmax(x)_k) = log(exp(x_k) / sum_j exp(x_j))
                                 = x_k - log(sum_j exp(x_j))
        Problem: exp(x_j) overflows for x_j > 709.

        Stable (log-sum-exp trick):
        log(sum_j exp(x_j)) = max(x) + log(sum_j exp(x_j - max(x)))
        Since x_j - max(x) ≤ 0, all terms in the inner sum are ≤ 1 (no overflow).

        Final stable formula:
        log_softmax(x)_k = x_k - max(x) - log(sum_j exp(x_j - max(x)))

    DERIVATION of gradient w.r.t. logits:
        Let s_k = softmax(logits)_k = exp(logits_k) / sum_j exp(logits_j)
        L = -(1/N) * sum_i logits_i[y_i] - log(sum_k exp(logits_i_k))

        For correct class (j = y_i):
            dL/d(logits_ij) = -(1/N) + (1/N) * s_ij = (1/N) * (s_ij - 1)

        For incorrect class (j ≠ y_i):
            dL/d(logits_ij) = (1/N) * s_ij

        Vectorized: dL/d(logits) = (1/N) * (softmax(logits) - one_hot(y))

    WHEN TO USE:
        Multi-class classification (mutually exclusive classes).
        Maximizes likelihood under softmax model.
    """

    def __init__(self, eps: float = 1e-9):
        self.eps      = eps
        self._logits  = None
        self._targets = None
        self._softmax = None

    def forward(self, logits: np.ndarray, targets: np.ndarray) -> float:
        """
        logits:  (N, C) — unnormalized scores
        targets: (N,)   — integer class indices
        """
        N = logits.shape[0]
        self._logits  = logits
        self._targets = targets

        # Numerically stable softmax
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp_z   = np.exp(shifted)
        softmax = exp_z / (exp_z.sum(axis=1, keepdims=True) + self.eps)
        self._softmax = softmax

        # NLL at correct class
        log_probs = np.log(softmax + self.eps)
        loss = -log_probs[np.arange(N), targets].mean()
        return float(loss)

    def backward(self) -> np.ndarray:
        """dL/d(logits) = (softmax - one_hot(targets)) / N"""
        N = self._logits.shape[0]
        grad = self._softmax.copy()
        grad[np.arange(N), self._targets] -= 1.0
        return grad / N


class FocalLoss:
    """
    Focal Loss (Lin et al., 2017 — RetinaNet).

    Addresses extreme class imbalance in dense object detection:
    ~100,000 background anchors vs ~10 foreground anchors per image.

    Standard cross-entropy: L = -log(p_t)

    Focal Loss: FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    where:
        p_t = p     if y=1   (predicted prob of correct class)
              1-p   if y=0
        α_t = α     if y=1   (class weight, typically 0.25 for foreground)
              1-α   if y=0
        γ   = focusing parameter (typically 2.0)

    THE KEY IDEA:
        (1 - p_t)^γ is the "modulating factor":
        - If p_t ≈ 1 (easy correct): (1-p_t)^2 ≈ 0 → loss ≈ 0 (down-weighted)
        - If p_t ≈ 0 (hard example): (1-p_t)^2 ≈ 1 → loss ≈ CE (full weight)

        When γ=2, p_t=0.9: modulation = 0.01 → 100× less than CE
        When γ=2, p_t=0.5: modulation = 0.25 → only 4× less than CE

    DERIVATION of gradient:
        Let L = -α * (1-p)^γ * log(p)
        dL/dp = -α * [γ*(1-p)^(γ-1)*(-1)*log(p) + (1-p)^γ * (1/p)]
              = α * [(1-p)^(γ-1) * (γ*log(p)*(p-1)/(1-p) - ... )]

        Simplified:
        dL/dp = α * (1-p)^(γ-1) * [γ*(-log(p))*(1-p)/... + ...]

        Full expression:
        dL/dp = -α * (1-p)^γ / p + α * γ * (1-p)^(γ-1) * log(p)
              = α * (1-p)^(γ-1) * [-(1-p)/p - γ*log(p)]

        Then dL/d(logit) = dL/dp * dp/dlogit = dL/dp * p*(1-p)

    WHEN TO USE:
        Class-imbalanced object detection / segmentation.
        When easy examples dominate and prevent learning on hard examples.
        The γ parameter trades off between CE (γ=0) and full focus (γ→∞).
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, eps: float = 1e-7):
        self.alpha   = alpha
        self.gamma   = gamma
        self.eps     = eps
        self._pred   = None
        self._target = None

    def forward(self, pred: np.ndarray, target: np.ndarray) -> float:
        """
        Binary focal loss.
        pred:   (N,) — predicted probabilities in (0, 1) after sigmoid
        target: (N,) — binary labels {0, 1}
        """
        p = np.clip(pred, self.eps, 1 - self.eps)
        self._pred   = p
        self._target = target

        # p_t: probability of the correct class
        p_t = np.where(target == 1, p, 1 - p)
        # alpha_t: class weight
        alpha_t = np.where(target == 1, self.alpha, 1 - self.alpha)

        focal_weight = (1 - p_t) ** self.gamma
        loss = -alpha_t * focal_weight * np.log(p_t + self.eps)
        return float(loss.mean())

    def backward(self) -> np.ndarray:
        """
        dFL/dp for binary case.

        dFL/dp_t = -α * [(1-p_t)^γ / p_t - γ*(1-p_t)^(γ-1)*log(p_t)]
        Then use dp_t/dp = +1 if y=1, -1 if y=0.
        """
        p  = self._pred
        y  = self._target
        N  = p.size

        p_t     = np.where(y == 1, p, 1 - p)
        alpha_t = np.where(y == 1, self.alpha, 1 - self.alpha)

        # d(FL)/d(p_t)
        log_pt  = np.log(p_t + self.eps)
        d_fl_pt = -alpha_t * (
            (1 - p_t) ** self.gamma / (p_t + self.eps)
            - self.gamma * (1 - p_t) ** (self.gamma - 1) * log_pt
        )

        # Chain rule: dp_t/dp = 1 if y==1, -1 if y==0
        d_fl_dp = d_fl_pt * np.where(y == 1, 1.0, -1.0)
        return d_fl_dp / N


class TripletLoss:
    """
    Triplet Loss (Schroff et al., 2015 — FaceNet).

    Used for metric learning: learn an embedding space where:
    - same-class examples are close
    - different-class examples are far apart

    L(a, p, n) = max(0, d(a,p) - d(a,n) + margin)

    where:
        a = anchor embedding
        p = positive embedding (same class as anchor)
        n = negative embedding (different class from anchor)
        d(u,v) = ||u - v||²   (squared Euclidean distance)
        margin = minimum gap between positive and negative distances

    INTERPRETATION:
        d(a,p): distance to positive (same class) — want small
        d(a,n): distance to negative (different class) — want large
        We want d(a,n) > d(a,p) + margin
        If this holds: max(0, ...) = 0, no gradient (satisfied constraint)
        If not: loss > 0, gradients push a closer to p and away from n

    DERIVATION of gradient:
        Let f = d(a,p) - d(a,n) + margin
        If f <= 0: gradient = 0 (satisfied constraint)
        If f > 0:
            d(a,p) = ||a - p||² = sum_i (a_i - p_i)²
            d(a,n) = ||a - n||² = sum_i (a_i - n_i)²

            dL/da = d(d(a,p))/da - d(d(a,n))/da
                  = 2(a-p) - 2(a-n)
            dL/dp = d(d(a,p))/dp = -2(a-p)  = 2(p-a)
            dL/dn = -d(d(a,n))/dn = 2(a-n)  = -2(n-a)

    APPLICATIONS:
        Face verification (FaceNet), signature verification,
        image similarity, drug-target interaction prediction.
    """

    def __init__(self, margin: float = 1.0):
        self.margin  = margin
        self._anchor   = None
        self._positive = None
        self._negative = None
        self._active   = None   # which triplets are "active" (loss > 0)

    def forward(self, anchor: np.ndarray, positive: np.ndarray,
                negative: np.ndarray) -> float:
        """
        anchor:   (N, D) — anchor embeddings
        positive: (N, D) — positive embeddings (same class)
        negative: (N, D) — negative embeddings (different class)
        Returns scalar loss.
        """
        self._anchor   = anchor
        self._positive = positive
        self._negative = negative

        # Squared Euclidean distances
        d_pos = np.sum((anchor - positive) ** 2, axis=-1)  # (N,)
        d_neg = np.sum((anchor - negative) ** 2, axis=-1)  # (N,)

        losses = np.maximum(0.0, d_pos - d_neg + self.margin)
        self._active = losses > 0   # mask of active triplets

        return float(losses.mean())

    def backward(self) -> tuple:
        """
        Returns (d_anchor, d_positive, d_negative).
        Gradient is zero for inactive triplets (already satisfied).
        """
        a = self._anchor
        p = self._positive
        n = self._negative
        N = a.shape[0]

        # Active mask: (N, 1) for broadcasting
        mask = self._active[:, np.newaxis].astype(float)

        d_a = mask * (2 * (a - p) - 2 * (a - n)) / N
        d_p = mask * (-2 * (a - p)) / N
        d_n = mask * ( 2 * (a - n)) / N

        return d_a, d_p, d_n


if __name__ == "__main__":
    print("=== Loss Functions Smoke Test ===\n")
    np.random.seed(42)

    # MSE
    pred   = np.array([1.0, 2.0, 3.0])
    target = np.array([1.5, 2.5, 2.0])
    mse    = MSELoss()
    print(f"MSE loss:    {mse.forward(pred, target):.6f}")
    print(f"MSE grad:    {mse.backward()}")

    # MAE
    mae = MAELoss()
    print(f"\nMAE loss:    {mae.forward(pred, target):.6f}")
    print(f"MAE grad:    {mae.backward()}")

    # Huber
    hub = HuberLoss(delta=1.0)
    print(f"\nHuber loss:  {hub.forward(pred, target):.6f}")
    print(f"Huber grad:  {hub.backward()}")

    # BCE
    p_bin = np.array([0.8, 0.3, 0.6])
    y_bin = np.array([1.0, 0.0, 1.0])
    bce   = BinaryCrossEntropy()
    print(f"\nBCE loss:    {bce.forward(p_bin, y_bin):.6f}")
    print(f"BCE grad:    {bce.backward()}")

    # Cross-entropy
    logits  = np.array([[2.0, 1.0, 0.1], [0.5, 3.0, 0.2]])
    targets = np.array([0, 1])
    ce      = CrossEntropyLoss()
    print(f"\nCE loss:     {ce.forward(logits, targets):.6f}")
    print(f"CE grad:\n{ce.backward()}")

    # Focal
    fl = FocalLoss(alpha=0.25, gamma=2.0)
    print(f"\nFocal loss:  {fl.forward(p_bin, y_bin):.6f}")
    print(f"Focal grad:  {fl.backward()}")

    # Triplet
    a = np.random.randn(4, 8)
    p_emb = a + 0.1 * np.random.randn(4, 8)   # close to anchor
    n_emb = a + 2.0 * np.random.randn(4, 8)   # far from anchor
    trip  = TripletLoss(margin=1.0)
    print(f"\nTriplet loss: {trip.forward(a, p_emb, n_emb):.6f}")
    da, dp, dn = trip.backward()
    print(f"grad shapes: d_anchor={da.shape}, d_pos={dp.shape}, d_neg={dn.shape}")
    print("\nAll smoke tests passed.")

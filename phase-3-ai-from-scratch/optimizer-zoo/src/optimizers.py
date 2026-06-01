"""
optimizers.py — All major gradient-based optimizers implemented in pure NumPy.

MOTIVATION
  Optimization is the engine of deep learning. Given a loss surface L(θ),
  we want to find θ* = argmin L(θ). All methods here are first-order (use
  only gradient information, not Hessian).

COMMON INTERFACE
  Every optimizer has:
    step(params_and_grads)
      params_and_grads: list of (param_array, grad_array) pairs
      Modifies param_array in-place. Does NOT zero grads — caller must do that.

  This mirrors PyTorch's interface:
      loss.backward()         # accumulate grads
      optimizer.step()        # update weights
      optimizer.zero_grad()   # clear grads for next iteration
"""

import numpy as np


class SGD:
    """
    Vanilla Stochastic Gradient Descent.

    Update rule:
        θ = θ - lr * g

    Derivation:
        We want to minimize L(θ). Taylor expansion to first order:
            L(θ - ε * g) ≈ L(θ) - ε * ||g||²   (where g = ∇L)
        This is always negative when ε > 0, so moving in the -gradient
        direction always decreases L (locally). The step size lr = ε trades
        off between convergence speed and stability.

    Pros: simple, well-understood, works well with momentum
    Cons: sensitive to learning rate, slow on ravines (gradient points
          diagonally, steps oscillate rather than progressing toward optimum)
    """

    def __init__(self, lr: float = 0.01):
        self.lr = lr

    def step(self, params_and_grads: list):
        for p, g in params_and_grads:
            p -= self.lr * g


class SGDMomentum:
    """
    SGD with Momentum (Polyak, 1964).

    Update rule:
        v_t = momentum * v_{t-1} + g_t
        θ   = θ - lr * v_t

    Derivation / intuition:
        Think of θ as a ball rolling down a hill. Without momentum, the ball
        takes small discrete steps. With momentum, past velocity v_{t-1}
        persists — the ball accelerates in consistent gradient directions
        and decelerates in oscillating directions.

        For a quadratic L(θ) = 0.5 * θ^T H θ, convergence rate improves from:
            O((1 - lr * λ_min(H))^t)          (plain SGD)
        to:
            O(((√κ - 1) / (√κ + 1))^t)        (momentum, optimal lr)
        where κ = λ_max(H) / λ_min(H) is the condition number.
        For ill-conditioned problems (large κ), this is much faster.

    Why it escapes shallow local minima:
        If gradient is small but consistent, v accumulates. The ball rolls
        through small bumps it would otherwise stop at. True local minima
        require the ball to slow down naturally (oscillating gradients
        cancel in v) — but shallow minima with small Hessian don't.

    Typical: momentum=0.9 (keep 90% of previous velocity)
    """

    def __init__(self, lr: float = 0.01, momentum: float = 0.9):
        self.lr       = lr
        self.momentum = momentum
        self._v       = []

    def step(self, params_and_grads: list):
        if not self._v:
            self._v = [np.zeros_like(p) for p, g in params_and_grads]

        for i, (p, g) in enumerate(params_and_grads):
            self._v[i] = self.momentum * self._v[i] + g
            p -= self.lr * self._v[i]


class Nesterov:
    """
    Nesterov Accelerated Gradient (NAG) (Nesterov, 1983).

    Update rule:
        v_t = momentum * v_{t-1} + g(θ - momentum * v_{t-1})
        θ   = θ - lr * v_t

    Equivalently (reformulated for practical use):
        θ_look = θ - momentum * v_{t-1}              (look ahead)
        g      = ∇L(θ_look)                          (gradient at look-ahead)
        v_t    = momentum * v_{t-1} + lr * g
        θ      = θ - v_t

    Why it's better than standard momentum:
        Standard momentum computes gradient at current θ, then "overshoots."
        Nesterov computes gradient at the anticipated position (after momentum step),
        correcting course before the step. This gives better theoretical convergence:

        Convergence rate: O(1/t²) vs O(1/t) for plain SGD.
        For convex problems, this is the optimal first-order method.

    Implementation note:
        In practice, we implement the equivalent form that avoids two parameter
        arrays by using the look-ahead gradient directly. The PyTorch implementation
        uses this formulation.
    """

    def __init__(self, lr: float = 0.01, momentum: float = 0.9):
        self.lr       = lr
        self.momentum = momentum
        self._v       = []

    def step(self, params_and_grads: list):
        if not self._v:
            self._v = [np.zeros_like(p) for p, g in params_and_grads]

        for i, (p, g) in enumerate(params_and_grads):
            # Nesterov: update velocity with current grad, then apply momentum look-ahead
            v_prev      = self._v[i].copy()
            self._v[i]  = self.momentum * v_prev + self.lr * g
            # Update: "undo" previous velocity, apply new velocity
            p -= (1 + self.momentum) * self._v[i] - self.momentum * v_prev


class RMSProp:
    """
    RMSProp (Hinton, 2012 — unpublished Coursera lecture).

    Update rule:
        E[g²]_t = decay * E[g²]_{t-1} + (1 - decay) * g²_t
        θ        = θ - (lr / sqrt(E[g²]_t + eps)) * g_t

    Derivation / motivation:
        AdaGrad (Duchi et al., 2011) divides lr by the cumulative sum of squared
        gradients: θ = θ - lr / sqrt(sum_{i=1}^t g_i²) * g_t

        Problem: the denominator grows without bound → lr → 0, learning stops.

        RMSProp fixes this with an exponential moving average of squared gradients:
        E[g²]_t = decay * E[g²]_{t-1} + (1-decay) * g²_t

        This effectively has a "window" of 1/(1-decay) recent gradient magnitudes.
        Gradients from long ago contribute exponentially less.

    Why adaptive rates help (especially for NLP):
        In language models, rare words have sparse gradients (updated infrequently).
        Common words have dense gradients (updated every step).
        Fixed lr: rare word embeddings barely move (too small effective lr).
        RMSProp: each parameter gets its own effective lr = lr / sqrt(E[g²]).
        Rare parameters have small E[g²] → large effective lr → catch up.
        Common parameters have large E[g²] → small effective lr → don't overshoot.

    Typical: decay=0.9, eps=1e-8
    """

    def __init__(self, lr: float = 0.001, decay: float = 0.9, eps: float = 1e-8):
        self.lr    = lr
        self.decay = decay
        self.eps   = eps
        self._eg2  = []   # E[g²]

    def step(self, params_and_grads: list):
        if not self._eg2:
            self._eg2 = [np.zeros_like(p) for p, g in params_and_grads]

        for i, (p, g) in enumerate(params_and_grads):
            self._eg2[i] = self.decay * self._eg2[i] + (1 - self.decay) * g * g
            p -= self.lr / (np.sqrt(self._eg2[i]) + self.eps) * g


class Adam:
    """
    Adam: Adaptive Moment Estimation (Kingma & Ba, 2015).

    Update rule:
        m_t = beta1 * m_{t-1} + (1 - beta1) * g_t        (first moment / mean)
        v_t = beta2 * v_{t-1} + (1 - beta2) * g²_t       (second moment / variance)
        m̂_t = m_t / (1 - beta1^t)                         (bias correction)
        v̂_t = v_t / (1 - beta2^t)                         (bias correction)
        θ   = θ - lr * m̂_t / (sqrt(v̂_t) + eps)

    Why bias correction?
        At t=1, m_1 = (1-beta1)*g_1. If beta1=0.9, this is 0.1*g_1 — much
        smaller than the true gradient g_1. Without correction, early steps
        are tiny. Dividing by (1-beta1^t) = 1-0.9 = 0.1 rescales back to g_1.
        As t grows, beta1^t → 0, correction → 1 (no longer needed).

    Adam combines:
        - Momentum (m_t): accelerates in consistent directions
        - Adaptive rates (v_t): per-parameter learning rates based on gradient history

    Typical: lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8

    Why eps?
        Prevents division by zero when v̂_t is very small (rare/unupdated parameter).
        Also acts as a lower bound on the effective learning rate: lr/eps.
        Larger eps → more uniform step sizes (closer to SGD with momentum).
        Smaller eps → more adaptive (parameters with small grads get larger updates).
    """

    def __init__(self, lr: float = 0.001,
                 beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.lr    = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps   = eps
        self.t     = 0
        self._m    = []
        self._v    = []

    def step(self, params_and_grads: list):
        if not self._m:
            self._m = [np.zeros_like(p) for p, g in params_and_grads]
            self._v = [np.zeros_like(p) for p, g in params_and_grads]

        self.t += 1
        bc1 = 1 - self.beta1 ** self.t   # bias correction denominator
        bc2 = 1 - self.beta2 ** self.t

        for i, (p, g) in enumerate(params_and_grads):
            self._m[i] = self.beta1 * self._m[i] + (1 - self.beta1) * g
            self._v[i] = self.beta2 * self._v[i] + (1 - self.beta2) * g * g
            m_hat = self._m[i] / bc1
            v_hat = self._v[i] / bc2
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class AdamW:
    """
    AdamW: Adam with Decoupled Weight Decay (Loshchilov & Hutter, 2019).

    Update rule:
        m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
        v_t = beta2 * v_{t-1} + (1 - beta2) * g²_t
        m̂   = m_t / (1 - beta1^t)
        v̂   = v_t / (1 - beta2^t)
        θ   = θ - lr * (m̂ / (sqrt(v̂) + eps) + weight_decay * θ)
                                                ^^^^^^^^^^^^^^^^^^^
                                                decoupled — applied directly to θ

    WHY AdamW ≠ Adam + L2 Regularization
        L2 regularization adds weight_decay * θ to the GRADIENT before Adam:
            g_modified = g + weight_decay * θ

        Adam then computes the adaptive update using g_modified:
            m_t = beta1*m + (1-beta1) * (g + wd*θ)
            v_t = beta2*v + (1-beta2) * (g + wd*θ)²
            θ  -= lr * m̂ / (sqrt(v̂) + eps)

        The weight decay term (wd*θ) is SCALED by the adaptive rate 1/sqrt(v̂).
        For frequently-updated parameters with large v̂ (common words in NLP),
        the effective weight decay is small: wd * θ / sqrt(v̂) << wd * θ.
        The regularization is not applied uniformly — common parameters are
        effectively under-regularized.

        AdamW applies weight decay directly to θ, bypassing the adaptive scaling:
            θ = θ - lr * m̂/(sqrt(v̂)+eps)   ← Adam update
            θ = θ - lr * weight_decay * θ   ← direct decay (not through Adam)

        This gives correct, parameter-independent regularization — every weight
        decays toward zero at the same rate regardless of gradient history.
        AdamW consistently outperforms Adam+L2 in language model fine-tuning.
    """

    def __init__(self, lr: float = 0.001,
                 beta1: float = 0.9, beta2: float = 0.999,
                 eps: float = 1e-8, weight_decay: float = 0.01):
        self.lr           = lr
        self.beta1        = beta1
        self.beta2        = beta2
        self.eps          = eps
        self.weight_decay = weight_decay
        self.t            = 0
        self._m           = []
        self._v           = []

    def step(self, params_and_grads: list):
        if not self._m:
            self._m = [np.zeros_like(p) for p, g in params_and_grads]
            self._v = [np.zeros_like(p) for p, g in params_and_grads]

        self.t += 1
        bc1 = 1 - self.beta1 ** self.t
        bc2 = 1 - self.beta2 ** self.t

        for i, (p, g) in enumerate(params_and_grads):
            self._m[i] = self.beta1 * self._m[i] + (1 - self.beta1) * g
            self._v[i] = self.beta2 * self._v[i] + (1 - self.beta2) * g * g
            m_hat = self._m[i] / bc1
            v_hat = self._v[i] / bc2
            # Adam update (gradient-adaptive)
            adam_update = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            # Decoupled weight decay (direct, not through Adam)
            wd_update   = self.lr * self.weight_decay * p
            p -= adam_update + wd_update


if __name__ == "__main__":
    print("=== Optimizer smoke tests ===")
    np.random.seed(42)

    # Simple quadratic: L(x) = x^2, minimum at x=0
    def test_optimizer(opt, name, steps=1000):
        x = np.array([5.0])   # start far from optimum
        for _ in range(steps):
            g = 2 * x     # gradient of x^2
            opt.step([(x, g)])
        print(f"  {name:15s}: x={x[0]:.6f}  (target: 0.0)")

    test_optimizer(SGD(lr=0.01),              "SGD")
    test_optimizer(SGDMomentum(lr=0.01),      "SGDMomentum")
    test_optimizer(Nesterov(lr=0.01),         "Nesterov")
    test_optimizer(RMSProp(lr=0.01),          "RMSProp")
    test_optimizer(Adam(lr=0.1),              "Adam")
    test_optimizer(AdamW(lr=0.1, weight_decay=0.0), "AdamW (wd=0)")

    # Verify AdamW ≠ Adam+L2 (they give different results)
    print("\n=== AdamW vs Adam+L2 difference ===")
    x_aw = np.array([2.0])
    x_al = np.array([2.0])
    opt_aw = AdamW(lr=0.01, weight_decay=0.1)
    opt_al = Adam(lr=0.01)

    for _ in range(100):
        # AdamW: gradient only, weight decay separate
        g_aw = 2 * x_aw
        opt_aw.step([(x_aw, g_aw)])

        # Adam+L2: add wd*x to gradient
        g_al = 2 * x_al + 0.1 * x_al
        opt_al.step([(x_al, g_al)])

    print(f"  AdamW result:  {x_aw[0]:.6f}")
    print(f"  Adam+L2 result:{x_al[0]:.6f}")
    print(f"  Difference:    {abs(x_aw[0] - x_al[0]):.6f}  (should be nonzero)")
    assert abs(x_aw[0] - x_al[0]) > 1e-6, "AdamW and Adam+L2 should differ!"
    print("All optimizer tests passed.")

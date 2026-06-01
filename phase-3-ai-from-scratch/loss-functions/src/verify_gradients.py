"""
verify_gradients.py — Numerical gradient checking for all loss functions.

For each loss:
  1. Compute analytical gradient via backward()
  2. Compute numerical gradient via central finite differences
  3. Assert max absolute difference < 1e-5
  4. Report result

Central finite differences formula:
    df/dx_i ≈ (f(x + ε*e_i) - f(x - ε*e_i)) / (2ε)

where e_i is the i-th unit vector and ε = 1e-5 is a small perturbation.

This is the standard gradient check used in deep learning frameworks.
Error is O(ε²) (second-order accuracy of central differences vs O(ε) for forward).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from losses import (MSELoss, MAELoss, HuberLoss,
                    BinaryCrossEntropy, CrossEntropyLoss,
                    FocalLoss, TripletLoss)


def numerical_gradient(loss_forward_fn, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """
    Compute gradient via central finite differences.

    loss_forward_fn: callable that takes x and returns scalar loss
    x:   parameter array to differentiate with respect to
    eps: finite difference step size

    Returns gradient array with same shape as x.

    Why central differences (not forward)?
        Forward: (f(x+ε) - f(x)) / ε — error O(ε)
        Central: (f(x+ε) - f(x-ε)) / 2ε — error O(ε²)
        For ε=1e-5, central is ~ε = 1e-5 times more accurate.
    """
    grad = np.zeros_like(x, dtype=np.float64)
    x_flat = x.ravel()
    grad_flat = grad.ravel()

    for idx in range(len(x_flat)):
        x_plus  = x.copy().ravel()
        x_minus = x.copy().ravel()
        x_plus[idx]  += eps
        x_minus[idx] -= eps

        f_plus  = loss_forward_fn(x_plus.reshape(x.shape))
        f_minus = loss_forward_fn(x_minus.reshape(x.shape))

        grad_flat[idx] = (f_plus - f_minus) / (2 * eps)

    return grad


def check_gradient(name: str, analytical: np.ndarray, numerical: np.ndarray,
                   threshold: float = 1e-5) -> bool:
    """
    Compare analytical and numerical gradients.
    Returns True if check passes.
    """
    max_err    = float(np.abs(analytical - numerical).max())
    rel_err    = max_err / (float(np.abs(numerical).max()) + 1e-8)
    passed     = max_err < threshold
    status     = "PASSED" if passed else "FAILED"
    print(f"  {name:<30}: {status}  (max_err={max_err:.2e}, rel_err={rel_err:.2e})")
    return passed


def verify_mse():
    """MSE gradient check."""
    np.random.seed(1)
    pred   = np.random.randn(5).astype(np.float64)
    target = np.random.randn(5).astype(np.float64)

    loss = MSELoss()
    loss.forward(pred, target)
    ana_grad = loss.backward()

    def fn(p): return MSELoss().forward(p, target)
    num_grad = numerical_gradient(fn, pred)

    return check_gradient("MSELoss", ana_grad, num_grad)


def verify_mae():
    """MAE gradient check (at non-zero points to avoid subgradient issues)."""
    np.random.seed(2)
    pred   = np.random.randn(5).astype(np.float64) + 0.5   # avoid 0
    target = np.random.randn(5).astype(np.float64)

    loss = MAELoss()
    loss.forward(pred, target)
    ana_grad = loss.backward()

    def fn(p): return MAELoss().forward(p, target)
    num_grad = numerical_gradient(fn, pred)

    return check_gradient("MAELoss", ana_grad, num_grad)


def verify_huber():
    """Huber gradient check (both regimes)."""
    np.random.seed(3)
    # Test both small errors (quadratic regime) and large errors (linear regime)
    pred   = np.array([0.1, 0.5, 2.0, -1.5, 0.3])
    target = np.zeros(5)

    loss = HuberLoss(delta=1.0)
    loss.forward(pred, target)
    ana_grad = loss.backward()

    def fn(p): return HuberLoss(delta=1.0).forward(p, target)
    num_grad = numerical_gradient(fn, pred)

    return check_gradient("HuberLoss", ana_grad, num_grad)


def verify_bce():
    """Binary cross-entropy gradient check."""
    np.random.seed(4)
    pred   = np.clip(np.random.rand(5), 0.05, 0.95)
    target = (np.random.rand(5) > 0.5).astype(np.float64)

    loss = BinaryCrossEntropy()
    loss.forward(pred, target)
    ana_grad = loss.backward()

    def fn(p): return BinaryCrossEntropy().forward(np.clip(p, 1e-7, 1-1e-7), target)
    num_grad = numerical_gradient(fn, pred)

    return check_gradient("BinaryCrossEntropy", ana_grad, num_grad)


def verify_cross_entropy():
    """Cross-entropy gradient check."""
    np.random.seed(5)
    N, C = 4, 5
    logits  = np.random.randn(N, C)
    targets = np.random.randint(0, C, N)

    loss = CrossEntropyLoss()
    loss.forward(logits, targets)
    ana_grad = loss.backward()

    def fn(l): return CrossEntropyLoss().forward(l, targets)
    num_grad = numerical_gradient(fn, logits)

    return check_gradient("CrossEntropyLoss", ana_grad, num_grad)


def verify_focal():
    """Focal loss gradient check."""
    np.random.seed(6)
    pred   = np.clip(np.random.rand(5), 0.05, 0.95)
    target = (np.random.rand(5) > 0.5).astype(np.float64)

    loss = FocalLoss(alpha=0.25, gamma=2.0)
    loss.forward(pred, target)
    ana_grad = loss.backward()

    def fn(p): return FocalLoss(alpha=0.25, gamma=2.0).forward(
        np.clip(p, 1e-7, 1-1e-7), target)
    num_grad = numerical_gradient(fn, pred)

    return check_gradient("FocalLoss", ana_grad, num_grad)


def verify_triplet_anchor():
    """Triplet loss gradient check w.r.t. anchor."""
    np.random.seed(7)
    N, D = 3, 4
    anchor   = np.random.randn(N, D)
    positive = anchor + 0.2 * np.random.randn(N, D)
    negative = anchor + 1.5 * np.random.randn(N, D)

    loss = TripletLoss(margin=1.0)
    loss.forward(anchor, positive, negative)
    d_a, _, _ = loss.backward()

    def fn(a): return TripletLoss(margin=1.0).forward(a, positive, negative)
    num_grad = numerical_gradient(fn, anchor)

    return check_gradient("TripletLoss (d_anchor)", d_a, num_grad)


def verify_triplet_positive():
    """Triplet loss gradient check w.r.t. positive."""
    np.random.seed(8)
    N, D = 3, 4
    anchor   = np.random.randn(N, D)
    positive = anchor + 0.2 * np.random.randn(N, D)
    negative = anchor + 1.5 * np.random.randn(N, D)

    loss = TripletLoss(margin=1.0)
    loss.forward(anchor, positive, negative)
    _, d_p, _ = loss.backward()

    def fn(p): return TripletLoss(margin=1.0).forward(anchor, p, negative)
    num_grad = numerical_gradient(fn, positive)

    return check_gradient("TripletLoss (d_positive)", d_p, num_grad)


def verify_triplet_negative():
    """Triplet loss gradient check w.r.t. negative."""
    np.random.seed(9)
    N, D = 3, 4
    anchor   = np.random.randn(N, D)
    positive = anchor + 0.2 * np.random.randn(N, D)
    negative = anchor + 1.5 * np.random.randn(N, D)

    loss = TripletLoss(margin=1.0)
    loss.forward(anchor, positive, negative)
    _, _, d_n = loss.backward()

    def fn(n): return TripletLoss(margin=1.0).forward(anchor, positive, n)
    num_grad = numerical_gradient(fn, negative)

    return check_gradient("TripletLoss (d_negative)", d_n, num_grad)


def run_all():
    print("=" * 60)
    print("GRADIENT VERIFICATION — Analytical vs Numerical")
    print(f"Method: Central finite differences (ε = 1e-5)")
    print(f"Threshold: max |analytical - numerical| < 1e-5")
    print("=" * 60)

    checks = [
        verify_mse,
        verify_mae,
        verify_huber,
        verify_bce,
        verify_cross_entropy,
        verify_focal,
        verify_triplet_anchor,
        verify_triplet_positive,
        verify_triplet_negative,
    ]

    results = [fn() for fn in checks]
    n_passed = sum(results)
    n_total  = len(results)

    print(f"\n{'=' * 60}")
    print(f"Result: {n_passed}/{n_total} gradient checks passed.")

    if n_passed == n_total:
        print("ALL PASSED — all gradients are correct.")
    else:
        print("WARNING: Some gradients failed. Check implementations above.")

    return all(results)


if __name__ == "__main__":
    np.random.seed(42)
    success = run_all()
    import sys
    sys.exit(0 if success else 1)

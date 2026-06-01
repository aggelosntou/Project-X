"""
compare_with_pytorch.py — Verify NumPy backprop matches PyTorch exactly.

We train the same architecture (identical initialization, same hyperparameters,
same batches) with both implementations and compare loss curves.

If our backprop is correct, the loss curves should be IDENTICAL (to floating-
point precision), since both compute the same math on the same numbers.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
np.random.seed(0)

from layers import Linear, ReLU
from losses import CrossEntropyLoss
from network import Sequential
from optimizers import SGD as NumpySGD

# ------------------------------------------------------------------ #
#  Synthetic dataset: 4-class classification, 8-dimensional features  #
# ------------------------------------------------------------------ #

N = 100
n_features = 8
n_classes  = 4

X_np = np.random.randn(N, n_features).astype(np.float64)
y_np = np.random.randint(0, n_classes, N)


def run_numpy(W1, b1, W2, b2, lr, n_steps):
    """Train with our NumPy implementation using fixed initial weights."""
    model = Sequential([
        Linear(n_features, 16),
        ReLU(),
        Linear(16, n_classes),
    ])
    # Override weights with provided values (for identical init)
    model.layers[0].W = W1.copy()
    model.layers[0].b = b1.copy()
    model.layers[2].W = W2.copy()
    model.layers[2].b = b2.copy()

    criterion = CrossEntropyLoss()
    optimizer = NumpySGD(model, lr=lr)

    losses = []
    for step in range(n_steps):
        logits = model.forward(X_np)
        loss   = criterion.forward(logits, y_np)
        losses.append(loss)
        grad = criterion.backward()
        model.backward(grad)
        optimizer.step()
    return losses


def run_pytorch(W1, b1, W2, b2, lr, n_steps):
    """Train with PyTorch using identical weights."""
    try:
        import torch
        import torch.nn as nn

        model = nn.Sequential(
            nn.Linear(n_features, 16),
            nn.ReLU(),
            nn.Linear(16, n_classes),
        )

        # Copy weights
        with torch.no_grad():
            model[0].weight.copy_(torch.tensor(W1.T, dtype=torch.float64))
            model[0].bias.copy_(torch.tensor(b1.flatten(), dtype=torch.float64))
            model[2].weight.copy_(torch.tensor(W2.T, dtype=torch.float64))
            model[2].bias.copy_(torch.tensor(b2.flatten(), dtype=torch.float64))

        model = model.double()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        X_t = torch.tensor(X_np, dtype=torch.float64)
        y_t = torch.tensor(y_np, dtype=torch.long)

        losses = []
        for step in range(n_steps):
            optimizer.zero_grad()
            logits = model(X_t)
            loss   = criterion(logits, y_t)
            losses.append(float(loss))
            loss.backward()
            optimizer.step()
        return losses

    except ImportError:
        return None


def main():
    print("=== NumPy vs PyTorch Comparison ===\n")

    # Shared initialization
    np.random.seed(42)
    W1 = np.random.randn(n_features, 16) * 0.1
    b1 = np.zeros((1, 16))
    W2 = np.random.randn(16, n_classes) * 0.1
    b2 = np.zeros((1, n_classes))

    lr = 0.01
    n_steps = 20

    print("Running NumPy implementation...")
    losses_np = run_numpy(W1, b1, W2, b2, lr, n_steps)

    print("Running PyTorch implementation...")
    losses_pt = run_pytorch(W1, b1, W2, b2, lr, n_steps)

    print(f"\n{'Step':>6}  {'NumPy Loss':>12}  {'PyTorch Loss':>12}  {'Diff':>12}")
    print("-" * 50)
    for i, l_np in enumerate(losses_np):
        if losses_pt is not None:
            l_pt = losses_pt[i]
            diff = abs(l_np - l_pt)
            print(f"{i+1:6d}  {l_np:12.6f}  {l_pt:12.6f}  {diff:12.2e}")
        else:
            print(f"{i+1:6d}  {l_np:12.6f}  {'(PyTorch N/A)':>12}  {'N/A':>12}")

    if losses_pt is not None:
        max_diff = max(abs(a - b) for a, b in zip(losses_np, losses_pt))
        print(f"\nMax absolute difference across all steps: {max_diff:.2e}")
        if max_diff < 1e-10:
            print("PASS: Loss curves match exactly (within numerical precision)")
        elif max_diff < 1e-5:
            print("PASS: Loss curves match closely (may differ due to float32 vs float64)")
        else:
            print("FAIL: Loss curves differ significantly — check backprop implementation")
    else:
        print("\n(PyTorch not available — skipping comparison)")
        print(f"NumPy final loss: {losses_np[-1]:.6f}")


if __name__ == '__main__':
    main()

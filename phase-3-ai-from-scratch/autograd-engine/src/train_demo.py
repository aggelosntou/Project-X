"""
train_demo.py — Train a 2-layer MLP on the XOR dataset using our autograd engine.

XOR is the classic non-linearly-separable problem:
  (0,0) → 0,  (0,1) → 1,  (1,0) → 1,  (1,1) → 0

A linear classifier cannot solve XOR (the two classes are not separable by
a single hyperplane).  A 2-layer network can, using the hidden layer to
create a new feature space where the classes ARE linearly separable.

We also attempt to visualize the decision boundary if matplotlib is available.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import random
random.seed(42)

from engine import Value
from nn import MLP
from optim import Adam

# ------------------------------------------------------------------ #
#  Dataset: XOR                                                        #
# ------------------------------------------------------------------ #

# XOR truth table — the target is binary (0 or 1)
X_data = [[0.0, 0.0],
           [0.0, 1.0],
           [1.0, 0.0],
           [1.0, 1.0]]

y_data = [0.0, 1.0, 1.0, 0.0]

# ------------------------------------------------------------------ #
#  Model: 2 → 4 → 4 → 1 MLP with tanh activations                    #
# ------------------------------------------------------------------ #

model = MLP(2, [4, 4, 1], hidden_activation='tanh')
print(f"Model: {model}")
print(f"Total parameters: {len(model.parameters())}")

optimizer = Adam(model.parameters(), lr=0.05)


def mse_loss(predictions, targets):
    """Mean squared error: (1/N) * sum((pred - target)^2)"""
    n = len(predictions)
    return sum((p - t)**2 for p, t in zip(predictions, targets)) * (1.0 / n)


# ------------------------------------------------------------------ #
#  Training loop                                                       #
# ------------------------------------------------------------------ #

n_epochs = 2000
print(f"\nTraining for {n_epochs} epochs...")

for epoch in range(n_epochs):
    # Forward pass: evaluate the model on all training points
    predictions = [model([Value(xi) for xi in x]) for x in X_data]

    # Compute loss
    loss = mse_loss(predictions, [Value(y) for y in y_data])

    # Backward pass: compute all gradients
    optimizer.zero_grad()
    loss.backward()

    # Update parameters
    optimizer.step()

    if epoch % 200 == 0 or epoch == n_epochs - 1:
        preds_raw = [p.data for p in predictions]
        print(f"  Epoch {epoch:4d}: loss = {loss.data:.6f}, "
              f"preds = [{', '.join(f'{p:.3f}' for p in preds_raw)}]")

# ------------------------------------------------------------------ #
#  Evaluation                                                          #
# ------------------------------------------------------------------ #

print("\nFinal predictions vs targets:")
for x, y in zip(X_data, y_data):
    pred = model([Value(xi) for xi in x])
    correct = "✓" if abs(pred.data - y) < 0.1 else "✗"
    print(f"  XOR({int(x[0])}, {int(x[1])}) = {y:.0f}  |  pred = {pred.data:.4f}  {correct}")

# ------------------------------------------------------------------ #
#  Decision boundary visualization (optional)                          #
# ------------------------------------------------------------------ #

try:
    import matplotlib
    matplotlib.use('Agg')  # non-interactive backend
    import matplotlib.pyplot as plt
    import numpy as np

    resolution = 30
    xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, resolution),
                          np.linspace(-0.5, 1.5, resolution))
    Z = np.zeros_like(xx)
    for i in range(resolution):
        for j in range(resolution):
            inp = [Value(float(xx[i,j])), Value(float(yy[i,j]))]
            Z[i,j] = model(inp).data

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.contourf(xx, yy, Z, levels=20, cmap='RdBu', alpha=0.7)
    ax.contour(xx, yy, Z, levels=[0.5], colors='black')
    for x, y in zip(X_data, y_data):
        ax.scatter(x[0], x[1], c='red' if y == 0 else 'blue', s=200, zorder=5,
                   edgecolors='black')
    ax.set_title('XOR Decision Boundary')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    plt.tight_layout()
    plt.savefig('xor_decision_boundary.png', dpi=100)
    print("\nSaved decision boundary to xor_decision_boundary.png")
except ImportError:
    print("\n(matplotlib not available — skipping decision boundary plot)")

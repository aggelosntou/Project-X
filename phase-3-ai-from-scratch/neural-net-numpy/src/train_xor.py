"""
train_xor.py — Train a 2-4-4-1 network on XOR using NumPy backprop.

XOR truth table:
  (0,0) → 0,  (0,1) → 1,  (1,0) → 1,  (1,1) → 0

Expected: loss < 0.01 within 2000 epochs.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
np.random.seed(42)

from layers import Linear, Tanh
from losses import MSELoss
from network import Sequential
from optimizers import Adam

# ------------------------------------------------------------------ #
#  Data                                                                #
# ------------------------------------------------------------------ #

X = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)
y = np.array([[0], [1], [1], [0]], dtype=np.float32)

# ------------------------------------------------------------------ #
#  Model: 2 → 4 → 4 → 1 with tanh                                    #
# ------------------------------------------------------------------ #

model = Sequential([
    Linear(2, 4),
    Tanh(),
    Linear(4, 4),
    Tanh(),
    Linear(4, 1),
])

optimizer = Adam(model, lr=0.05)
criterion = MSELoss()

# ------------------------------------------------------------------ #
#  Training loop                                                       #
# ------------------------------------------------------------------ #

n_epochs = 3000
print(f"Training XOR (NumPy) for {n_epochs} epochs...")

for epoch in range(n_epochs):
    # Forward
    pred = model.forward(X)

    # Loss
    loss = criterion.forward(pred, y)

    # Backward
    grad = criterion.backward()
    model.backward(grad)

    # Update
    optimizer.step()

    if epoch % 500 == 0 or epoch == n_epochs - 1:
        pred_vals = pred.flatten()
        print(f"  Epoch {epoch:4d}: loss = {loss:.6f}, "
              f"preds = [{', '.join(f'{v:.3f}' for v in pred_vals)}]")

    if loss < 0.001:
        print(f"  Converged at epoch {epoch} (loss={loss:.6f})")
        break

# ------------------------------------------------------------------ #
#  Evaluation                                                          #
# ------------------------------------------------------------------ #

print("\nFinal predictions:")
pred = model.forward(X)
for i, (xi, yi) in enumerate(zip(X, y)):
    p = pred[i, 0]
    correct = abs(p - yi[0]) < 0.1
    symbol = "OK" if correct else "FAIL"
    print(f"  XOR({int(xi[0])},{int(xi[1])}) = {yi[0]:.0f} | pred = {p:.4f}  [{symbol}]")

final_loss = criterion.forward(model.forward(X), y)
print(f"\nFinal loss: {final_loss:.6f}")
print("SUCCESS: loss < 0.01" if final_loss < 0.01 else "Note: not fully converged")

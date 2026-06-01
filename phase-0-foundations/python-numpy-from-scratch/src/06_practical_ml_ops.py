"""
06_practical_ml_ops.py

The mathematical operations that appear constantly in machine learning,
implemented from scratch using only NumPy.

Understanding these from scratch prevents "magic function" thinking and
helps you debug when things go wrong (e.g., NaN in loss, exploding gradients).

Topics:
  1. Sigmoid and ReLU activation functions
  2. Softmax (numerically stable version)
  3. One-hot encoding
  4. MSE loss
  5. Cross-entropy loss
  6. Feature normalisation (z-score, min-max)
  7. Batch operations (computing over entire datasets at once)

Run: python3 src/06_practical_ml_ops.py
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)

print("=" * 60)
print("Practical ML Operations with NumPy")
print("=" * 60)

# -----------------------------------------------------------------------
# SECTION 1: Activation Functions
#
# Activation functions introduce non-linearity.  Without them, stacking
# linear layers would just give another linear transformation:
#   W2 @ (W1 @ x) = (W2 @ W1) @ x  — still just a matrix multiply
# -----------------------------------------------------------------------
print("\n--- 1. Activation Functions ---")

x = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
print(f"Input x: {x}")

# Sigmoid: σ(x) = 1 / (1 + e^{-x})
# Maps any real number to (0, 1).
# Interpretation: probability output for binary classification.
# Problem: saturates at 0 and 1, causing vanishing gradients.
def sigmoid(x):
    # Numerically stable version: for very negative x, exp(-x) can overflow.
    # np.clip prevents this: clip x to [-500, 500] before computing.
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

sigma = sigmoid(x)
print(f"sigmoid(x):  {sigma}")
print(f"  sigmoid(0) = 0.5 (midpoint)  ✓")
print(f"  Range is (0, 1) — maps to probability")

# ReLU: max(0, x)
# The most widely used activation in deep learning.
# Advantages: no vanishing gradient for positive x, computationally trivial.
# Disadvantage: "dying ReLU" — neurons with x < 0 always output 0.
def relu(x):
    return np.maximum(0.0, x)   # np.maximum is element-wise max with a scalar

relu_out = relu(x)
print(f"\nReLU(x):     {relu_out}")
print(f"  Zeroes all negatives, identity for positives")

# Leaky ReLU: max(αx, x) — small slope for negative region
def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

print(f"LeakyReLU(x):{leaky_relu(x)}")

# -----------------------------------------------------------------------
# SECTION 2: Softmax (numerically stable)
#
# Softmax converts a vector of real numbers (logits) into a probability
# distribution that sums to 1:
#
#   softmax(z)_i = exp(z_i) / sum_j(exp(z_j))
#
# NUMERICAL STABILITY PROBLEM:
# If z contains large values like [1000, 1001, 1002], exp(1002) overflows
# to inf.  Fix: subtract the max from z before computing exp.
# This doesn't change the result because:
#   softmax(z) = softmax(z - max(z))
# (dividing numerator and denominator by exp(max) cancels out)
# -----------------------------------------------------------------------
print("\n--- 2. Softmax ---")

def softmax(z):
    """Numerically stable softmax."""
    z_shifted = z - np.max(z)        # subtract max to prevent overflow
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z)

logits = np.array([2.0, 1.0, 0.1])
probs = softmax(logits)
print(f"Logits:        {logits}")
print(f"Softmax:       {probs}")
print(f"Sum of probs:  {probs.sum():.6f}  (should be 1.0)")

# Demonstrate numerical stability
big_logits = np.array([1000.0, 1001.0, 1002.0])
unstable = np.exp(big_logits) / np.sum(np.exp(big_logits))  # NaN due to inf/inf
stable = softmax(big_logits)
print(f"\nBig logits {big_logits}:")
print(f"  Naive softmax: {unstable}  ← NaN!")
print(f"  Stable softmax:{stable}   ← correct")

# Batched softmax (one row per sample)
def softmax_batch(Z):
    """Z shape: (batch_size, num_classes)."""
    # keepdims=True so we can broadcast: shape (batch,1) broadcasts with (batch,C)
    Z_shifted = Z - Z.max(axis=1, keepdims=True)
    exp_Z = np.exp(Z_shifted)
    return exp_Z / exp_Z.sum(axis=1, keepdims=True)

Z_batch = np.array([[2.0, 1.0, 0.1],
                    [0.5, 2.5, 0.0],
                    [1.0, 1.0, 1.0]])
print(f"\nBatched softmax (3 samples, 3 classes):\n{softmax_batch(Z_batch)}")
print(f"Row sums: {softmax_batch(Z_batch).sum(axis=1)}")

# -----------------------------------------------------------------------
# SECTION 3: One-Hot Encoding
#
# Convert integer class labels [0, 2, 1, 0] into binary indicator vectors.
# Class k gets a 1 in position k and 0s everywhere else.
#
# Used in: cross-entropy loss, output layers of classifiers.
# -----------------------------------------------------------------------
print("\n--- 3. One-Hot Encoding ---")

def one_hot(labels, num_classes):
    """
    labels: 1D integer array of class indices
    Returns: 2D float array of shape (len(labels), num_classes)
    """
    n = len(labels)
    result = np.zeros((n, num_classes), dtype=np.float64)
    # Fancy indexing: set position [i, labels[i]] to 1 for each i
    result[np.arange(n), labels] = 1.0
    return result

labels = np.array([0, 2, 1, 0, 3])
num_classes = 4
Y = one_hot(labels, num_classes)
print(f"Labels: {labels}")
print(f"One-hot:\n{Y}")

# -----------------------------------------------------------------------
# SECTION 4: MSE Loss (Mean Squared Error)
#
# Used for regression problems.
#   MSE = (1/n) * sum_i (y_pred_i - y_true_i)^2
#
# Gradient w.r.t. y_pred:  d(MSE)/d(y_pred) = 2/n * (y_pred - y_true)
# -----------------------------------------------------------------------
print("\n--- 4. MSE Loss ---")

def mse_loss(y_pred, y_true):
    """Mean Squared Error."""
    diff = y_pred - y_true
    return np.mean(diff ** 2)

def mse_gradient(y_pred, y_true):
    """Gradient of MSE w.r.t. y_pred."""
    n = len(y_pred)
    return (2.0 / n) * (y_pred - y_true)

y_pred = np.array([2.5, 0.0, 2.0, 8.0])
y_true = np.array([3.0, -0.5, 2.0, 7.0])

loss = mse_loss(y_pred, y_true)
grad = mse_gradient(y_pred, y_true)
print(f"y_pred: {y_pred}")
print(f"y_true: {y_true}")
print(f"MSE = (1/4)[(2.5-3)² + (0-(-0.5))² + (2-2)² + (8-7)²]")
print(f"    = (1/4)[{0.25:.2f} + {0.25:.2f} + {0:.2f} + {1:.2f}] = {loss:.4f}")
print(f"Gradient: {grad}")

# -----------------------------------------------------------------------
# SECTION 5: Cross-Entropy Loss
#
# Used for classification problems.  Measures how well predicted
# probabilities match the true class distribution.
#
# Binary cross-entropy (for sigmoid output):
#   L = -(y_true * log(y_pred) + (1-y_true) * log(1-y_pred))
#
# Categorical cross-entropy (for softmax output):
#   L = -sum_k y_true_k * log(y_pred_k)
#
# The -log penalises low confidence on the correct class heavily:
#   if y_pred_correct = 0.99 → loss = -log(0.99) ≈ 0.01
#   if y_pred_correct = 0.01 → loss = -log(0.01) ≈ 4.6  (huge penalty)
#
# EPS prevents log(0) = -inf
# -----------------------------------------------------------------------
print("\n--- 5. Cross-Entropy Loss ---")

EPS = 1e-12   # small value to prevent log(0)

def binary_cross_entropy(y_pred, y_true):
    """Binary cross-entropy for sigmoid outputs."""
    y_pred = np.clip(y_pred, EPS, 1 - EPS)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

y_pred_bin = np.array([0.9, 0.1, 0.8, 0.3])
y_true_bin = np.array([1.0, 0.0, 1.0, 0.0])
bce = binary_cross_entropy(y_pred_bin, y_true_bin)
print(f"Binary CE loss: {bce:.4f}  (low = good predictions)")

y_pred_bad = np.array([0.1, 0.9, 0.2, 0.7])   # wrong predictions
bce_bad = binary_cross_entropy(y_pred_bad, y_true_bin)
print(f"Bad predictions BCE: {bce_bad:.4f}  (higher = worse)")

def categorical_cross_entropy(y_pred, y_true_onehot):
    """Categorical cross-entropy for softmax outputs."""
    y_pred = np.clip(y_pred, EPS, 1.0)
    # -sum over classes of (true_label * log(pred_prob))
    # Only the correct class term is non-zero (one-hot encodes 0s and 1s)
    return -np.mean(np.sum(y_true_onehot * np.log(y_pred), axis=1))

probs = softmax_batch(Z_batch)
labels_batch = np.array([0, 1, 2])
Y_oh = one_hot(labels_batch, 3)
cce = categorical_cross_entropy(probs, Y_oh)
print(f"\nCategorical CE loss: {cce:.4f}")
print(f"Probabilities:\n{probs}")
print(f"True labels: {labels_batch}")

# -----------------------------------------------------------------------
# SECTION 6: Feature Normalisation
#
# Neural networks are sensitive to the SCALE of inputs.
# If feature A is in [0,1] and feature B is in [0,1000000],
# gradients for B will be 10^6 times larger → unstable training.
#
# Two standard approaches:
#   Z-score (standardisation): x_norm = (x - mean) / std
#     → result has mean ≈ 0, std ≈ 1
#   Min-max scaling:           x_norm = (x - min) / (max - min)
#     → result is in [0, 1]
# -----------------------------------------------------------------------
print("\n--- 6. Feature Normalisation ---")

# Simulate a small dataset: 5 samples, 3 features
rng = np.random.default_rng(0)
X = rng.random((5, 3))
X[:, 0] *= 1000   # feature 0 is in [0, 1000]
X[:, 1] *= 0.01   # feature 1 is in [0, 0.01]
X[:, 2] *= 100    # feature 2 is in [0, 100]
print(f"Raw features (5 samples, 3 features):\n{np.round(X, 4)}")

def zscore_normalise(X):
    """Standardise each COLUMN (feature) to mean=0, std=1."""
    mean = X.mean(axis=0)   # shape (3,) — one mean per feature
    std  = X.std(axis=0)    # shape (3,)
    return (X - mean) / (std + EPS), mean, std   # +EPS prevents /0

def minmax_normalise(X):
    """Scale each column to [0, 1]."""
    mn = X.min(axis=0)
    mx = X.max(axis=0)
    return (X - mn) / (mx - mn + EPS), mn, mx

X_z, mean, std = zscore_normalise(X)
X_mm, mn, mx   = minmax_normalise(X)

print(f"\nZ-score normalised:\n{np.round(X_z, 3)}")
print(f"  Column means: {np.round(X_z.mean(axis=0), 6)}  (≈ 0)")
print(f"  Column stds:  {np.round(X_z.std(axis=0), 6)}   (≈ 1)")

print(f"\nMin-max normalised:\n{np.round(X_mm, 3)}")
print(f"  Column mins:  {np.round(X_mm.min(axis=0), 6)}  (= 0)")
print(f"  Column maxes: {np.round(X_mm.max(axis=0), 6)}  (= 1)")

print("""
Summary of ML operations implemented:
  sigmoid(x)         : 1/(1+e^(-x))           — binary output
  relu(x)            : max(0,x)                — hidden layers
  softmax(z)         : exp(z)/sum(exp(z))      — multiclass output
  one_hot(labels, C) : indicator matrix         — encode labels
  mse_loss           : mean((pred-true)^2)     — regression loss
  bce_loss           : -mean(y log(p)+(1-y)log(1-p)) — binary loss
  cce_loss           : -mean(sum(y log(p)))    — multiclass loss
  zscore_normalise   : (x-μ)/σ                 — standardisation
  minmax_normalise   : (x-min)/(max-min)       — [0,1] scaling
""")

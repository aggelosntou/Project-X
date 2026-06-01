# Learnings: Neural Network in NumPy

## 1. How Matrix Calculus Makes Backprop Elegant

In scalar calculus, `d(wx)/dw = x`. In matrix calculus:

For `Z = XW` where X is (N, d_in) and W is (d_in, d_out):
```
dL/dW = X^T @ dL/dZ       [shape: (d_in, N) @ (N, d_out) = (d_in, d_out)] ✓
dL/dX = dL/dZ @ W^T       [shape: (N, d_out) @ (d_out, d_in) = (N, d_in)] ✓
```

The transpose rule: when a matrix appears on the left of a product in the
forward pass, it appears transposed on the RIGHT in the backward pass, and
vice versa. This is the matrix version of the chain rule.

**Why is this elegant?** Each layer's backward pass is just two matrix
multiplies. No scalar loops, no explicit Jacobians.

## 2. Why Numerically Stable Softmax Matters (with numbers)

Unstable: `exp(1000) / (exp(1000) + exp(999)) = inf/inf = NaN`

Stable: `exp(1000-1000) / (exp(1000-1000) + exp(999-1000)) = 1/(1+exp(-1)) = 0.731`

The stable computation gives the correct answer (the first logit dominates).

## 3. Fused Softmax + Cross-Entropy Gradient

The gradient of cross-entropy w.r.t. logits (not probabilities) is:
```
dL/dz_i = softmax(z)_i - t_i
```

This is the "simplest possible" gradient for a classification output.
The loss function is designed so that this elegant gradient results.
It works because the softmax and log cancel in the gradient computation.

## 4. Connection Between NumPy and C Implementations

The math is identical. The difference is memory management:

| Aspect | C implementation | NumPy implementation |
|--------|-----------------|---------------------|
| Matrix product | Tiled for-loops | `np.dot()` (calls BLAS) |
| Memory | `malloc`/`free` explicit | Python GC handles it |
| Speed | ~10× faster (depends on implementation) | Faster in practice (BLAS optimized) |
| Readability | Requires understanding pointers | Clean, mathematical |

The C version is more educational for systems understanding.
The NumPy version is more educational for mathematical intuition.

## 5. The Confusion Matrix and Class-Wise Performance

A 10×10 confusion matrix `C[i][j]` = number of samples with true class `i`
predicted as class `j`.

- Diagonal = correct predictions.
- Off-diagonal entry `C[i][j]` (i≠j) = errors: true class `i` mistaken for `j`.

Useful metrics derived from CM:
- **Precision**: `C[k][k] / sum(C[:, k])` — of all predicted class k, how many are actually k?
- **Recall**: `C[k][k] / sum(C[k, :])` — of all actual class k, how many did we predict correctly?
- **F1** = 2 * Precision * Recall / (Precision + Recall)

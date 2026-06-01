# Learnings: Neural Network in C

## 1. Xavier Initialization and Variance Preservation

Suppose layer l has n_in inputs and n_out outputs, and inputs have unit variance.

Output variance: `Var(z) = n_in * Var(W) * Var(a_{l-1})`

To preserve variance through the layer, we want `Var(z) = Var(a_{l-1})`, so:
```
n_in * Var(W) = 1  →  Var(W) = 1/n_in
```

For both forward and backward signal preservation (Glorot & Bengio, 2010):
```
Var(W) = 2/(n_in + n_out)
```

For Uniform(-a, a): `Var = a²/3`, so `a = sqrt(6/(n_in + n_out))`.

**Why it matters:** Without proper initialization:
- Too large → activations and gradients explode exponentially with depth.
- Too small → activations and gradients vanish exponentially with depth.
- With Xavier → signals stay in a reasonable range regardless of network depth.

## 2. What Backprop Actually Computes

Backprop computes `dL/dW` for every weight W, which is the direction to move
W to decrease the loss most steeply (gradient descent).

The key insight: the chain rule for matrix operations gives us efficient formulas.

For weight matrix `W_l` in a multi-layer network:
```
dL/dW_l = a_{l-1}^T @ delta_l
```

This is the outer product of:
- `a_{l-1}`: the input to layer l (what the layer "saw")
- `delta_l`:  the error signal at layer l (what the layer "should have done differently")

**Connection to the scalar engine:** In the autograd engine, each scalar multiplication `w * x` had the backward rule `dL/dw += x * dL/dout`. The matrix formula `dL/dW = x^T @ delta` is exactly this, summed over all samples in the batch.

## 3. Why We Store Pre-Activations

The backward pass needs `f'(z_l)` to compute delta:
```
delta_l = (delta_{l+1} @ W_{l+1}^T) ⊙ f'(z_l)
```

For ReLU: `f'(z) = 1 if z > 0, else 0` — requires knowing the sign of z, not just a = ReLU(z).

For sigmoid: `f'(z) = σ(z)(1-σ(z)) = a(1-a)` — here we could use a. But storing z is cleaner and works for all activations uniformly.

**Memory cost:** O(batch_size × n_hidden) per layer, which is typically small.

## 4. Connection Between C Implementation and Python Autograd

| Concept            | Python autograd engine       | C implementation           |
|--------------------|------------------------------|----------------------------|
| Forward pass       | `Value.__add__`, `__mul__`, etc. | `network_forward()`    |
| Backward pass      | Topological sort + closures  | Explicit backward loops    |
| Chain rule         | `_backward()` per operation  | Matrix formulas per layer  |
| Parameter update   | `p.data -= lr * p.grad`     | `network_update()`         |

The C version implements the same math but "unrolled" into efficient matrix operations, while the Python version builds the computation graph dynamically.

## 5. The Delta Formulation

`delta_l` (also called the "error signal" or "local gradient") compactly represents:
```
delta_l = dL/dz_l   (gradient of loss w.r.t. pre-activation)
```

Once we have delta, weight gradients follow immediately:
- `dL/dW_l = a_{l-1}^T @ delta_l` (by chain rule through z = aW+b)
- `dL/db_l = sum(delta_l, axis=0)` (bias gradient, averaged over batch)

This is the "delta rule" and is at the heart of every neural network training algorithm.

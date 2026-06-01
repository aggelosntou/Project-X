# Learnings: Matrix Library in C

## 1. Row-Major vs Column-Major Memory Layout

A 2D matrix is stored in a 1D array. Two conventions exist:

**Row-major (C, NumPy default):** row `i` is contiguous.
```
M = [[a, b],    data = [a, b, c, d]
     [c, d]]    M[i][j] = data[i*cols + j]
```

**Column-major (Fortran, MATLAB):** column `j` is contiguous.
```
M = [[a, b],    data = [a, c, b, d]
     [c, d]]    M[i][j] = data[j*rows + i]
```

**Why it matters for cache:** CPUs load memory in cache lines (typically 64 bytes = 16 floats). If you iterate in the wrong order, each access causes a cache miss, slowing down by 10–100×.

## 2. Tiled Matrix Multiply and Cache Efficiency

Matrix multiply `C = A @ B` requires reading A row-wise (good) and B column-wise (bad for cache).

The tiled algorithm reorders computation so that both A and B tiles fit in L1 cache:
- Each TILE×TILE block = 8×8×4 bytes = 256 bytes = 4 cache lines.
- L1 cache is typically 32–64 KB, so we can hold many tiles simultaneously.
- Result: dramatically fewer cache misses → 5–10× speedup on large matrices.

Real libraries (BLAS, cuBLAS) use this idea plus SIMD vectorization and multi-threading.

## 3. Why Bias Addition Requires Broadcasting

In a fully-connected layer with batch size B:
- Input `X`: shape (B, n_in)
- Weight `W`: shape (n_in, n_out)
- Bias `b`: shape (1, n_out) — ONE vector, NOT B copies

The operation is `X @ W + b` where `b` is added to each row of `X @ W`.
Broadcasting avoids allocating a (B, n_out) matrix just to hold `b` repeated B times.

Mathematical meaning: the bias shifts the hyperplane `w·x = 0` to `w·x + b = 0`.

## 4. Softmax: Mathematical Definition and Numerical Stability

**Definition:** Given logits `z ∈ R^K`:
```
softmax(z)_j = exp(z_j) / sum_{k=1}^{K} exp(z_k)
```

Softmax turns arbitrary real numbers into a probability distribution (non-negative, sums to 1).

**Why it's numerically unstable:** If `z_j = 1000`, then `exp(1000) = 2.07e434`, which overflows IEEE 754 float32 (max ~3.4e38).

**The max-shift trick:** For any constant `c`,
```
softmax(z)_j = exp(z_j - c) / sum_k exp(z_k - c)
```
because the `exp(-c)` factor cancels from numerator and denominator.

Choose `c = max(z)`. Then all arguments to `exp` are ≤ 0, so outputs are in `(0, 1]` — no overflow.

## 5. ReLU Gradient: The Indicator Function

ReLU `f(x) = max(0, x)` is not differentiable at x=0, but has a well-defined
subgradient everywhere:
```
f'(x) = 1  if x > 0
         0  if x ≤ 0
```

During backprop, we need to know which neurons were "active" (x > 0) during the forward pass.
This is why we **store pre-activations** — we need them during backward, not just the output.

## 6. Float vs Double in ML

Modern ML uses float32 (not float64) for:
- **Memory:** 2× smaller → fits 2× more parameters in GPU VRAM.
- **Speed:** GPU hardware is optimized for float32 (and increasingly float16/bfloat16).
- **Precision:** float32 has ~7 significant decimal digits, which is sufficient for most ML tasks.
  The stochastic nature of SGD means extreme numerical precision doesn't help.

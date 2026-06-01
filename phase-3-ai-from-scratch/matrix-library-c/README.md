# Matrix Library in C

A complete, dependency-free matrix library for machine learning, written in C11.

## Build & Test

```bash
make test
```

## Design

### Row-major Storage

Element `(i, j)` of an `m x n` matrix is stored at `data[i * n + j]`.

- Row `i` occupies indices `[i*n, i*n+1, ..., i*n + n-1]` — all contiguous.
- Iterating over a row is sequential memory access → cache-line friendly.
- Iterating over a column is strided (stride = n) → cache-unfriendly.

This is why the tiled matrix multiply matters: it reorders computation to maximize row-wise access.

### Tiled Matrix Multiply

The naive O(m×k×n) loop `c[i][j] += a[i][k] * b[k][j]` reads column `j` of B in the innermost loop — strided and cache-hostile. The tiled version (`TILE=8`) processes 8×8 sub-blocks that fit in L1 cache, dramatically reducing cache misses on large matrices.

### Bias Broadcasting

In a neural net layer, `output = input @ W + b` where `input` is `(batch × n_in)`, `W` is `(n_in × n_out)`, and `b` is a single vector of length `n_out`. Broadcasting means we add the same `b` to each row of the batch without allocating a full `(batch × n_out)` copy of `b`.

### Numerically Stable Softmax

The naive softmax `exp(x_j) / sum(exp(x_k))` overflows for large logits. The stable version subtracts the row maximum before exponentiating:

```
p_j = exp(x_j - max) / sum_k exp(x_k - max)
```

This is algebraically identical (the `exp(-max)` cancels) but keeps all exponents ≤ 0, preventing overflow.

## Files

- `src/matrix.h` — struct definition and function declarations
- `src/matrix.c` — full implementation with inline math derivations
- `src/test_matrix.c` — unit tests with hand-computed expected values

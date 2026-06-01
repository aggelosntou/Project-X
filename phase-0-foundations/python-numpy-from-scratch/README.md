# python-numpy-from-scratch

Six Python files that build a complete understanding of NumPy — from
how arrays are stored in memory to implementing core ML operations.

## Files

| File | What it teaches |
|------|-----------------|
| `src/01_array_basics.py` | Creation, dtypes, shape, reshape, strides (why transpose is O(1)) |
| `src/02_indexing_slicing.py` | Basic/boolean/fancy indexing, views vs copies, np.where |
| `src/03_broadcasting.py` | Broadcasting rules, outer product, why no memory is copied |
| `src/04_vectorization.py` | Timing benchmarks: Python loop vs NumPy, SIMD explanation, ufuncs |
| `src/05_linear_algebra.py` | Matrix multiply, linear solve, SVD, eigendecomposition, norms |
| `src/06_practical_ml_ops.py` | Sigmoid, ReLU, softmax, one-hot, MSE, cross-entropy, normalisation |

## Prerequisites

```bash
pip install numpy
```

Python 3.8+ required.

## How to run

```bash
python3 src/01_array_basics.py
python3 src/02_indexing_slicing.py
python3 src/03_broadcasting.py
python3 src/04_vectorization.py     # shows timing benchmarks
python3 src/05_linear_algebra.py
python3 src/06_practical_ml_ops.py
```

All files are self-contained and print their output with explanations.
Run them in order — each builds on concepts from the previous.

## Key concepts to watch for

- **01**: `a.strides` — the raw representation of a NumPy array. After transpose,
  the strides swap but the data bytes are identical.

- **02**: `np.shares_memory(a, b)` — use this to verify whether a slice is
  a view or a copy.

- **03**: Run the broadcasting examples and look at the shapes printed.
  Understand why `(4,3) + (3,)` works but `(4,3) + (4,)` needs `reshape`.

- **04**: The speedup numbers vary by machine but should be 10x-200x.
  If you have an AVX-512 CPU, NumPy can do 16 floats per instruction.

- **05**: After `U, s, Vt = np.linalg.svd(A)`, verify that `U @ np.diag(s) @ Vt`
  reconstructs A exactly.

- **06**: Try passing `logits = [1000, 1001, 1002]` to the naive softmax (before
  the stability fix) and watch it return NaN.

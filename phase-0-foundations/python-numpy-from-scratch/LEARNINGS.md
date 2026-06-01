# Learnings — python-numpy-from-scratch

## 1. What Strides Are (and Why Transpose Is O(1))

A NumPy array in memory is just a flat block of bytes.  The **strides** tuple
tells NumPy how many bytes to skip to move one step along each axis.

For a (3, 4) array of float64 (8 bytes each):
```
strides = (32, 8)
  axis 0 (rows):    skip 32 bytes = 4 elements × 8 bytes
  axis 1 (columns): skip  8 bytes = 1 element  × 8 bytes
```

Memory layout (row-major, the default "C order"):
```
[row0_col0][row0_col1][row0_col2][row0_col3] | [row1_col0] ... | [row2_col0] ...
```

**Transpose** does not move any bytes.  It just swaps the strides:
```python
a.strides = (32, 8)      # original (3,4)
a.T.strides = (8, 32)    # transposed (4,3) — same bytes, different interpretation
```

To access element `[i, j]` of the transposed array:
```
address = base + i * 8 + j * 32
```
Which is the same as `[j, i]` of the original — exactly what transpose means.

This makes `a.T` an O(1) operation regardless of matrix size.  In contrast,
a language without stride support (like early MATLAB) would copy all data.

---

## 2. What Broadcasting Is at the Memory Level

Broadcasting allows operations between arrays of different shapes.

At the implementation level, broadcasting is done by setting the **stride to 0**
for the broadcasted dimension.  A stride of 0 means: when you step along this
axis, stay at the same memory address — read the same data repeatedly.

Example: adding `bias` of shape (3,) to `matrix` of shape (4, 3):
```
bias has virtual shape (1, 3) with strides (0, 8)
  → moving along axis 0 adds 0 bytes → stays at same memory!
  → the 3 values are read once per row, 4 times total
```

No (4, 3) copy of `bias` is ever created.  The CPU just reads the same
3 values 4 times, which is extremely cache-friendly.

This is exactly how BLAS matrix multiplication works for the weight matrix
in a neural network layer.

---

## 3. Why Vectorization Is Fast (SIMD + Cache Locality)

**Python loops are slow because:**
1. Each iteration: Python bytecode dispatch, type checking, reference count updates
2. Python objects are heap-allocated and scattered in memory (cache misses)
3. One value processed per interpreter step

**NumPy is fast because:**

### SIMD (Single Instruction, Multiple Data)
Modern CPUs have special registers (128, 256, or 512 bits wide) that hold
multiple values at once.  One instruction operates on all of them:
- SSE2:    128 bits = 2 doubles or 4 floats
- AVX2:    256 bits = 4 doubles or 8 floats  (most modern desktops)
- AVX-512: 512 bits = 8 doubles or 16 floats (servers, recent CPUs)

So `a + b` on a (1,000,000,) float32 array takes ~62,500 AVX-512 instructions
instead of 1,000,000 Python additions.

### Cache locality
NumPy arrays are contiguous in memory.  The CPU's prefetcher can predict
the next memory access and load it into L1 cache before the code needs it.
Python lists store pointers to separate heap objects — each access is a
potential cache miss (random memory access = ~100 cycle penalty vs ~4 cycles
from L1 cache).

---

## 4. What a ufunc Is

A **ufunc** (universal function) is a NumPy function that:
1. Operates **element-wise** on arrays of any shape
2. Handles **broadcasting** automatically
3. Supports **type casting** (will upcast int to float when needed)
4. Has an **`out` parameter** to write into an existing array (no allocation)
5. Has built-in **reduce**, **accumulate**, and **outer** methods

Examples: `np.sin`, `np.cos`, `np.exp`, `np.log`, `np.sqrt`, `np.add`,
`np.multiply`, `np.maximum`, `np.logical_and`, etc.

You can create your own ufunc with `np.frompyfunc()` or `numba.vectorize()`,
but they won't be as fast as the built-in ones (which are hand-optimised C).

The ufunc abstraction is what makes this work:
```python
np.sin(scalar)          # returns a scalar
np.sin(vector)          # returns a vector
np.sin(matrix)          # returns a matrix
np.sin(3d_tensor)       # returns a 3D tensor
```

All with the same function call — no overloading, no if/else for shape.

---

## 5. Numerically Stable Softmax

The naive softmax formula `exp(z) / sum(exp(z))` overflows when `z` contains
large values (exp(1000) = inf on float64).

The stable trick: subtract the maximum element before taking exp.

**Mathematical proof it's equivalent:**
```
softmax(z)_i = exp(z_i) / Σ exp(z_j)

Let c = max(z).  Then:
softmax(z)_i = exp(z_i - c) / Σ exp(z_j - c)
             = [exp(z_i) / exp(c)] / [Σ exp(z_j) / exp(c)]
             = exp(z_i) / Σ exp(z_j)
             = softmax(z)_i  ✓
```

The exp(c) cancels in numerator and denominator — mathematically identical,
but now the maximum exponent is 0 (exp(0) = 1), and all others are negative,
so exp never exceeds 1.  No overflow.

This is a microcosm of a larger lesson: **always think about floating-point
stability** when implementing numerical algorithms.  The mathematically
equivalent formula that overflows in IEEE 754 arithmetic is not "correct"
for a computer implementation.

"""
04_vectorization.py

Demonstrate WHY NumPy is fast: vectorization replaces slow Python loops
with compiled C/Fortran code that uses SIMD instructions and cache-friendly
memory access patterns.

This file measures real speedups using time.perf_counter().

Run: python3 src/04_vectorization.py
"""

import numpy as np
import time

print("=" * 60)
print("Vectorization and Performance")
print("=" * 60)

N = 1_000_000   # 1 million elements — large enough to see a clear difference

# -----------------------------------------------------------------------
# SECTION 1: Python loop vs NumPy vectorized — element-wise square
# -----------------------------------------------------------------------
print(f"\n--- 1. Element-wise Square (N={N:,}) ---")

data = list(range(N))
arr  = np.arange(N, dtype=np.float64)

# Python loop — interpreted, one Python object per iteration
start = time.perf_counter()
result_python = [x * x for x in data]
t_python = time.perf_counter() - start

# NumPy vectorized — the loop runs in C, with SIMD if available
start = time.perf_counter()
result_numpy = arr * arr
t_numpy = time.perf_counter() - start

speedup = t_python / t_numpy
print(f"  Python list comprehension : {t_python*1000:.2f} ms")
print(f"  NumPy vectorized          : {t_numpy*1000:.2f} ms")
print(f"  Speedup                   : {speedup:.0f}x")

# -----------------------------------------------------------------------
# SECTION 2: Sum — Python vs NumPy
# -----------------------------------------------------------------------
print(f"\n--- 2. Sum of {N:,} Elements ---")

start = time.perf_counter()
s_python = sum(data)
t_python = time.perf_counter() - start

start = time.perf_counter()
s_numpy = np.sum(arr)
t_numpy = time.perf_counter() - start

print(f"  Python sum() : {t_python*1000:.2f} ms")
print(f"  np.sum()     : {t_numpy*1000:.2f} ms")
print(f"  Speedup      : {t_python/t_numpy:.0f}x")
print(f"  Results match: {int(s_python) == int(s_numpy)}")

# -----------------------------------------------------------------------
# SECTION 3: Universal Functions (ufuncs)
#
# A ufunc is a function that operates element-wise on arrays.
# NumPy provides ~60 built-in ufuncs (sin, cos, exp, log, sqrt, abs, etc.)
#
# Key properties of ufuncs:
#   - Written in C, compiled and highly optimised
#   - Support broadcasting automatically
#   - Support 'out' parameter to write results into an existing array
#     (avoids allocating a new array — important in performance-critical code)
#   - Support 'where' parameter for conditional application
#   - Have .reduce(), .accumulate(), .outer() methods
# -----------------------------------------------------------------------
print("\n--- 3. Universal Functions (ufuncs) ---")

x = np.linspace(0, 2 * np.pi, 6)
print(f"x = {np.round(x, 3)}")

# np.sin, np.cos, np.exp, etc. are ufuncs
print(f"np.sin(x)   = {np.round(np.sin(x), 3)}")
print(f"np.cos(x)   = {np.round(np.cos(x), 3)}")
print(f"np.exp(x)   = {np.round(np.exp(x), 3)}")

# ufunc methods:
y = np.array([1, 2, 3, 4, 5])
print(f"\nnp.add.reduce([1,2,3,4,5])     = {np.add.reduce(y)}")       # sum
print(f"np.add.accumulate([1,2,3,4,5]) = {np.add.accumulate(y)}")    # cumsum
print(f"np.multiply.reduce([1,2,3,4,5])= {np.multiply.reduce(y)}")   # product

# 'out' parameter — write directly into an existing array (no temp allocation)
result_buf = np.empty_like(x)
np.sin(x, out=result_buf)
print(f"\nnp.sin(x, out=result_buf) → avoids extra allocation")
print(f"result_buf = {np.round(result_buf, 3)}")

# -----------------------------------------------------------------------
# SECTION 4: Matrix multiply timing — the biggest speedup
#
# Matrix multiplication (A @ B) is O(n^3).  For n=500, that's 125 million
# multiply-add operations.  NumPy uses BLAS (Basic Linear Algebra Subroutines)
# — a library of extremely optimised routines with hand-written assembly
# for each CPU microarchitecture.
# -----------------------------------------------------------------------
print("\n--- 4. Matrix Multiply Speedup ---")

n = 300
A_py = [[float(i*n+j) for j in range(n)] for i in range(n)]  # Python list of lists
A_np = np.array(A_py)

# Python triple loop (just a small sample to estimate — full would take minutes)
sample = 50   # run on smaller n to estimate
A_sm = [[float(i*sample+j) for j in range(sample)] for i in range(sample)]
start = time.perf_counter()
C_sm = [[sum(A_sm[i][k]*A_sm[k][j] for k in range(sample)) for j in range(sample)] for i in range(sample)]
t_python_matmul = (time.perf_counter() - start) * (n/sample)**3  # estimate for full n
print(f"  Estimated Python matmul ({n}x{n}): ~{t_python_matmul:.1f} s")

A_np_sq = np.random.randn(n, n)
start = time.perf_counter()
C_np = A_np_sq @ A_np_sq
t_numpy_matmul = time.perf_counter() - start
print(f"  NumPy matmul   ({n}x{n}): {t_numpy_matmul*1000:.2f} ms")
print(f"  Estimated speedup: ~{t_python_matmul/t_numpy_matmul:.0f}x")

# -----------------------------------------------------------------------
# SECTION 5: Why vectorization is fast — the theory
# -----------------------------------------------------------------------
print("\n--- 5. Why Vectorization Is Fast ---")
print("""
Three reasons NumPy crushes Python loops:

1. COMPILED CODE
   NumPy operations run as compiled C (or Fortran) code.
   A Python loop has overhead per iteration: bytecode dispatch, type checking,
   reference count updates, garbage collection checks.
   NumPy's C loop has none of that.

2. SIMD (Single Instruction, Multiple Data)
   Modern CPUs can process 4, 8, or 16 floats in a SINGLE instruction.
   AVX-512: 512-bit registers = 8 doubles or 16 floats at once.
   NumPy uses these automatically (via BLAS/libm).
   A Python loop processes ONE value per instruction.

3. CACHE LOCALITY
   NumPy arrays store data contiguously in memory.
   The CPU cache can prefetch the next chunk while processing the current one.
   Python lists store objects scattered across the heap — each element is
   a separate heap-allocated object with a pointer in the list.
   This means random memory accesses → cache misses → slow.

The combination: compiled + SIMD + cache = 10x to 1000x speedup over Python.
""")

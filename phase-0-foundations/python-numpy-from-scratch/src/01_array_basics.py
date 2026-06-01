"""
01_array_basics.py

Understand what a NumPy array IS at the memory level, and the core
concepts of dtype, shape, and strides.

Run: python3 src/01_array_basics.py
"""

import numpy as np

print("=" * 60)
print("NumPy Array Basics")
print("=" * 60)

# -----------------------------------------------------------------------
# SECTION 1: Array creation
#
# Unlike Python lists, NumPy arrays:
#   - Store elements in CONTIGUOUS memory (like C arrays)
#   - All elements have the SAME dtype (no mixing ints and strings)
#   - Support vectorized operations (applied to all elements at once)
# -----------------------------------------------------------------------
print("\n--- 1. Array Creation ---")

# zeros / ones: create arrays filled with 0 or 1
a_zeros = np.zeros(5)              # shape (5,), dtype float64 by default
a_ones  = np.ones((3, 4))          # shape (3,4) matrix of 1.0
a_full  = np.full((2, 3), 7)       # filled with a constant value

print("zeros(5)       :", a_zeros)
print("ones((3,4))    :\n", a_ones)
print("full((2,3), 7) :\n", a_full)

# arange: like Python range() but returns an array
# np.arange(start, stop, step)
a_range = np.arange(0, 10, 2)       # [0, 2, 4, 6, 8]
print("\narange(0, 10, 2):", a_range)

# linspace: N evenly-spaced points INCLUSIVE on both ends
# Useful for plotting functions, creating coordinate grids
a_lin = np.linspace(0, 1, 6)        # [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
print("linspace(0,1,6) :", a_lin)

# random arrays
rng = np.random.default_rng(seed=42)   # seeded for reproducibility
a_rand = rng.random((3, 3))            # uniform [0, 1)
a_norm = rng.standard_normal((3, 3))   # standard normal N(0,1)
print("\nrandom (3,3):\n", np.round(a_rand, 3))
print("normal (3,3):\n", np.round(a_norm, 3))

# From Python list — NumPy infers the dtype
a_from_list = np.array([1, 2, 3, 4, 5])
print("\nFrom list [1..5]:", a_from_list)

# -----------------------------------------------------------------------
# SECTION 2: The dtype system
#
# dtype = data type of each element.  NumPy stores elements as raw bytes
# in a C-style block.  The dtype tells NumPy how many bytes each element
# is and how to interpret those bytes.
#
# Common dtypes:
#   float64  = 8 bytes, IEEE 754 double precision (default for floats)
#   float32  = 4 bytes, single precision (ML models often use this)
#   int64    = 8 bytes signed integer
#   int32    = 4 bytes signed integer
#   bool     = 1 byte, stores 0 or 1
#   complex128 = 16 bytes (two float64s)
# -----------------------------------------------------------------------
print("\n--- 2. dtype System ---")

dtypes_to_show = [np.float64, np.float32, np.int64, np.int32, np.bool_, np.complex128]
for dt in dtypes_to_show:
    a = np.zeros(1, dtype=dt)
    print(f"  dtype={str(dt.__name__):12s}  itemsize={a.itemsize} bytes")

# WHY dtype matters for ML:
# A float32 model uses HALF the memory of float64 with minimal accuracy loss.
# GPUs natively support float32 (and even float16); float64 is ~8x slower on GPU.
a32 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
a64 = np.array([1.0, 2.0, 3.0], dtype=np.float64)
print(f"\nfloat32 array nbytes: {a32.nbytes}  (3 elements × 4 bytes)")
print(f"float64 array nbytes: {a64.nbytes}  (3 elements × 8 bytes)")

# Type casting
a_int = np.array([1, 2, 3], dtype=np.int32)
a_float = a_int.astype(np.float64)   # creates a COPY with new dtype
print(f"\nCast int32→float64: {a_float}  dtype={a_float.dtype}")

# -----------------------------------------------------------------------
# SECTION 3: Shape, reshape, flatten
#
# Shape is a tuple describing the size along each dimension.
# A 1D array of 12 elements, a (3,4) matrix, and a (2,2,3) tensor all
# hold 12 elements but have different shapes.
#
# reshape() and ravel()/flatten() change how the same bytes are VIEWED —
# no data is moved (usually).
# -----------------------------------------------------------------------
print("\n--- 3. Shape, Reshape, Flatten ---")

a = np.arange(12)
print("Original (12 elements):", a, "  shape:", a.shape)

# reshape: reinterpret the flat buffer as a different shape
# The TOTAL number of elements must be the same (12 = 3*4 = 2*6 = 1*12 = ...)
a_3x4 = a.reshape(3, 4)
print("\nreshaped to (3,4):\n", a_3x4)

a_2x2x3 = a.reshape(2, 2, 3)
print("\nreshaped to (2,2,3):\n", a_2x2x3)

# -1 in reshape means "figure out this dimension automatically"
a_auto = a.reshape(4, -1)    # 12/4 = 3, so shape is (4,3)
print("\nreshaped (4,-1) → (4,3):\n", a_auto)

# flatten(): always returns a COPY as 1D
# ravel():   returns a VIEW when possible (no copy)
a_flat = a_3x4.flatten()
a_ravel = a_3x4.ravel()
print(f"\nflatten: {a_flat}  (copy: {not np.shares_memory(a_3x4, a_flat)})")
print(f"ravel:   {a_ravel}  (copy: {not np.shares_memory(a_3x4, a_ravel)})")

# -----------------------------------------------------------------------
# SECTION 4: Strides — the low-level representation
#
# A NumPy array's strides tuple tells you: to advance by 1 element along
# axis k, how many BYTES do you need to move in memory?
#
# For a C-contiguous (row-major) (3,4) array of float64:
#   - Moving along axis 1 (columns): 1 element = 8 bytes
#   - Moving along axis 0 (rows):    1 row = 4 columns × 8 bytes = 32 bytes
#
# Strides are what make transpose O(1): instead of copying data, NumPy
# just swaps the stride values!  A (3,4) array transposed to (4,3)
# still has the same bytes; only the interpretation changes.
# -----------------------------------------------------------------------
print("\n--- 4. Strides (the real magic of NumPy) ---")

a = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10,11,12]], dtype=np.float64)

print(f"Array shape:   {a.shape}")
print(f"Itemsize:      {a.itemsize} bytes")
print(f"Strides:       {a.strides}  bytes")
print(f"  axis 0 stride = {a.strides[0]} bytes = {a.strides[0]//a.itemsize} elements (= ncols)")
print(f"  axis 1 stride = {a.strides[1]} bytes = {a.strides[1]//a.itemsize} element")

# Now transpose — no data is copied, only strides are swapped!
a_T = a.T
print(f"\nTransposed shape:   {a_T.shape}")
print(f"Transposed strides: {a_T.strides}  ← strides SWAPPED, no data moved")
print(f"Same memory?        {np.shares_memory(a, a_T)}")

# Prove it: modifying the transposed view modifies the original
a_T[0, 0] = 999
print(f"\nAfter a_T[0,0] = 999:  a[0,0] = {a[0,0]}  (same memory!)")
a_T[0, 0] = 1   # restore

# Manual stride trick: step through every other element
every_other = np.lib.stride_tricks.as_strided(
    a.ravel()[:6],
    shape=(3,),
    strides=(a.itemsize * 2,)   # step 2 elements at a time
)
print(f"\nEvery other element via strides: {every_other}")
print("(This is how slicing with step>1 works internally)")

print("\nKey insight:")
print("  NumPy arrays are just a pointer + shape + strides + dtype.")
print("  Operations like transpose, reshape, slice often only change")
print("  the shape/strides metadata — not the underlying bytes.")

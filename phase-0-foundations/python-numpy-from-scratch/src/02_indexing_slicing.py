"""
02_indexing_slicing.py

Master NumPy's three indexing styles:
  1. Basic indexing / slicing  → returns VIEWS (no copy!)
  2. Boolean (mask) indexing   → returns copies
  3. Fancy (integer array) indexing → returns copies

Understanding views vs copies is critical:
  - views share memory with the original array (fast, but mutations propagate)
  - copies are independent (safe, but cost memory)

Run: python3 src/02_indexing_slicing.py
"""

import numpy as np

print("=" * 60)
print("NumPy Indexing and Slicing")
print("=" * 60)

# -----------------------------------------------------------------------
# SECTION 1: Basic indexing — same as Python lists but multi-dimensional
# -----------------------------------------------------------------------
print("\n--- 1. Basic Indexing ---")

a = np.array([[10, 20, 30],
              [40, 50, 60],
              [70, 80, 90]])

print("Array:\n", a)
print("a[0]         :", a[0])         # entire first row
print("a[1, 2]      :", a[1, 2])      # row 1, col 2 → 60
print("a[-1]        :", a[-1])        # last row → [70, 80, 90]
print("a[-1, -1]    :", a[-1, -1])    # bottom-right → 90

# -----------------------------------------------------------------------
# SECTION 2: Slicing — VIEWS not copies
#
# Python slice syntax:   start : stop : step
# NumPy extends it to multiple dimensions, separated by commas.
#
# Critical: NumPy slices return VIEWS, meaning the slice shares memory
# with the original array.  Modifying the slice modifies the original.
# -----------------------------------------------------------------------
print("\n--- 2. Slicing — Views ---")

b = np.arange(1, 13).reshape(3, 4)
print("b:\n", b)

# Slice: all rows, first 2 columns
s = b[:, :2]
print("\nb[:, :2] =\n", s)

# Prove it's a VIEW by modifying it
print(f"Shares memory with b? {np.shares_memory(b, s)}")
s[0, 0] = 999
print(f"After s[0,0] = 999,  b[0,0] = {b[0,0]}  ← original changed!")
s[0, 0] = 1   # restore

# To get a COPY, use .copy()
c = b[:, :2].copy()
c[0, 0] = 999
print(f"\nAfter copy: b[0,0] = {b[0,0]}  ← original unchanged")

# Slicing examples
print("\n2D slicing examples on b =\n", b)
print("b[0, :]      → first row:  ", b[0, :])
print("b[:, 0]      → first col:  ", b[:, 0])
print("b[1:, 1:]    → bottom-right:\n", b[1:, 1:])
print("b[::2, ::2]  → every other row and col:\n", b[::2, ::2])
print("b[::-1, :]   → rows reversed:\n", b[::-1, :])

# -----------------------------------------------------------------------
# SECTION 3: Boolean (mask) indexing
#
# Create a boolean array of the same shape.  Then use it as an index.
# Only elements where the mask is True are selected.
#
# Returns a COPY (the selected elements may not be contiguous in memory).
# -----------------------------------------------------------------------
print("\n--- 3. Boolean (Mask) Indexing ---")

data = np.array([3, -1, 7, -4, 5, -2, 8])
print("data:", data)

# A boolean mask has the same shape as data
mask = data > 0
print("mask (data > 0):", mask)    # [True False True False True False True]

# Apply the mask — selects only where mask is True
positive = data[mask]
print("data[mask]:", positive)

# One-liner shorthand
print("data[data > 0]:", data[data > 0])

# Boolean indexing with assignment — set negatives to zero
data_copy = data.copy()
data_copy[data_copy < 0] = 0
print("After setting negatives to 0:", data_copy)

# 2D boolean mask
matrix = np.arange(1, 10).reshape(3, 3)
print("\nmatrix:\n", matrix)
even_mask = (matrix % 2 == 0)
print("even_mask:\n", even_mask)
print("Even elements:", matrix[even_mask])   # 1D array of selected elements

# -----------------------------------------------------------------------
# SECTION 4: Fancy (integer array) indexing
#
# Use an array of integers as indices.  You pick specific elements in
# any order, potentially repeating.
#
# Returns a COPY because the selected elements may be scattered in memory.
# -----------------------------------------------------------------------
print("\n--- 4. Fancy (Integer Array) Indexing ---")

a = np.array([10, 20, 30, 40, 50])
idx = np.array([0, 2, 4, 2, 1])    # pick elements at these positions
print("a:", a)
print("idx:", idx)
print("a[idx]:", a[idx])    # [10, 30, 50, 30, 20] — with repetition!

# Reordering / shuffling
order = np.array([4, 3, 2, 1, 0])
print("a reversed with fancy index:", a[order])

# 2D fancy indexing: pick specific (row, col) pairs
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])
rows = np.array([0, 1, 2])
cols = np.array([0, 1, 2])
print("\nmatrix:\n", matrix)
print("Diagonal elements (fancy):", matrix[rows, cols])

# -----------------------------------------------------------------------
# SECTION 5: np.where — vectorized conditional selection
#
# np.where(condition, value_if_true, value_if_false)
# Applies element-wise — the entire array at once, no Python loop.
# -----------------------------------------------------------------------
print("\n--- 5. np.where ---")

x = np.array([-3, -1, 0, 2, 5, -7, 4])
print("x:", x)

# Replace negatives with 0 (ReLU — the activation function in neural nets!)
relu = np.where(x > 0, x, 0)
print("ReLU (np.where(x>0, x, 0)):", relu)

# More complex: sign function
sign = np.where(x > 0, 1, np.where(x < 0, -1, 0))
print("Sign function:", sign)

# np.where with no value args: returns INDICES of True elements
indices = np.where(x > 0)
print("Indices where x > 0:", indices[0])   # returns a tuple of arrays

print("\nSummary: view vs copy")
print("  Slicing (a[1:3]) → VIEW  (shares memory, fast)")
print("  Boolean index    → COPY  (elements may not be contiguous)")
print("  Fancy index      → COPY  (arbitrary scatter, can't be a view)")

"""
03_broadcasting.py

Broadcasting is NumPy's rule for performing operations between arrays of
DIFFERENT shapes, without actually copying data.

The mathematical model: if two arrays have shapes that are "compatible",
NumPy conceptually stretches the smaller array to match the larger one.
But no memory is actually allocated for the stretched copy — the CPU reads
the same data multiple times instead.

Broadcasting rules (applied right-to-left on shape tuples):
  1. If arrays have different numbers of dimensions, prepend 1s to the
     shape of the smaller one.
  2. Dimensions are compatible if they are equal OR one of them is 1.
  3. The output shape in each dimension is the maximum of the two.
  4. If neither is 1 and they're not equal → error.

Run: python3 src/03_broadcasting.py
"""

import numpy as np

print("=" * 60)
print("NumPy Broadcasting")
print("=" * 60)

# -----------------------------------------------------------------------
# SECTION 1: Scalar broadcasting (the simplest case)
#
# Adding a scalar to an array is the original broadcasting:
# the scalar is "broadcast" to match every element's position.
# -----------------------------------------------------------------------
print("\n--- 1. Scalar Broadcasting ---")

a = np.array([1, 2, 3, 4, 5])
result = a * 3
print(f"a * 3 = {result}")
print("Conceptually: [1,2,3,4,5] * [3,3,3,3,3]")
print("Actual: the scalar 3 is read 5 times — no copy of [3,3,3,3,3] is made")

# -----------------------------------------------------------------------
# SECTION 2: Adding a 1D array to a 2D array
#
# Common case: add a bias vector to every row of a matrix
# Shape (4,3) + shape (3,) → the (3,) is broadcast across all 4 rows
# -----------------------------------------------------------------------
print("\n--- 2. Adding (3,) to (4,3) Matrix ---")

matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9],
                   [10,11,12]])

bias = np.array([100, 200, 300])   # shape (3,)

print(f"matrix shape: {matrix.shape}")
print(f"bias shape:   {bias.shape}")
print()

# Rule application:
# matrix: (4, 3)
# bias:   (   3)  ← prepend 1 → (1, 3)
# Compatible? dim 0: 4 vs 1 → OK (broadcast bias)
#             dim 1: 3 vs 3 → OK (equal)
# Output shape: (4, 3)

result = matrix + bias
print("matrix + bias:\n", result)
print("\nEach ROW of matrix had [100,200,300] added to it.")
print("bias was NOT copied into a (4,3) array — the CPU added it 4 times.")

# -----------------------------------------------------------------------
# SECTION 3: Column broadcasting — need to reshape
#
# If you want to add a COLUMN vector to each column of a matrix, you must
# ensure the column vector has shape (n, 1), not (n,).
# -----------------------------------------------------------------------
print("\n--- 3. Column Vector Broadcasting ---")

col_vec = np.array([10, 20, 30, 40]).reshape(4, 1)   # shape (4,1)
print(f"col_vec shape: {col_vec.shape}")
print(f"matrix shape:  {matrix.shape}")

# matrix: (4, 3)
# col:    (4, 1) → broadcast col across the 3 columns
result_col = matrix + col_vec
print("matrix + col_vec (each COLUMN offset by [10,20,30,40]):\n", result_col)

# -----------------------------------------------------------------------
# SECTION 4: Outer product via broadcasting
#
# The outer product of two 1D vectors a (shape m,) and b (shape n,) is the
# (m,n) matrix where result[i,j] = a[i] * b[j].
#
# With broadcasting:
#   a.reshape(m,1) has shape (m, 1)
#   b has shape (n,)  → treated as (1, n)
#   Product is (m, n): a[i]*b[j] for all i,j
#
# No copy of any expanded array is made.
# -----------------------------------------------------------------------
print("\n--- 4. Outer Product via Broadcasting ---")

a = np.array([1, 2, 3])       # shape (3,)
b = np.array([10, 20, 30, 40]) # shape (4,)

outer = a.reshape(3, 1) * b   # shapes: (3,1) * (4,) → (3,4)
print(f"a shape: {a.shape},  b shape: {b.shape}")
print(f"a.reshape(3,1) * b → shape {outer.shape}:")
print(outer)

# Verify: outer[i,j] == a[i] * b[j]
assert outer[2, 3] == a[2] * b[3], "Outer product formula failed"
print(f"\nVerification: outer[2,3] = {outer[2,3]} == a[2]*b[3] = {a[2]}*{b[3]} = {a[2]*b[3]}")

# Compare with np.outer
outer_np = np.outer(a, b)
assert np.array_equal(outer, outer_np)
print("Matches np.outer(): yes")

# -----------------------------------------------------------------------
# SECTION 5: Broadcasting rules walkthrough
# -----------------------------------------------------------------------
print("\n--- 5. Broadcasting Rules Walkthrough ---")

examples = [
    ((3, 4), (4,),    "Bias per column"),
    ((3, 1), (1, 4),  "Matrix outer product"),
    ((4, 1), (3,),    "Error — 4 vs 3 are incompatible"),
    ((1, 5), (5,),    "Works — both are 1 or equal"),
    ((3, 4, 5), (4, 5), "3D array + 2D: prepend 1 to (4,5) → (1,4,5)"),
]

for shapeA, shapeB, note in examples:
    try:
        A = np.zeros(shapeA)
        B = np.zeros(shapeB)
        C = A + B
        print(f"  {str(shapeA):15s} + {str(shapeB):10s} → {str(C.shape):15s}  [{note}]")
    except ValueError as e:
        print(f"  {str(shapeA):15s} + {str(shapeB):10s} → ERROR: {note}")

# -----------------------------------------------------------------------
# SECTION 6: Why broadcasting is fast
# -----------------------------------------------------------------------
print("\n--- 6. Why Broadcasting is Fast ---")
print("""
Broadcasting is fast because:
  1. NO data is copied into the "stretched" shape.
     Instead, the stride for that dimension is set to 0.
     A stride of 0 means: read the SAME data for every step along that axis.

  2. Example: broadcasting a (3,) vector across a (4,3) matrix:
     The vector's virtual (1,3) shape has stride (0, itemsize).
     The inner loop reads row 0 of the vector for row 0, 1, 2, 3 of the matrix
     — the same 3 values, reused 4 times.

  3. This is cache-friendly: the small vector fits in CPU cache.
     Reading it 4 times has near-zero memory bandwidth cost.

  Contrast with Python:
     Adding [100,200,300] to each of 4 rows in a Python list would require
     a Python loop with 4 iterations and 3 additions each = 12 Python
     interpreter steps.
     NumPy does all 12 additions in C, with SIMD instructions doing 4+ at once.
""")

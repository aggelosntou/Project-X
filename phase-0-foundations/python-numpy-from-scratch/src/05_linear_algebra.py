"""
05_linear_algebra.py

Linear algebra with NumPy — connecting the code to the mathematics.

Topics:
  1. Matrix multiplication (@ operator, np.dot)
  2. Solving linear systems Ax = b
  3. SVD (Singular Value Decomposition)
  4. Eigendecomposition
  5. Matrix norms and properties

Run: python3 src/05_linear_algebra.py
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)

print("=" * 60)
print("Linear Algebra with NumPy")
print("=" * 60)

# -----------------------------------------------------------------------
# SECTION 1: Matrix multiplication
#
# The @ operator (PEP 465, Python 3.5+) is the matrix product.
# For matrices A (m×k) and B (k×n):
#   (A @ B)[i,j] = sum_{p=0}^{k-1} A[i,p] * B[p,j]
#
# This is the standard mathematical matrix product.
# ----------------------------------------------------------------------- */
print("\n--- 1. Matrix Multiplication ---")

A = np.array([[1, 2, 3],
              [4, 5, 6]])        # shape (2,3)

B = np.array([[7,  8],
              [9,  10],
              [11, 12]])         # shape (3,2)

# A @ B is the (2,2) matrix product
C = A @ B
print(f"A shape: {A.shape},  B shape: {B.shape}")
print(f"A @ B:\n{C}")

# Verify manually: C[0,0] = 1*7 + 2*9 + 3*11 = 7+18+33 = 58
assert C[0,0] == 1*7 + 2*9 + 3*11, "Matrix product formula wrong"
print(f"\nC[0,0] = {C[0,0]} = 1×7 + 2×9 + 3×11 = {1*7}+{2*9}+{3*11} ✓")

# np.dot is equivalent to @ for 2D arrays
C2 = np.dot(A, B)
assert np.array_equal(C, C2)
print("np.dot(A, B) == A @ B: True")

# Dot product of two vectors (inner product)
u = np.array([1.0, 2.0, 3.0])
v = np.array([4.0, 5.0, 6.0])
dot = u @ v                         # scalar
dot2 = np.dot(u, v)
print(f"\nVector dot product u·v = {dot} = {1*4}+{2*5}+{3*6}")

# Note: for 1D arrays, @ gives the inner product (scalar)
# For 2D, @ gives the matrix product (matrix)

# -----------------------------------------------------------------------
# SECTION 2: Solving Ax = b
#
# Given matrix A and vector b, find x such that A @ x = b.
# This is equivalent to x = A^{-1} b, but computing the inverse explicitly
# is numerically unstable and slow.  np.linalg.solve uses LU factorisation.
# -----------------------------------------------------------------------
print("\n--- 2. Solving Linear Systems: Ax = b ---")

# System of equations:
#   2x + 3y = 8
#   5x - 1y = 3
A_sys = np.array([[2.0, 3.0],
                  [5.0, -1.0]])
b_sys = np.array([8.0, 3.0])

x = np.linalg.solve(A_sys, b_sys)
print(f"System:  2x + 3y = 8")
print(f"         5x -  y = 3")
print(f"Solution: x = {x[0]:.4f}, y = {x[1]:.4f}")

# Verify: A @ x should equal b
residual = np.linalg.norm(A_sys @ x - b_sys)
print(f"Residual ||Ax - b|| = {residual:.2e}  (should be ~0)")

# Batch solve (multiple right-hand sides simultaneously)
B_rhs = np.array([[8.0, 1.0],   # two b vectors
                  [3.0, 2.0]])
X = np.linalg.solve(A_sys, B_rhs)
print(f"\nBatch solve (2 RHS) result shape: {X.shape}")

# -----------------------------------------------------------------------
# SECTION 3: SVD — Singular Value Decomposition
#
# Every matrix A (m×n) can be decomposed as:
#   A = U @ diag(s) @ V^T
#
# Where:
#   U   is (m×m) orthogonal: U^T @ U = I  (left singular vectors)
#   s   is a 1D array of singular values, sorted descending
#   V^T is (n×n) orthogonal  (rows are right singular vectors)
#
# SVD is the foundation of:
#   - PCA (principal components are the right singular vectors)
#   - Low-rank approximation (compress a matrix)
#   - Solving least-squares problems
#   - Pseudo-inverse (np.linalg.pinv uses SVD)
# -----------------------------------------------------------------------
print("\n--- 3. Singular Value Decomposition (SVD) ---")

A_svd = np.array([[3.0, 2.0, 1.0],
                  [0.0, 2.0, 0.0],
                  [1.0, 0.0, 2.0]])

U, s, Vt = np.linalg.svd(A_svd)
print(f"A shape: {A_svd.shape}")
print(f"U shape: {U.shape}  (orthogonal: U@U.T ≈ I)")
print(f"s shape: {s.shape}  singular values: {np.round(s, 3)}")
print(f"Vt shape: {Vt.shape}")

# Reconstruct A from the decomposition
A_reconstructed = U @ np.diag(s) @ Vt
print(f"\nReconstruction error: {np.linalg.norm(A_svd - A_reconstructed):.2e}")

# Low-rank approximation: keep only the k largest singular values
# This is how JPEG compression and word embeddings work!
k = 2
A_approx = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
error = np.linalg.norm(A_svd - A_approx, 'fro')
print(f"\nRank-{k} approximation (Frobenius error): {error:.4f}")
print(f"Original:\n{A_svd}")
print(f"Approx:\n{np.round(A_approx, 3)}")

# -----------------------------------------------------------------------
# SECTION 4: Eigendecomposition
#
# For a SQUARE matrix A, find eigenvalues λ and eigenvectors v such that:
#   A @ v = λ * v
#
# The eigenvectors are the "principal directions" — multiplying by A
# only scales them, doesn't rotate them.
#
# Applications:
#   - PCA: eigenvectors of the covariance matrix are principal components
#   - PageRank: leading eigenvector of the link matrix
#   - Physics: normal modes of vibration, quantum states
# -----------------------------------------------------------------------
print("\n--- 4. Eigendecomposition ---")

# Symmetric matrix — real eigenvalues guaranteed
A_eig = np.array([[4.0, 2.0, 1.0],
                  [2.0, 5.0, 3.0],
                  [1.0, 3.0, 6.0]])

eigenvalues, eigenvectors = np.linalg.eigh(A_eig)   # eigh = symmetric/Hermitian
# eigenvalues are sorted ascending for eigh
print(f"Eigenvalues:  {np.round(eigenvalues, 4)}")
print(f"Eigenvectors (columns):\n{np.round(eigenvectors, 4)}")

# Verify: A @ v = λ * v for each (λ, v) pair
for i in range(3):
    lam = eigenvalues[i]
    v   = eigenvectors[:, i]
    lhs = A_eig @ v
    rhs = lam * v
    err = np.linalg.norm(lhs - rhs)
    print(f"  λ_{i} = {lam:.4f}:  ||A@v - λv|| = {err:.2e}")

# Reconstruct A from eigen-decomposition: A = Q @ diag(λ) @ Q^T
Q = eigenvectors
A_rec = Q @ np.diag(eigenvalues) @ Q.T
print(f"\nReconstruction error: {np.linalg.norm(A_eig - A_rec):.2e}")

# -----------------------------------------------------------------------
# SECTION 5: Norms and other useful operations
# -----------------------------------------------------------------------
print("\n--- 5. Norms and Useful Operations ---")

v = np.array([3.0, 4.0])
print(f"v = {v}")
print(f"L2 norm (Euclidean):  ||v||_2 = {np.linalg.norm(v, 2):.4f}  (sqrt(9+16)=5)")
print(f"L1 norm (Manhattan):  ||v||_1 = {np.linalg.norm(v, 1):.4f}  (3+4=7)")
print(f"L∞ norm (max):        ||v||_∞ = {np.linalg.norm(v, np.inf):.4f}")

M = np.array([[1.0, 2.0], [3.0, 4.0]])
print(f"\nMatrix M =\n{M}")
print(f"Frobenius norm: {np.linalg.norm(M, 'fro'):.4f}  (sqrt of sum of squares)")
print(f"Determinant:    {np.linalg.det(M):.4f}")
print(f"Trace:          {np.trace(M):.4f}  (sum of diagonal = sum of eigenvalues)")
print(f"Rank:           {np.linalg.matrix_rank(M)}")

# Matrix inverse (use solve instead in practice — this is just for demonstration)
M_inv = np.linalg.inv(M)
print(f"\nM^{{-1}}:\n{M_inv}")
print(f"M @ M^{{-1}} ≈ I:\n{np.round(M @ M_inv, 10)}")

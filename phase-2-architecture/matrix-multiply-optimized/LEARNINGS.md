# Matrix Multiply — Learnings

## Why Loop Order Matters

For C = A × B (row-major storage), the ikj order is optimal for naive code:

```c
// ikj — BEST naive order
for i:
  for k:
    a = A[i][k]        // load once
    for j:
      C[i][j] += a * B[k][j]   // B sequential, C sequential
```

The inner loop accesses `B[k][j]` and `C[i][j]` with unit stride — sequential, cache-friendly.

Compare with ijk:
```c
// ijk — BAD
for i:
  for j:
    for k:
      C[i][j] += A[i][k] * B[k][j]   // B[k][j] has stride N!
```

The inner loop accesses `B[k][j]` with stride N (jumping N elements for each k step). For N=1024 with float=4 bytes, each step jumps 4 KB — a different cache line every single access.

## SIMD — Single Instruction, Multiple Data

CPUs have wide registers that can hold multiple values. AVX2 gives you 256-bit registers holding 8 × float32. One `_mm256_fmadd_ps` instruction does 8 multiply-add operations simultaneously.

```c
// Scalar (1 float/op)
for j: C[i][j] += a * B[k][j]

// AVX2 SIMD (8 floats/op)
a_vec = _mm256_set1_ps(A[i][k])           // [a,a,a,a,a,a,a,a]
b_vec = _mm256_loadu_ps(&B[k][j])         // [B[k][j]..B[k][j+7]]
c_vec = _mm256_loadu_ps(&C[i][j])         // [C[i][j]..C[i][j+7]]
c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec)  // 8 FMAs at once
_mm256_storeu_ps(&C[i][j], c_vec)
```

Speedup: 8× compute throughput (ignoring memory bandwidth limits).

Modern compilers (gcc -O3, clang -O2) auto-vectorize simple loops. Explicit intrinsics give more control over vectorization decisions.

## Tiling — Working Set Fits in L1

Without tiling, for a 1024×1024 matrix:
- B alone is 4 MB — doesn't fit in L1 (32 KB) or L2 (256 KB)
- Every access to a new row of B is a cache miss

With tiling (T=32):
- Tile size: 32×32×4 = 4 KB
- Three tiles (A_tile, B_tile, C_tile) = 12 KB — fits in L1!
- Each tile of B is reused T=32 times while computing C_tile
- Cache misses per byte: O(1/T) instead of O(1)

The math: without tiling, B is read O(N) times (once per row of A). With tiling of size T, B tiles are read O(N/T) times each. For T=32, N=1024: 1024 → 32 reads of each B tile.

## Roofline Model

The roofline model bounds performance:
```
attainable_perf = min(peak_GFLOPS, bandwidth_GB/s × arithmetic_intensity)
```

Arithmetic intensity = FLOPs / bytes of memory traffic.

For matmul:
- FLOPs: 2N³
- Memory traffic (no cache): reads A+B (2N² floats) + writes C (N² floats) = 3N² × 4 bytes
- Arithmetic intensity = 2N³ / (12N²) = N/6

For N=1024: intensity = 1024/6 ≈ 171 FLOPs/byte

Typical DRAM bandwidth: 50 GB/s
Ceiling from bandwidth: 50 × 171 = 8550 GFLOPS — higher than compute peak
So matmul is COMPUTE BOUND for large N — cache is not the bottleneck with tiling!

## Why BLAS Is So Fast

BLAS (Basic Linear Algebra Subroutines) implementations like OpenBLAS and MKL achieve >90% of theoretical peak through:

1. **Multiple registers**: Keep 4×8 = 32 output elements in registers, amortizing load/store
2. **Software pipelining**: Interleave loads and computations to hide memory latency
3. **Micro-kernel**: Hand-written assembly for the innermost tile computation
4. **Packing**: Copy input tiles into contiguous memory to avoid strided access
5. **Multi-threading**: Parallelize across all CPU cores

The progression from our naive (2% peak) to tiled+SIMD (~30% peak) to BLAS (90% peak) represents decades of research.

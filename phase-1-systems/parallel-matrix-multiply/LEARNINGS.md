# Learnings — Parallel Matrix Multiplication

## 1. Why ikj beats ijk: cache access patterns

For an N×N matrix stored row-major, element `[i][j]` is at offset `i*N+j`.

### ijk loop (classic, slow)

```c
for i:
  for j:        // C[i][j] is the target
    for k:      // inner: vary k
      C[i][j] += A[i][k] * B[k][j];
      //                      ^^^^
      //  B[k][j] for fixed j: k steps through a COLUMN of B
      //  stride = N floats = N×4 bytes
      //  For N=512: stride = 2,048 bytes — likely a cache miss every step
```

The CPU's hardware prefetcher works well for **sequential** (stride-1)
access.  Stride-N access defeats it entirely.  For N=512, L1 cache line
(64 bytes = 16 floats) reuse drops to essentially zero on B.

### ikj loop (fast)

```c
for i:
  for k:        // fix a row of A and a row of B
    a_ik = A[i][k];   // scalar — held in register
    for j:      // inner: vary j
      C[i][j] += a_ik * B[k][j];
      //                 ^^^^^^
      //  B[k][j] for fixed k: j steps through ROW k of B → stride 1
      //  C[i][j] for fixed i: j steps through ROW i of C → stride 1
```

Now both B and C are accessed sequentially.  The prefetcher can keep up.
Cache line reuse is ~16 elements per load (for float, 64-byte lines).

**Rule of thumb**: the loop variable that appears in the innermost position
should be the one that causes sequential memory access for all arrays used.

---

## 2. Why transposing B helps

Even with ikj, the middle loop `for k` accesses `A[i][k]` sequentially
(row i of A), but `B[k][j]` for varying k is a column access when you look
at the full picture across both the k and j loops.

The transpose trick copies B into Bt where `Bt[j][k] = B[k][j]`.

After transposing:

```c
for i:
  for j:
    sum = 0
    for k:      // inner: vary k
      sum += A[i][k] * Bt[j][k];
      //      ^^^^^^   ^^^^^^^^
      //  A[i][k]: row i of A    → sequential (stride 1)
      //  Bt[j][k]: row j of Bt  → sequential (stride 1)
```

Cost: one O(N^2) transpose up front.  For large N that transpose may itself
cause cache misses, but the main O(N^3) multiply runs fully sequentially.
Net benefit: typically 2–5x over naive for N ≥ 256.

---

## 3. What tiling achieves

Without tiling, for each output element `C[i][j]` we load:
- Row i of A: N floats
- Column j of B: N floats (or row j of Bt)

For N=1024, one row = 4,096 bytes = 64 cache lines.  The L1 cache (32 KB
typical) can hold 8 rows simultaneously.  With N=1024 rows, data is
constantly evicted before it can be reused.

### Tiling principle

Divide A, B, C into T×T sub-matrices (tiles).  Process one tile triple
`(A_tile, B_tile, C_tile)` at a time:

```
Working set = 3 tiles × T×T × 4 bytes = 12 T^2 bytes
```

| T  | Working set | Fits in |
|----|-------------|---------|
| 16 | 3,072 B     | L1 (32 KB) easily |
| 32 | 12,288 B    | L1 (32 KB) easily |
| 64 | 49,152 B    | L2 (256 KB) easily |
|128 | 196,608 B   | L2/L3 boundary |

For T=32 and L1=32 KB: the three tiles fit comfortably.  Each element in
a tile is reused T times before the tile leaves L1.  Cache miss rate drops
by a factor of ~T compared to the untiled case.

### Loop structure

```c
for ii in 0..N step T:          // tile row index
  for jj in 0..N step T:        // tile col index
    for kk in 0..N step T:      // tile depth index
      // inner: process one T×T tile triple
      for i in ii..ii+T:
        for k in kk..kk+T:
          a_ik = A[i][k]
          for j in jj..jj+T:
            C[i][j] += a_ik * B[k][j]
```

The six-loop structure ensures that when the inner triple runs, all of
`A[ii..ii+T][kk..kk+T]`, `B[kk..kk+T][jj..jj+T]`, and
`C[ii..ii+T][jj..jj+T]` reside in L1.

---

## 4. Amdahl's Law

### Formula

If a fraction `s` of the algorithm is **serial** (cannot be parallelised),
then with `p` threads the maximum speedup is:

```
S(p) = 1 / (s + (1-s)/p)
```

### Derivation

Total work = 1 (normalised).
- Serial part takes time `s` regardless of thread count.
- Parallel part takes time `(1-s)/p` with p threads.
- Total time = `s + (1-s)/p`.
- Speedup = original time / new time = `1 / (s + (1-s)/p)`.

### Implications

| Serial fraction s | Max speedup (p→∞) |
|---|---|
| 1% (0.01) | 100x |
| 5% (0.05) | 20x |
| 10% (0.10) | 10x |
| 50% (0.50) | 2x |

Even 5% serial code caps you at 20x speedup no matter how many cores you add.

### Example with measurements

Suppose for N=512, tiled_openmp:

```
1 thread:  120 ms  → speedup 1.00, efficiency 100%
2 threads:  62 ms  → speedup 1.94, efficiency  97%
4 threads:  33 ms  → speedup 3.64, efficiency  91%
8 threads:  21 ms  → speedup 5.71, efficiency  71%
16 threads: 18 ms  → speedup 6.67, efficiency  42%  ← below 50%
```

Here the estimated serial fraction is s ≈ 0.02 (2%), giving a theoretical
maximum speedup of 50x — but memory bandwidth saturates long before that.

---

## 5. Why efficiency drops with more threads

Three causes:

### a) Synchronisation overhead

OpenMP `parallel for` has a fork/join cost at each loop boundary.  This
cost is roughly constant (µs range), so as work-per-thread shrinks (more
threads = less work each), the overhead becomes a larger fraction.

### b) Memory bandwidth saturation

Matrix multiply reads A and B from RAM.  The total bandwidth needed is:

```
bandwidth = 2 * N^2 * sizeof(float) * (N/T) bytes/second
```

A typical server has ~100 GB/s of memory bandwidth.  Once all threads
together saturate the memory bus, adding more threads provides no
additional throughput — and the synchronisation overhead still grows.

### c) False sharing

If two threads write to adjacent cache lines (within 64 bytes of each
other), every write by one core invalidates the other core's cached copy.
In our implementation, threads own separate rows of C (each row ≥ 512 bytes
for N≥128), so false sharing is not a concern here.  But it is a common
pitfall in naive parallel reductions.

---

## 6. GFLOPS calculation

Matrix multiply C = A*B for N×N matrices:
- N^2 dot products (one per output element)
- Each dot product: N multiplications + N additions = 2N operations
- Total: 2 * N^3 floating-point operations

```
GFLOPS = 2 * N^3 / time_seconds / 1e9
```

Peak theoretical performance of a modern core running at 3 GHz with
vectorised FMA:
```
peak = 3e9 cycles/sec × 8 FP32 per AVX2 lane × 2 (FMA) = 48 GFLOPS/core
```

Our tiled_openmp implementation with 8 cores might reach 50–150 GFLOPS
depending on hardware — well below peak because of memory bandwidth limits
and loop overhead.  Reaching near-peak requires BLAS-level optimisation
(SIMD intrinsics, register blocking, software prefetch).

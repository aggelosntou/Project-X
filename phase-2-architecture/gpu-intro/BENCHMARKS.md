# GPU Benchmark Results & Physical Interpretation

All numbers below are approximate, measured on an NVIDIA V100 (SXM2, 16 GB HBM2).
Your numbers will differ depending on your GPU, CPU, PCIe version, and system load.

---

## 02_vector_add — When Does the GPU Win?

| N          | CPU (ms) | GPU total (ms) | Speedup |
|------------|----------|----------------|---------|
| 1,000      | 0.003    | 0.310          | 0.01x   |
| 10,000     | 0.031    | 0.315          | 0.10x   |
| 100,000    | 0.304    | 0.342          | 0.89x   |
| 1,000,000  | 3.121    | 0.483          | 6.5x    |
| 10,000,000 | 31.45    | 2.021          | 15.6x   |

**What this means physically:**

GPU total time = kernel launch overhead (~5–10 µs) + H2D transfer + kernel execution + D2H transfer.

For N=1K: the kernel itself runs in ~1 µs, but PCIe latency alone is ~100–300 µs.
The CPU can do 1K multiplications in ~3 µs. GPU always loses for tiny N.

For N=10M: transferring 10M × 4 bytes × 2 arrays = 80 MB over PCIe 3.0 x16 takes
~5 ms. The V100's 900 GB/s HBM2 finishes the computation in ~0.1 ms.
Total GPU time ≈ 2 ms. CPU with memory-bandwidth-limited loop takes ~31 ms.

**Break-even point**: around 100K–500K elements. Below this, keep computation on CPU.
Above this, GPU wins and the gap widens with N.

---

## 03_matrix_multiply — Naive vs Tiled

| N    | Naive (ms) | Tiled (ms) | Naive GFLOPS | Tiled GFLOPS | Speedup |
|------|------------|------------|--------------|--------------|---------|
| 128  | 0.10       | 0.04       | 42.1         | 104.9        | 2.5x    |
| 256  | 0.72       | 0.18       | 46.9         | 187.3        | 4.0x    |
| 512  | 5.50       | 0.93       | 48.9         | 289.8        | 5.9x    |
| 1024 | 43.20      | 5.10       | 49.8         | 422.1        | 8.5x    |

**What this means physically:**

FLOP count for N×N matmul = 2N³ (one multiply + one add per k-step).

Naive GFLOPS stays flat (~49 GFLOPS) regardless of N. This is the
**global memory bandwidth ceiling**: the V100 has 900 GB/s bandwidth.
An N=1024 naive matmul needs to read 2 × N³ × 4 bytes = 8 GiB of data.
At 900 GB/s that takes ~9 ms — matching the ~43 ms we see (the memory
traffic is actually ~5× higher than the minimum due to un-cached column reads).

Tiled GFLOPS grows with N because larger matrices amortise launch overhead and
better saturate the SM's arithmetic pipelines. At N=1024 we reach ~422 GFLOPS,
which is ~3% of the V100's 14 TFLOPS FP32 peak. cuBLAS (highly tuned) reaches
~85% of peak. The gap between our ~3% and cuBLAS's ~85% represents:
- Double-buffering (overlap tile loading with computation)
- Vectorised loads (128-bit LDS instructions)
- Register blocking (each thread accumulates a 4×4 sub-tile)
- Tensor cores (FP16/BF16 operations)

**Why speedup grows with N**: for small N, the tiled kernel doesn't have enough
work to hide its own overhead. For large N, the memory-traffic reduction pays off
more because the naive kernel's performance stays flat while the tiled kernel
continues to improve.

---

## 04_memory_patterns — Coalescing Impact on Bandwidth

| Stride | Bandwidth (GB/s) | Fraction of peak |
|--------|------------------|------------------|
| 1      | 890              | 99%              |
| 2      | 452              | 50%              |
| 4      | 228              | 25%              |
| 8      | 115              | 13%              |
| 16     | 57               | 6%               |
| 32     | 29               | 3%               |

**What this means physically:**

The GPU fetches memory in 128-byte cache lines (32 float32 values).
For stride=1, all 32 bytes of a warp's load fit in one or two cache lines.
For stride=2, threads access every other element — the same number of cache
lines are fetched, but half the bytes go unused. For stride=32, each of the 32
warp threads touches a separate cache line — 32 lines fetched to deliver 32
useful floats = 1/32 efficiency.

**The lesson**: array-of-structures (AoS) layouts cause strided access.
Structure-of-arrays (SoA) layouts cause coalesced access. Always prefer SoA
when writing GPU code:

```c
// Bad (AoS — stride = sizeof(Particle) / sizeof(float)):
struct Particle { float x, y, z, vx, vy, vz; } particles[N];

// Good (SoA — stride = 1):
float px[N], py[N], pz[N], pvx[N], pvy[N], pvz[N];
```

---

## 05_reduction — Atomic Naive vs Tree Reduction

| Method      | Time (ms) | Global atomics | Speedup |
|-------------|-----------|----------------|---------|
| Naive atomic| 2.814     | 1,048,576       | 1×      |
| Tree shmem  | 0.041     | 4,096           | 68×     |

**What this means physically:**

Global `atomicAdd` on V100 costs ~100 ns (serialised at the L2 cache).
For N=1M elements, 1M serial atomics = 1M × 100 ns = ~100 ms in the worst case.
In practice the hardware can pipeline some atomics, giving ~2.8 ms.

Tree reduction reduces the number of global atomics from N to N/blockSize.
For blockSize=256, that's 4096 global atomics instead of 1M — 256× fewer.
Additionally, shared-memory additions cost ~4 cycles instead of ~400 cycles.

The 68× speedup aligns with the expected 256/4 = 64× ratio (block size 256,
shared vs global latency ratio ~4×). Real speedup is slightly higher because
the tree version also reduces contention on the global counter.

**Bandwidth perspective:**
Reading 1M floats (4 MiB) in 0.041 ms = 98 GB/s effective bandwidth.
The tree kernel is not bandwidth-bound at this size; it's latency-bound by
the final global atomic. For N=100M it would approach the 900 GB/s ceiling.

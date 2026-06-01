# Learnings — Benchmark Suite

## What MLPerf Measures

**MLPerf** (now ML Commons) is a standardised benchmark suite for
training and inference so that different hardware can be compared fairly.

Key properties:
- **Fixed workloads** — BERT fine-tuning, ResNet-50 image classification,
  recommendation systems, etc.  The model architecture and dataset are
  specified exactly.
- **Training metric**: time to reach a target quality (e.g., 75.9% top-1
  accuracy on ImageNet).  This prevents "training longer and calling it
  faster hardware."
- **Inference metric**: latency at a target throughput level (Queries Per
  Second / QPS) with a latency SLA (e.g., 99th percentile < 10 ms).
- **Closed division**: hardware vendors must run the exact reference
  implementation.  Only system-level optimisations (batching, precision)
  are allowed.
- **Open division**: any model variant is allowed (NAS, quantisation,
  pruning).  Useful for research comparisons.

Why it matters: before MLPerf, vendors quoted "peak TFLOPS" which
is meaningless for real workloads (matrix multiply never achieves peak
on irregular shapes).  MLPerf forces measurement of end-to-end,
application-level performance.

## Statistical Rigour in Benchmarking

### Why Discard the First Run (Warmup)

The **cold cache effect**: the first call to NumPy (or any library) loads
code pages into the instruction cache, loads weight tensors into the data
cache, and initialises BLAS thread pools.  This can add 5–100× latency
compared to subsequent "warm" runs.

Rule of thumb: discard the first 2–5 runs before measuring.

### Min vs Mean

- **Min of N runs**: use for deterministic operations (matrix multiply,
  attention forward).  The minimum gives the *hardware capability* — what
  the chip can do under ideal conditions.  Higher values are caused by OS
  jitter (scheduler preemption, interrupts, memory page faults).

- **Mean ± std**: use for operations that have inherent variability
  (training steps where batch content matters for compute patterns,
  file I/O, network calls).

### Clock Resolution

`time.perf_counter()` has nanosecond resolution on modern OSes.
For operations < 100 µs, average over many repetitions:

```python
# Wrong: single call for a 1 µs operation
t0 = time.perf_counter()
fast_op()  # 1 µs
print(time.perf_counter() - t0)  # clock jitter ≈ 1 µs — unreliable!

# Correct: loop and divide
N = 10000
t0 = time.perf_counter()
for _ in range(N): fast_op()
print((time.perf_counter() - t0) / N)
```

## Roofline Analysis

The **Roofline Model** (Williams et al., 2009) places every kernel on a
2D plot of arithmetic intensity (FLOP/byte) vs achievable performance
(GFLOP/s).  Two lines form the "roof":

```
                     ┌─ Compute Bound
                     │    (peak GFLOPS)
Perf (GFLOPS/s)     /
                    / ← Memory Bound
                   /    (peak GB/s × intensity)
──────────────────/─────────────────────────────
              ridge point
                     Arithmetic Intensity (FLOP/byte)
```

**Memory-bound** (left of ridge): more compute units won't help.
  Optimization: reduce memory traffic (tiling, fusing, quantisation).

**Compute-bound** (right of ridge): reducing memory traffic won't help.
  Optimization: use faster math (TensorCores, FMA), reduce FLOPs (pruning).

**Attention is memory-bound** (intensity ≈ 0.25 FLOP/byte).
  This is why FlashAttention (fewer DRAM accesses) > tiling (fewer ops).

## Arithmetic Intensity of Attention

```
Q @ K^T:  2 * d_head FLOPs, reads d_head + d_head floats = 8 * d_head bytes
Intensity = 2*d / 8*d = 0.25 FLOP/byte
```

Softmax is even worse:
```
exp + sum + divide over seq_len elements: ~5 FLOPs each
reads seq_len floats: 4 * seq_len bytes
Intensity = 5 / 4 = 1.25 FLOP/byte (still memory-bound)
```

Compare to dense GEMM (N=1024):
```
FLOPs = 2 * 1024^3 ≈ 2.1 × 10^9
Bytes = 2 * 1024^2 * 4 ≈ 8 × 10^6  (if Q and K fit in cache)
Intensity ≈ 256 FLOP/byte  ← compute-bound!
```

## Peak GFLOPS for Your CPU

```
peak_gflops = physical_cores
            × clock_GHz
            × simd_width_floats    # 8 for AVX2 (256-bit / 32-bit), 16 for AVX-512
            × fma_factor           # 2 (one FMA = 1 multiply + 1 add)
```

Example — Apple M2 Pro (12 P-cores, 3.5 GHz, NEON 128-bit):
```
peak = 12 × 3.5 × 4 × 2 = 336 GFLOPS
```

Example — Intel i9-12900K (8 P-cores, 5.2 GHz, AVX2):
```
peak = 8 × 5.2 × 8 × 2 = 665 GFLOPS
```

In practice, NumPy with OpenBLAS/MKL achieves 50–80% of peak for large
square GEMM.  Smaller or non-square matrices achieve less because of
reduced cache reuse and setup overhead.

## Why Batch Size Matters for Inference

At batch_size=1:
- Each forward pass is a **matrix-vector** multiply (GEMV), not GEMM.
- GEMV arithmetic intensity ≈ O(1): for each output element, we read
  one row of W (n bytes) and do n FLOPs → intensity = 1 FLOP/byte.
- GEMV is memory-bandwidth bound — compute units idle.
- Python/NumPy call overhead (2–10 µs) is comparable to compute.

At batch_size=128:
- Each forward pass is a **matrix-matrix** multiply (GEMM).
- GEMM intensity ≈ N FLOP/byte — compute-bound for large N.
- Call overhead is amortised over 128 samples.
- Throughput improves 50–100× vs batch_size=1.

Rule of thumb: **always batch for throughput; use small batches for latency SLAs**.

# Benchmarks — Parallel Matrix Multiplication

## Formulas

### GFLOPS

```
GFLOPS = (2 * N^3) / time_seconds / 1e9
```

For N=512: 2 * 512^3 = 268,435,456 ≈ 2.68 × 10^8 operations.
At 1 GFLOPS that takes 268 ms.  A good tiled implementation might do
10–50 GFLOPS → 5–27 ms.

### Speedup

```
speedup(impl) = time(naive) / time(impl)
```

### Amdahl's Law — theoretical speedup

```
S(p, s) = 1 / (s + (1-s)/p)

where:
  p = number of threads
  s = serial fraction (estimated from 1-thread vs 2-thread timing)
```

### Efficiency

```
efficiency(p) = speedup(p) / p
```

100% efficiency = perfect linear scaling.  In practice, 70–90% is
excellent for memory-bound workloads.

### Memory bandwidth required

```
bytes_read = 2 * N^2 * sizeof(float)  (one pass over A and B)
bandwidth_GB/s = bytes_read / time_seconds / 1e9
```

When the required bandwidth approaches your hardware's memory bandwidth
limit, adding more threads will not help.

---

## Part 1: Expected speedups (relative to naive)

These are approximate ranges observed on modern hardware (x86-64, arm64).
Actual values depend heavily on CPU model, cache sizes, and clock speed.

| N    | naive | transposed | openmp (8T) | tiled_openmp (8T) |
|------|-------|------------|-------------|-------------------|
| 128  | 1.0x  | 1.5–3x     | 4–7x        | 4–8x              |
| 256  | 1.0x  | 2–4x       | 4–7x        | 5–10x             |
| 512  | 1.0x  | 2–5x       | 4–7x        | 6–12x             |
| 1024 | (skip)| 3–6x       | 5–8x        | 7–15x             |

Observations to look for:
- Transposed beats naive by more as N grows (larger matrices = more
  cache misses, more benefit from removing stride-N access on B).
- OpenMP speedup plateaus before reaching the thread count (Amdahl + BW).
- Tiled_openmp is the clear winner for large N.

---

## Part 2: Expected Amdahl's Law table (N=512, tiled_openmp)

Fill in from your actual run:

```
Platform  : _____________________________
OS        : _____________________________
Compiler  : _____________________________  (version)
Max threads: ____
Date      : _____________________________

threads  time_ms   speedup  efficiency  Amdahl_limit
-------  --------  -------  ----------  ------------
1        ______    1.000    100.0%      1.000
2        ______    ______   ______%     ______
4        ______    ______   ______%     ______
8        ______    ______   ______%     ______
16       ______    ______   ______%     ______
...

Estimated serial fraction s ≈ ______  (______%)
Theoretical max speedup     ≈ ______x
Efficiency dropped below 50% at ______ threads
```

---

## What to look for

### Efficiency curve shape

```
efficiency
100% |*
     | *
 80% |  *
     |    *
 60% |       *
     |           *
 40% |                *----  (Amdahl ceiling)
     +--1--2--4--8--16--32-- threads
```

The "knee" of the curve — where efficiency starts dropping sharply — is
where memory bandwidth or synchronisation becomes the bottleneck, not the
serial fraction.

### Diminishing returns

If doubling the thread count gives less than a 1.8x speedup, you have
entered the diminishing-returns regime.  Common causes:
1. Memory bandwidth saturation (profile with `perf stat -e LLC-load-misses`)
2. Serial fraction too large (profile with a single-threaded run and
   identify non-parallel sections)
3. NUMA effects (threads on different sockets access remote memory)

### Interpreting the Amdahl limit column

The "Amdahl limit" in the table is the theoretical maximum speedup for that
thread count given the estimated serial fraction.  If your measured speedup
is significantly below the Amdahl limit, the bottleneck is not the serial
fraction — it is memory bandwidth or synchronisation overhead.

---

## Profiling commands

```bash
# Linux: hardware performance counters
perf stat -e cache-misses,cache-references,LLC-load-misses,instructions \
    ./benchmark 2>&1 | tee perf_output.txt

# Linux: per-function profile
perf record -g ./benchmark && perf report

# macOS: Instruments
xcrun xctrace record --template 'Time Profiler' --launch ./benchmark

# Check memory bandwidth utilisation (Linux)
sudo pcm-memory ./benchmark
```

## Cache miss analysis (rough estimate)

For the naive ikj implementation at N=512:

```
Inner loop iterations: N^3 = 134,217,728
B accesses per iteration: 1 (stride-1 → cache hit after first)
A accesses (hoisted): N^2 = 262,144 (one per k-iteration, then register)
C accesses: N^2 = 262,144 (stride-1, cache hits)

Estimated cache misses: ~N^2 / 16  (one cache line per 16 floats)
                       ≈ 16,384 misses for A/C combined
```

For the ijk implementation at N=512:

```
B[k][j] accessed with stride N: 1 cache miss per access
Inner loop B misses: N^3 / 1 = 134,217,728 misses (worst case)
→ ~8,192x more cache misses than ikj
```

This explains why ijk can be 10–100x slower than ikj for large N.

# Parallel Matrix Multiplication — Amdahl's Law in Practice

This project implements matrix multiplication four different ways, from a
naive triple loop to a cache-tiled OpenMP version, measuring the effect of
cache optimisation and parallelism on real hardware.

## What this project demonstrates

| Concept | Where to see it |
|---|---|
| Cache-friendly loop order (ikj) | `src/matmul.c` — `matmul_naive` |
| Transpose trick (eliminate stride) | `src/matmul.c` — `matmul_transposed` |
| OpenMP thread parallelism | `src/matmul.c` — `matmul_openmp` |
| Cache tiling + OpenMP | `src/matmul.c` — `matmul_tiled_openmp` |
| Amdahl's Law measurement | `src/benchmark.c` — Part 2 |
| GFLOPS calculation | `src/benchmark.c` |

## Build

### Linux (GCC)

```bash
make
```

### macOS — Apple Clang (requires libomp)

```bash
brew install libomp
# Then edit Makefile: uncomment the MACOS lines and adjust the prefix,
# or use:
make CC=gcc-14   # replace 14 with your brew GCC version
```

### Run

```bash
make run
# or
./benchmark
```

## Output structure

```
=== Part 1: Implementation Comparison ===
N       Implementation   time (ms)     GFLOPS
128     naive            ...           ...
128     transposed       ...           ...
128     openmp           ...           ...
128     tiled_openmp     ...           ...
...
1024    (naive skipped)
1024    transposed       ...           ...
...

=== Part 2: Amdahl's Law (N=512, tiled_openmp) ===
threads  time (ms)   speedup  efficiency  Amdahl limit
1        ...         1.000    100.0%      1.000
2        ...         ...      ...%        ...
...
^ efficiency dropped below 50% at N threads (Amdahl ceiling reached)

Estimated serial fraction s ≈ 0.0012  (0.12%)
Theoretical max speedup (p→∞) ≈ 833x  [= 1/s]
```

## What to observe

1. **naive vs transposed**: transposed should be 2–5x faster for large N
   because it eliminates column-stride cache misses on B.

2. **transposed vs openmp**: openmp adds thread parallelism on top of the
   naive loop.  With 8 threads you might see 4–7x speedup (never exactly 8x
   due to synchronisation overhead and memory bandwidth limits).

3. **tiled_openmp**: the best of both — cache reuse inside tiles AND
   thread parallelism across tiles.  Should show the highest GFLOPS.

4. **Amdahl's Law table**: efficiency will start near 100% and drop as
   thread count increases.  The point where efficiency falls below 50% is
   the practical parallelism ceiling for this workload.

## Concepts

See `LEARNINGS.md` for detailed explanations of ikj loop order, the
transpose trick, tiling, and Amdahl's Law.

See `BENCHMARKS.md` for a template to record your measurements and the
formulas to interpret them.

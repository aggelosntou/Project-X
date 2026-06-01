# Cache Experiments

Five empirical experiments that measure CPU cache behavior directly, using nothing but timing.

## Experiments

| # | Program | What it shows |
|---|---|---|
| 01 | `cache_size` | Detect L1/L2/L3 cache sizes by latency vs array size |
| 02 | `cache_line` | Find cache line size (64 bytes) via strided access |
| 03 | `false_sharing` | Two threads thrash a shared cache line (~5x slowdown) |
| 04 | `seq_vs_random` | Sequential is ~10-100x faster than random access |
| 05 | `matrix_traversal` | Column-major traversal is 5-20x slower than row-major |

## Building

```bash
make
```

## Running All

```bash
make run-all
```

Or individually:

```bash
make run-01   # cache size
make run-03   # false sharing
```

## Expected Output (Apple M-series example)

### 01 — Cache Size
```
Array Size (KB) | Elements     | ns/access
1               | 256          | 0.84    ← L1 cache
32              | 8192         | 0.86    ← still L1
64              | 16384        | 2.1     ← L2 spillover
512             | 131072       | 4.2     ← L2 cache
8192            | 2097152      | 8.3     ← L3 cache
32768           | 8388608      | 48.1    ← DRAM
```

### 03 — False Sharing
```
Adjacent (false sharing):  0.820 s
Padded (no false sharing): 0.145 s
Speedup: 5.7x
```

## macOS Note

These experiments use `clock_gettime(CLOCK_MONOTONIC)` which is available on macOS 10.12+.
On Apple Silicon, the L3 cache is very large (8-24 MB depending on chip). The plateau in experiment 01 may shift accordingly.

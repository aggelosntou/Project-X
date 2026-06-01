# Cache Experiments — Learnings

## What Is a Cache?

A CPU cache is a small, fast memory between the processor and main RAM. Modern CPUs have 3 levels:

| Level | Size | Latency | Location |
|---|---|---|---|
| L1 | 32–64 KB | ~4 cycles (~1 ns) | Per core |
| L2 | 256 KB – 1 MB | ~12 cycles (~4 ns) | Per core |
| L3 | 8–64 MB | ~40 cycles (~15 ns) | Shared |
| DRAM | GBs | ~200 cycles (~70 ns) | Off-chip |

The processor always looks in L1 first. On a miss, it looks in L2, then L3, then DRAM. Each level has higher latency.

## Cache Lines

Memory is transferred between cache and DRAM in fixed-size blocks called **cache lines** — almost universally 64 bytes on modern CPUs.

When you access one byte, the entire 64-byte cache line is loaded. This is why sequential access is efficient: you get 15 "free" neighboring elements loaded alongside the one you requested.

**Implication**: Pad data structures to cache line boundaries to avoid false sharing.

## Why False Sharing Happens

MESI protocol: each cache line can be in one of four states — Modified, Exclusive, Shared, Invalid.

When Core 0 writes to a cache line, it marks all other cores' copies as **Invalid** (the "I" in MESI). Before Core 1 can read or write that line, it must fetch the updated version from L3 or the other core's L1.

If two variables are on the same cache line and different cores modify them:
- Core 0 writes `counter0` → invalidates Core 1's copy
- Core 1 writes `counter1` → invalidates Core 0's copy
- This "ping-pong" happens millions of times → catastrophic slowdown

**Fix**: Separate the variables onto different cache lines using `__attribute__((aligned(64)))` or a padding struct.

## Sequential vs Random Access

Sequential access triggers the hardware **prefetcher** — a circuit that detects linear access patterns and preloads cache lines before they are requested. The CPU can issue multiple memory requests in parallel (memory-level parallelism).

Random access:
- Defeats the prefetcher (no pattern)
- Each access must wait for the previous to complete (pointer chase serializes)
- 100% cache miss rate on large arrays → every access goes to DRAM

**Measurement**: Sequential bandwidth ~15-50 GB/s; random ~0.3-1 GB/s. The ratio (~50x) is the "memory wall" that ML hardware designers obsess over.

## Matrix Traversal and Row-Major Order

C stores 2D arrays `A[N][N]` in row-major order: `A[i][0], A[i][1], ..., A[i][N-1], A[i+1][0], ...`

Row-major traversal (`for i: for j: A[i][j]`):
- Inner loop accesses consecutive memory locations
- One cache line miss, then 15 free hits → ~1.6× cache miss ratio

Column-major traversal (`for j: for i: A[i][j]`):
- Inner loop accesses `A[0][j], A[1][j], ...` — each N×sizeof(float) = 8192 bytes apart
- Every access is a cache miss
- 100% miss rate → 10-20× slower

## Connection to ML

**Matrix multiplication C = A × B**: The innermost loop computes `C[i][j] += A[i][k] * B[k][j]`. For B, we're accessing column `j` for varying `k` — column-major if B is row-major. This is why:

1. **Tiling**: Break the computation into tiles that fit in L1/L2. Each tile is reused O(tile_size) times.
2. **Transposing B**: Compute `B^T` once, then access row-major. BLAS uses this trick.
3. **Attention mechanism**: Computing `softmax(Q*K^T) * V` requires handling matrix transposes carefully for cache efficiency.

## NUMA Effects (Multi-Socket Systems)

On multi-socket servers (common in data centers), each CPU socket has local DRAM. Accessing remote NUMA node memory adds another ~2-3× latency penalty on top of DRAM latency.

Modern ML training distributed setups must be NUMA-aware: keep computation local to the data. PyTorch's distributed data parallel (DDP) and NCCL do this automatically.

## Roofline Model Preview

The roofline model says: performance is bounded by either:
- **Compute bound**: FLOPs/s limit (arithmetic intensity is high enough that compute is the bottleneck)
- **Memory bound**: bandwidth limit (too many bytes fetched per FLOP)

For a cache-friendly matrix multiply, we can get close to compute-bound performance. A random access pattern is purely memory-bound and can't be improved without changing the algorithm.

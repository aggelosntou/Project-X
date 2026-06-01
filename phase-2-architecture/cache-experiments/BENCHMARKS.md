# Cache Experiments — Benchmark Results

Expected results on typical hardware (Apple M2 Pro, 10-core).

## 01 — Cache Size Detection

```
Array Size (KB) | Elements     | ns/access
1               | 256          | 0.82
2               | 512          | 0.83
4               | 1024         | 0.84
8               | 2048         | 0.84
16              | 4096         | 0.85     ← L1 cache limit ~48KB
32              | 8192         | 0.87
64              | 16384        | 2.14     ← L2 begins
128             | 32768        | 2.18
256             | 65536        | 2.81
512             | 131072       | 4.12     ← L3 begins ~12MB
1024            | 262144       | 4.45
4096            | 1048576      | 5.11
8192            | 2097152      | 7.83
16384           | 4194304      | 22.4     ← DRAM begins ~20MB L3
32768           | 8388608      | 48.1
65536           | 16777216     | 52.3
```

## 02 — Cache Line Size

```
Stride  | Bytes jumped  | ns/access   | Note
1       | 4             | 0.83
2       | 8             | 0.84
4       | 16            | 0.87
8       | 32            | 1.2
16      | 64            | 48.2        ← plateau begins = cache line (64 bytes)
32      | 128           | 48.5
64      | 256           | 48.7
128     | 512           | 48.9
```

**Detected cache line size: 64 bytes** (standard on all modern x86 and ARM CPUs)

## 03 — False Sharing

```
[1] Adjacent counters (false sharing)...
  Adjacent counters: c0=200000000, c1=200000000
[2] Padded counters (no false sharing)...
  Padded counters:   c0=200000000, c1=200000000

Results:
  Adjacent (false sharing):  0.823 s
  Padded (no false sharing): 0.142 s
  Speedup: 5.8x
```

**5-10x slowdown from false sharing** — common in production multi-threaded code

## 04 — Sequential vs Random Access

```
Sequential access:
  Time:       0.121 s
  Bandwidth:  31.4 GB/s
  ns/element: 0.45

Random access:
  Time:       14.3 s
  Bandwidth:  0.27 GB/s
  ns/element: 53.2
```

**Ratio: ~118x slower for random access**

## 05 — Matrix Traversal (2048×2048 floats, 16 MB)

```
Row-major (cache-friendly):    42.3 ms  |  23.8 GB/s
Column-major (cache-hostile):  643.1 ms |  1.57 GB/s

Slowdown from column-major: 15.2x
```

## Summary Table

| Experiment | Key Metric | Value |
|---|---|---|
| Cache size | L1 boundary | ~32-48 KB |
| Cache size | L2 boundary | ~256 KB - 1 MB |
| Cache size | L3 boundary | ~8-32 MB |
| Cache line | Line size | 64 bytes |
| False sharing | Slowdown | 5-10x |
| Random vs seq | Slowdown | 50-150x |
| Column traversal | Slowdown | 5-20x |

## Implications for ML Code

- **BatchNorm, LayerNorm**: normalize across features (often column-major). Fused CUDA kernels swap loop order to stay cache-friendly.
- **Attention**: `softmax(QK^T)V` — K^T is a column-major access pattern. Flash Attention fixes this with tiling.
- **Embedding tables**: random access into large embedding tables is the defining bottleneck in recommendation systems.
- **Parameter servers**: false sharing in gradient buffers can kill multi-threaded training throughput.

# Matrix Multiply — Optimization Journey

Full progression from naive to near-BLAS performance.

## Implementations

| File | Technique | Expected Speedup |
|---|---|---|
| `naive.c` | Triple loop, ikj order | 1× (baseline) |
| `transposed.c` | Transpose B, then ijk | 1.5–3× |
| `tiled.c` | Cache blocking (T=32) | 3–8× |
| `simd.c` | AVX2 vectorization | 4–8× |
| `tiled_simd.c` | Tiling + AVX2 | 8–20× |

## Building

```bash
make
```

On Apple Silicon (M1/M2), AVX2 is not available. The SIMD versions fall back to scalar but the tiled version still shows cache effects clearly.

## Running Benchmark

```bash
make run
```

Output includes time, GFLOPS, and % of theoretical peak for N=64,128,256,512,1024.

## Understanding the Numbers

For N=1024:
- Total FLOPs: 2 × 1024³ ≈ 2.1 billion
- Naive at 0.5 GFLOPS: ~4.2 s
- Tiled+SIMD at 20 GFLOPS: ~0.1 s

Theoretical peak (Intel, single core, AVX2 FMA):
```
peak = 3 GHz × 8 floats × 2 (FMA) = 48 GFLOPS
```

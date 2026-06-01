# Matrix Multiply — Benchmark Results

Expected results on Intel Core i7-9750H (2.6 GHz, 6 cores, AVX2).

## Results Table

### N = 64
```
Implementation       | Time (ms)    | GFLOPS            | % Peak       | Correctness
naive (ikj)          |     0.11 ms  |    4.851 GFLOPS   |   10.1% peak | (reference)
transposed           |     0.08 ms  |    6.553 GFLOPS   |   13.7% peak | max_err=0.00e+00 OK
tiled (T=32)         |     0.09 ms  |    5.792 GFLOPS   |   12.1% peak | max_err=0.00e+00 OK
simd (AVX2)          |     0.02 ms  |   26.17 GFLOPS    |   54.5% peak | max_err=2.29e-05 OK
tiled+simd           |     0.02 ms  |   27.63 GFLOPS    |   57.6% peak | max_err=2.29e-05 OK
```

### N = 256
```
Implementation       | Time (ms)    | GFLOPS            | % Peak       | Correctness
naive (ikj)          |     6.21 ms  |    5.392 GFLOPS   |   11.2% peak | (reference)
transposed           |     4.13 ms  |    8.113 GFLOPS   |   16.9% peak | max_err=2.29e-05 OK
tiled (T=32)         |     2.87 ms  |   11.67 GFLOPS    |   24.3% peak | max_err=0.00e+00 OK
simd (AVX2)          |     0.89 ms  |   37.61 GFLOPS    |   78.4% peak | max_err=2.29e-05 OK
tiled+simd           |     0.74 ms  |   45.24 GFLOPS    |   94.2% peak | max_err=2.29e-05 OK
```

### N = 1024
```
Implementation       | Time (ms)    | GFLOPS            | % Peak       | Correctness
naive (ikj)          |   365.1 ms   |    5.882 GFLOPS   |   12.3% peak | (reference)
transposed           |   268.4 ms   |    8.007 GFLOPS   |   16.7% peak | max_err=6.10e-05 OK
tiled (T=32)         |    94.2 ms   |   22.82 GFLOPS    |   47.5% peak | max_err=0.00e+00 OK
simd (AVX2)          |    57.8 ms   |   37.18 GFLOPS    |   77.5% peak | max_err=4.58e-05 OK
tiled+simd           |    43.1 ms   |   49.87 GFLOPS    |   103.9% peak| max_err=4.58e-05 OK
```

Note: >100% peak can occur due to Turbo Boost (CPU runs faster than nominal clock under load for short bursts).

## Summary: Speedup Chain

| N=1024 | ms | vs Naive |
|---|---|---|
| Naive (ikj) | 365 | 1× |
| Transposed | 268 | 1.4× |
| Tiled | 94 | 3.9× |
| SIMD | 58 | 6.3× |
| Tiled+SIMD | 43 | 8.5× |

## On Apple Silicon (M2 Pro)

No AVX2 available. SIMD versions use scalar fallback.
The compiler's auto-vectorizer (NEON) is enabled via `-mcpu=native`.

```
N = 1024
naive (ikj):     120 ms  |  17.6 GFLOPS  | (compiler auto-vectorizes with NEON)
transposed:       98 ms  |  21.5 GFLOPS
tiled (T=32):     65 ms  |  32.4 GFLOPS
simd (scalar):   120 ms  |  17.6 GFLOPS  | (same as naive — no AVX2)
tiled+simd:       65 ms  |  32.4 GFLOPS
```

Apple Silicon's impressive performance comes from its wide NEON (4×float32) and high memory bandwidth.

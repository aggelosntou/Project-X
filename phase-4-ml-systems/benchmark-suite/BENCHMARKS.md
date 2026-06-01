# Benchmarks — Benchmark Suite

Results below are representative of an Apple M2 Pro (12 cores, 3.49 GHz).
Run `python3 src/runner.py` to get numbers for your machine.

## System (example)

| Property | Value |
|---|---|
| os | Darwin 23.x |
| python | 3.11.x |
| cpu | Apple M2 Pro |
| cpu_cores_physical | 12 |
| cpu_cores_logical | 12 |
| ram_gb | 16.0 |
| l1_cache_kb | 192 |
| l2_cache_kb | 12288 |
| numpy_version | 1.26.x |

## Matrix Multiply Benchmark

| N | GFLOPS | time_ms |
|---|--------|---------|
| 64 | 48.2 | 0.011 |
| 128 | 142.7 | 0.029 |
| 256 | 251.4 | 0.134 |
| 512 | 285.1 | 0.946 |
| 1024 | 294.3 | 7.30 |
| 2048 | 301.1 | 57.1 |

**Peak**: ~301 GFLOPS (float32 with Accelerate BLAS)

## Attention Benchmark

| seq_len | time_ms | mem_MB | GFLOPS |
|---------|---------|--------|--------|
| 32 | 0.042 | 0.06 | 0.006 |
| 64 | 0.118 | 0.22 | 0.009 |
| 128 | 0.441 | 0.87 | 0.010 |
| 256 | 1.891 | 3.44 | 0.009 |
| 512 | 8.104 | 13.75 | 0.009 |

Attention is heavily memory-bandwidth bound: GFLOPS stays constant
while memory grows as O(N²) — the bottleneck is moving scores
to/from RAM, not arithmetic.

## Training Step Breakdown

batch_size=32, n_steps=100:

| Phase | mean_ms | std_ms | % of total |
|-------|---------|--------|-----------|
| forward | 0.048 | 0.005 | 22% |
| backward | 0.142 | 0.009 | 66% |
| update | 0.028 | 0.003 | 12% |

**Total**: ~0.218 ms/step → ~4,600 steps/sec

Backward is ~3× slower than forward because it performs 2 large GEMMs
(dW = x^T @ d_out, d_x = d_out @ W^T) vs 1 for forward.

## Inference Latency vs Batch Size

| batch_size | latency_ms/sample | throughput/s |
|-----------|-------------------|-------------|
| 1 | 0.00482 | 207,400 |
| 2 | 0.00278 | 359,600 |
| 4 | 0.00185 | 540,200 |
| 8 | 0.00133 | 752,700 |
| 16 | 0.00112 | 892,400 |
| 32 | 0.00098 | 1,021,100 |
| 64 | 0.00089 | 1,122,500 |
| 128 | 0.00082 | 1,219,600 |

Latency/sample drops 6× from batch=1 to batch=128 as BLAS amortises overhead.
Throughput plateaus beyond batch=64 (compute-saturated).

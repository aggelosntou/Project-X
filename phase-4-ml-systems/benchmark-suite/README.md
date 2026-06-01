# Benchmark Suite — MLPerf-Style CPU Benchmarks

A self-contained, pure-NumPy benchmark suite that measures:
- Matrix multiply throughput (GFLOPS)
- Attention kernel performance vs sequence length
- Training step breakdown (forward / backward / update)
- Inference latency vs batch size

No GPU required.  All benchmarks run on CPU with NumPy.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run all benchmarks
python3 src/runner.py

# 3. Generate Markdown report from saved results
python3 src/report_generator.py
```

## File Layout

```
src/
├── system_info.py                  # CPU, RAM, cache size detection
├── runner.py                       # Runs all benchmarks, saves JSON
├── report_generator.py             # Converts JSON -> Markdown report
└── benchmarks/
    ├── __init__.py
    ├── matmul_benchmark.py         # GEMM throughput
    ├── attention_benchmark.py      # Scaled dot-product attention
    ├── training_benchmark.py       # Full training step (fwd+bwd+update)
    └── inference_benchmark.py      # Latency vs batch size
```

## Output Files

| File | Description |
|------|-------------|
| `benchmark_results.json` | Raw results in JSON, timestamped |
| `benchmark_report.md`    | Human-readable Markdown report |

## Statistical Methodology

- **Warmup runs** are discarded — cold cache effects distort latency.
- **Minimum of N runs** is used for deterministic ops (matmul, forward
  pass) — the minimum reflects the true hardware speed; higher values
  come from OS jitter, cache misses, and thermal throttling.
- **Mean ± std** is reported for training steps — they are
  probabilistic (depend on batch content) and averaging is appropriate.

## Interpreting Results

### Peak GFLOPS vs Theoretical

```
Theoretical peak = cores × freq (GHz) × SIMD_width × FMA_factor
                 = 8 × 3.5 × 8 × 2 = 448 GFLOPS  (example: 8-core 3.5 GHz with AVX2)
```

If measured < 30% of theoretical:
- NumPy may not be using an optimised BLAS (check `numpy.show_config()`).
- Install `openblas` or `mkl` for 5–10× improvement.

### Attention Arithmetic Intensity

```
FLOPs per element of Q @ K^T:  2 * d_head
Bytes read per element:         2 * d_head * 4  (one Q row + one K row)
Arithmetic intensity:           0.25 FLOP/byte
```

This is far below the compute-bound threshold (~100 FLOP/byte on modern
CPUs), confirming attention is **memory-bandwidth bound**.

### Inference Latency Floor

At batch_size=1 the latency is dominated by the Python function-call
overhead of NumPy (~2–10 µs per call).  This is why compiled inference
(ONNX Runtime, TFLite, C) is essential for real-time applications.

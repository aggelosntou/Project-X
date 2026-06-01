"""
inference_benchmark.py
======================
Measure batch inference latency vs throughput for a 3-layer MLP.

For each batch_size in [1, 2, 4, 8, 16, 32, 64, 128]:
  - Total samples = max(batch_size * 100, 1000) to keep timing stable
  - Measure wall-clock time
  - Report:
      latency_ms   = total_time / total_samples × 1000
      throughput   = total_samples / total_time
      lat_per_sample (same as latency_ms — redundant but explicit)

This illustrates the key hardware insight:
  - At batch_size=1: latency is dominated by Python call overhead
  - At batch_size=128: latency/sample drops as BLAS amortises fixed costs
"""

import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


# ---------------------------------------------------------------------------
# Tiny MLP — same architecture as training_benchmark for consistency
# ---------------------------------------------------------------------------

class InferenceMLP:
    """MLP 128 -> 256 -> 128 -> 10, weights fixed at construction."""

    def __init__(self):
        rng = np.random.default_rng(99)
        layer_sizes = [128, 256, 128, 10]
        self.weights = []
        self.biases  = []
        for i in range(len(layer_sizes) - 1):
            fi, fo = layer_sizes[i], layer_sizes[i + 1]
            limit = np.sqrt(6.0 / (fi + fo))
            W = rng.uniform(-limit, limit, (fi, fo)).astype(np.float32)
            b = np.zeros(fo, dtype=np.float32)
            self.weights.append(W)
            self.biases.append(b)

    def forward(self, x: np.ndarray) -> np.ndarray:
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            x = x @ W + b
            if i < len(self.weights) - 1:
                x = np.maximum(0.0, x)  # ReLU
        return x  # logits

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.argmax(self.forward(x), axis=-1)


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def benchmark_inference(
    total_target: int = 1000,
    n_runs: int       = 5,
    warmup: int       = 3,
) -> dict:
    """
    Benchmark inference at various batch sizes.

    Returns dict with 'status' and 'rows'.
    """
    print("\n=== Inference Benchmark ===")
    header = (f"{'batch_size':>12}  {'latency_ms':>12}  "
              f"{'throughput/s':>14}  {'lat/sample_ms':>16}")
    print(header)
    print("-" * len(header))

    model      = InferenceMLP()
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
    rows       = []
    rng        = np.random.default_rng(13)

    for bs in batch_sizes:
        n_batches = max(total_target // bs, 10)
        total_samples = n_batches * bs

        x = rng.standard_normal((bs, 128)).astype(np.float32)

        # Warmup
        for _ in range(warmup):
            model.predict(x)

        # Timed runs: take minimum wall-clock over n_runs repetitions
        best_elapsed = float('inf')
        for _ in range(n_runs):
            t0 = time.perf_counter()
            for _ in range(n_batches):
                model.predict(x)
            elapsed = time.perf_counter() - t0
            if elapsed < best_elapsed:
                best_elapsed = elapsed

        latency_ms      = best_elapsed / total_samples * 1000   # ms per sample
        throughput      = total_samples / best_elapsed           # samples/sec

        row = {
            'batch_size':      bs,
            'latency_ms':      round(latency_ms,  5),
            'throughput_s':    round(throughput,  1),
            'lat_per_sample':  round(latency_ms,  5),
        }
        rows.append(row)
        print(f"{bs:>12}  {latency_ms:>12.5f}  "
              f"{throughput:>14.1f}  {latency_ms:>16.5f}")

    return {'status': 'completed', 'rows': rows}


if __name__ == '__main__':
    benchmark_inference()

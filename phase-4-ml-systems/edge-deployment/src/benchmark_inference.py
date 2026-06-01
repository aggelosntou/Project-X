"""
benchmark_inference.py
======================
Compare three inference strategies for the MLP:

  1. Python + NumPy (float32) — baseline
  2. NumPy float16            — halved memory footprint, possible SIMD speedup
  3. C binary                 — compiled c_inference binary (if available)

For batch_size in [1, 8, 32, 128]:
  - Run 1000 / batch_size forward passes (so total samples = 1000)
  - Report latency (ms/sample) and throughput (samples/sec)

Run:
    python3 src/benchmark_inference.py
"""

import os
import sys
import subprocess
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model_definition import MLP
from export_weights import export_to_binary

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TOTAL_SAMPLES = 1000
INPUT_DIM     = 784


def _make_model_fp32():
    model = MLP()
    return model


def _make_model_fp16(model_fp32):
    """Return a wrapper whose weights are cast to float16."""
    class FP16Wrapper:
        def __init__(self, m):
            self.weights = [W.astype(np.float16) for W in m.weights]
            self.biases  = [b.astype(np.float16) for b in m.biases]

        def forward(self, x):
            x = x.astype(np.float16)
            for i, (W, b) in enumerate(zip(self.weights, self.biases)):
                x = x @ W + b
                if i < len(self.weights) - 1:
                    x = np.maximum(0, x)
            return x.astype(np.float32)

    return FP16Wrapper(model_fp32)


def _time_numpy_inference(model, batch_size: int, total_samples: int,
                           n_warmup: int = 3) -> float:
    """Return total wall-clock seconds for total_samples inferences."""
    n_batches = total_samples // batch_size
    x = np.random.randn(batch_size, INPUT_DIM).astype(np.float32)

    # Warmup
    for _ in range(n_warmup):
        model.forward(x)

    t0 = time.perf_counter()
    for _ in range(n_batches):
        model.forward(x)
    return time.perf_counter() - t0


# ---------------------------------------------------------------------------
# C binary timing
# ---------------------------------------------------------------------------

def _run_c_binary(binary_path: str) -> float | None:
    """
    Run the pre-compiled C binary for 1000 single-sample inferences.
    Parse the 'Latency: X.XXXX ms/inference' line from its stdout.
    Returns latency in ms, or None if binary not available.
    """
    if not os.path.isfile(binary_path):
        return None
    try:
        result = subprocess.run(
            [binary_path, 'model.bin'],
            capture_output=True,
            text=True,
            timeout=30,
        )
        for line in result.stdout.splitlines():
            if 'Latency' in line and 'ms/inference' in line:
                # e.g. "Latency      : 0.0123 ms/inference"
                val = float(line.split(':')[1].strip().split()[0])
                return val  # ms per sample
    except Exception as e:
        print(f"  [C binary error: {e}]")
    return None


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def benchmark_inference():
    print("\n=== Inference Benchmark ===")
    print("Building models...")
    fp32_model = _make_model_fp32()
    fp16_model = _make_model_fp16(fp32_model)
    print(f"  {fp32_model}")

    # Export model.bin for C binary (best-effort)
    try:
        export_to_binary(fp32_model, 'model.bin')
    except Exception as e:
        print(f"  [export skipped: {e}]")

    c_binary = './c_inference'

    batch_sizes = [1, 8, 32, 128]

    header = (
        f"\n{'batch':>8} "
        f"{'fp32_lat':>12} {'fp32_tput':>14} "
        f"{'fp16_lat':>12} {'fp16_tput':>14} "
        f"{'fp16/fp32':>10}"
    )
    print(header)
    print("-" * len(header))

    rows = []
    for bs in batch_sizes:
        t_fp32 = _time_numpy_inference(fp32_model, bs, TOTAL_SAMPLES)
        t_fp16 = _time_numpy_inference(fp16_model, bs, TOTAL_SAMPLES)

        lat_fp32  = t_fp32 / TOTAL_SAMPLES * 1000  # ms per sample
        tput_fp32 = TOTAL_SAMPLES / t_fp32

        lat_fp16  = t_fp16 / TOTAL_SAMPLES * 1000
        tput_fp16 = TOTAL_SAMPLES / t_fp16

        speedup = t_fp32 / t_fp16

        rows.append({
            'batch_size': bs,
            'fp32_latency_ms': round(lat_fp32,  4),
            'fp32_throughput': round(tput_fp32, 1),
            'fp16_latency_ms': round(lat_fp16,  4),
            'fp16_throughput': round(tput_fp16, 1),
            'fp16_speedup':    round(speedup,   3),
        })

        print(f"{bs:>8} "
              f"{lat_fp32:>12.4f} {tput_fp32:>14.1f} "
              f"{lat_fp16:>12.4f} {tput_fp16:>14.1f} "
              f"{speedup:>10.3f}x")

    # --- C binary (batch_size=1 only) ---
    c_lat = _run_c_binary(c_binary)
    if c_lat is not None:
        c_tput = 1000.0 / c_lat
        # Compare against fp32 at batch_size=1
        fp32_lat_bs1 = rows[0]['fp32_latency_ms']
        speedup_c = fp32_lat_bs1 / c_lat
        print(f"\nC binary (batch=1):  latency={c_lat:.4f} ms  "
              f"throughput={c_tput:.1f}/s  vs_fp32={speedup_c:.2f}x")
    else:
        print(f"\nC binary not found at '{c_binary}'.")
        print("  Build it with:  make  (or gcc -O3 -o c_inference src/c_inference.c -lm)")

    return {
        'batch_results': rows,
        'c_latency_ms': c_lat,
    }


if __name__ == '__main__':
    results = benchmark_inference()

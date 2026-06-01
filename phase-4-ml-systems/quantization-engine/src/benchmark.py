"""
benchmark.py — Compare FP32 vs INT8 vs FP16 on a small MLP.

Measures:
  - Accuracy (MSE proxy on regression task)
  - Speed (time for matrix multiply)
  - Memory (model size in MB)
  - Quantization granularity sweep (per-tensor vs per-channel)
"""

import sys
import time
import numpy as np

# Allow imports from same src/ dir when run directly
sys.path.insert(0, __file__.replace("benchmark.py", ""))

from quantize import (
    quantize_tensor_int8,
    dequantize_int8,
    quantize_tensor_fp16,
    per_channel_quantize,
    dequantize_per_channel,
    quantization_snr,
)
from quantized_linear import QuantizedLinear


# ---------------------------------------------------------------------------
# Tiny MLP for benchmarking
# ---------------------------------------------------------------------------

class FP32MLP:
    """Three-layer MLP: input → hidden1 → hidden2 → output."""

    def __init__(self, layer_sizes, rng):
        self.weights = []
        self.biases  = []
        for i in range(len(layer_sizes) - 1):
            n_in, n_out = layer_sizes[i], layer_sizes[i + 1]
            W = rng.standard_normal((n_out, n_in)).astype(np.float32) * np.sqrt(2 / n_in)
            b = np.zeros(n_out, dtype=np.float32)
            self.weights.append(W)
            self.biases.append(b)

    def forward(self, x):
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            x = x @ W.T + b
            if i < len(self.weights) - 1:   # ReLU on hidden layers
                x = np.maximum(x, 0)
        return x

    def memory_bytes(self):
        return sum(W.nbytes + b.nbytes for W, b in zip(self.weights, self.biases))


def build_quantized_mlp(fp32_model, calibration_data, per_channel=True):
    """Convert an FP32MLP to a list of QuantizedLinear layers."""
    layers = []
    x = calibration_data.copy()
    for i, (W, b) in enumerate(zip(fp32_model.weights, fp32_model.biases)):
        ql = QuantizedLinear.from_fp32(W, b, per_channel=per_channel)
        ql.calibrate(x)
        ql.finalize_calibration()
        layers.append(ql)
        x = np.maximum(x @ W.T + b, 0) if i < len(fp32_model.weights) - 1 else x @ W.T + b
    return layers


def quantized_forward(layers, x):
    for i, layer in enumerate(layers):
        x = layer(x)
        if i < len(layers) - 1:
            x = np.maximum(x, 0)
    return x


def fp16_forward(fp32_model, x):
    """Run FP32 model with weights and activations cast to FP16."""
    x = x.astype(np.float16)
    for i, (W, b) in enumerate(zip(fp32_model.weights, fp32_model.biases)):
        x = x @ W.astype(np.float16).T + b.astype(np.float16)
        if i < len(fp32_model.weights) - 1:
            x = np.maximum(x, np.float16(0))
    return x.astype(np.float32)


# ---------------------------------------------------------------------------
# Speed benchmark: raw matrix multiply
# ---------------------------------------------------------------------------

def benchmark_matmul(N=512, M=512, K=512, n_repeats=50):
    """Time FP32 / FP16 / INT8-simulated matrix multiply."""
    rng = np.random.default_rng(0)
    A_fp32 = rng.standard_normal((N, K)).astype(np.float32)
    B_fp32 = rng.standard_normal((K, M)).astype(np.float32)

    # FP32
    times_fp32 = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        _ = A_fp32 @ B_fp32
        times_fp32.append(time.perf_counter() - t0)

    # FP16
    A_fp16 = A_fp32.astype(np.float16)
    B_fp16 = B_fp32.astype(np.float16)
    times_fp16 = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        _ = A_fp16 @ B_fp16
        times_fp16.append(time.perf_counter() - t0)

    # INT8 simulation: quantize, dequantize, then multiply in FP32
    # (Real INT8 GEMM would be faster on hardware with INT8 tensor cores)
    A_q, s_A, zp_A = quantize_tensor_int8(A_fp32, symmetric=True)
    B_q, s_B, zp_B = quantize_tensor_int8(B_fp32, symmetric=True)
    times_int8 = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        A_deq = dequantize_int8(A_q, s_A, zp_A)
        B_deq = dequantize_int8(B_q, s_B, zp_B)
        _ = A_deq @ B_deq
        times_int8.append(time.perf_counter() - t0)

    def stats(ts):
        ts = sorted(ts)
        return np.mean(ts) * 1000, np.min(ts) * 1000

    fp32_mean, fp32_min = stats(times_fp32)
    fp16_mean, fp16_min = stats(times_fp16)
    int8_mean, int8_min = stats(times_int8)

    return {
        "fp32_mean_ms": fp32_mean,
        "fp16_mean_ms": fp16_mean,
        "int8_sim_mean_ms": int8_mean,
        "fp16_speedup_vs_fp32": fp32_mean / fp16_mean,
    }


# ---------------------------------------------------------------------------
# Memory benchmark
# ---------------------------------------------------------------------------

def model_memory_report(fp32_model):
    total_params = sum(W.size + b.size for W, b in
                       zip(fp32_model.weights, fp32_model.biases))
    return {
        "n_params"    : total_params,
        "fp32_MB"     : total_params * 4 / 1e6,
        "fp16_MB"     : total_params * 2 / 1e6,
        "int8_MB"     : total_params * 1 / 1e6,
        "fp32_to_int8_ratio": 4.0,
    }


# ---------------------------------------------------------------------------
# Accuracy benchmark
# ---------------------------------------------------------------------------

def accuracy_benchmark(fp32_model, q_layers_pc, q_layers_pt, fp16_fn,
                        x_test, y_test):
    """Measure MSE for each precision on a regression-style task."""
    y_fp32  = fp32_model.forward(x_test)
    y_int8_pc = quantized_forward(q_layers_pc, x_test)
    y_int8_pt = quantized_forward(q_layers_pt, x_test)
    y_fp16  = fp16_fn(x_test)

    def mse(a, b):
        return float(np.mean((a - b) ** 2))

    return {
        "fp32_vs_fp32_mse"     : 0.0,
        "int8_pc_vs_fp32_mse"  : mse(y_fp32, y_int8_pc),
        "int8_pt_vs_fp32_mse"  : mse(y_fp32, y_int8_pt),
        "fp16_vs_fp32_mse"     : mse(y_fp32, y_fp16),
        "int8_pc_snr_db"       : quantization_snr(y_fp32, y_int8_pc),
        "int8_pt_snr_db"       : quantization_snr(y_fp32, y_int8_pt),
        "fp16_snr_db"          : quantization_snr(y_fp32, y_fp16),
    }


# ---------------------------------------------------------------------------
# Granularity sweep
# ---------------------------------------------------------------------------

def granularity_sweep():
    """Compare per-tensor vs per-channel SNR across different weight distributions."""
    rng = np.random.default_rng(7)
    results = []

    for scenario, W in [
        ("Uniform scale",     rng.standard_normal((64, 128)).astype(np.float32)),
        ("Mixed scale (10x)", _make_mixed_scale_weights(rng, 64, 128, 10.0)),
        ("Mixed scale (100x)",_make_mixed_scale_weights(rng, 64, 128, 100.0)),
    ]:
        # Per-tensor
        q_pt, s, zp = quantize_tensor_int8(W.flatten())
        W_pt = dequantize_int8(q_pt, s, zp).reshape(W.shape)
        snr_pt = quantization_snr(W, W_pt)

        # Per-channel
        q_pc, scales, zps = per_channel_quantize(W)
        W_pc = dequantize_per_channel(q_pc, scales, zps)
        snr_pc = quantization_snr(W, W_pc)

        results.append({
            "scenario"         : scenario,
            "per_tensor_snr_db": snr_pt,
            "per_channel_snr_db": snr_pc,
            "improvement_db"   : snr_pc - snr_pt,
        })
    return results


def _make_mixed_scale_weights(rng, rows, cols, scale_ratio):
    W = rng.standard_normal((rows, cols)).astype(np.float32)
    half = rows // 2
    W[:half] *= scale_ratio   # large channels
    return W


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    layer_sizes = [256, 512, 256, 64]
    model = FP32MLP(layer_sizes, rng)

    print("=" * 65)
    print("PHASE 4 — Quantization Engine Benchmark")
    print("=" * 65)

    # Generate calibration / test data
    n_calib = 512
    n_test  = 256
    x_calib = rng.standard_normal((n_calib, layer_sizes[0])).astype(np.float32)
    x_test  = rng.standard_normal((n_test,  layer_sizes[0])).astype(np.float32)
    y_test  = model.forward(x_test)   # "ground truth" = FP32 output

    # Build quantized models
    q_layers_pc = build_quantized_mlp(model, x_calib, per_channel=True)
    q_layers_pt = build_quantized_mlp(model, x_calib, per_channel=False)

    # ---------- Memory ----------
    mem = model_memory_report(model)
    print(f"\nMEMORY REPORT  ({mem['n_params']:,} parameters)")
    print(f"  FP32 : {mem['fp32_MB']:.3f} MB")
    print(f"  FP16 : {mem['fp16_MB']:.3f} MB  ({2}x smaller)")
    print(f"  INT8 : {mem['int8_MB']:.3f} MB  ({4}x smaller)")

    # ---------- Accuracy ----------
    acc = accuracy_benchmark(model, q_layers_pc, q_layers_pt,
                             lambda x: fp16_forward(model, x),
                             x_test, y_test)
    print(f"\nACCURACY (MSE vs FP32 reference):")
    print(f"  FP32            MSE={acc['fp32_vs_fp32_mse']:.2e}   SNR=∞")
    print(f"  FP16            MSE={acc['fp16_vs_fp32_mse']:.2e}   SNR={acc['fp16_snr_db']:.1f} dB")
    print(f"  INT8 per-tensor MSE={acc['int8_pt_vs_fp32_mse']:.2e}   SNR={acc['int8_pt_snr_db']:.1f} dB")
    print(f"  INT8 per-chan   MSE={acc['int8_pc_vs_fp32_mse']:.2e}   SNR={acc['int8_pc_snr_db']:.1f} dB")

    # ---------- Speed ----------
    speed = benchmark_matmul(N=512, M=512, K=512, n_repeats=30)
    print(f"\nSPEED  (512×512 matrix multiply, 30 runs):")
    print(f"  FP32          : {speed['fp32_mean_ms']:.3f} ms")
    print(f"  FP16          : {speed['fp16_mean_ms']:.3f} ms  "
          f"({speed['fp16_speedup_vs_fp32']:.2f}x vs FP32 on this CPU)")
    print(f"  INT8 (simul.) : {speed['int8_sim_mean_ms']:.3f} ms  "
          f"(simulation includes quantize+dequantize overhead)")
    print(f"  Note: Real INT8 GEMM on hardware with INT8 tensor cores")
    print(f"        would be ~4x faster than FP32.")

    # ---------- Granularity sweep ----------
    print(f"\nGRANULARITY SWEEP (per-tensor vs per-channel SNR):")
    print(f"  {'Scenario':<22}  {'Per-tensor':>12}  {'Per-channel':>12}  {'Delta':>8}")
    print(f"  {'-'*22}  {'-'*12}  {'-'*12}  {'-'*8}")
    for r in granularity_sweep():
        print(f"  {r['scenario']:<22}  {r['per_tensor_snr_db']:>10.1f}dB  "
              f"{r['per_channel_snr_db']:>10.1f}dB  "
              f"{r['improvement_db']:>+6.1f}dB")

    print("\nConclusion: per-channel quantization is most beneficial when")
    print("weight channels have very different magnitudes (mixed scale scenario).")

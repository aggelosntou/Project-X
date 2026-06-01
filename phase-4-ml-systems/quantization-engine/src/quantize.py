"""
quantize.py — Tensor quantization primitives: INT8 (symmetric + asymmetric),
FP16, and per-channel quantization.

All functions operate on numpy arrays.  No PyTorch required.
"""

import numpy as np


# ---------------------------------------------------------------------------
# INT8 quantization
# ---------------------------------------------------------------------------

def quantize_tensor_int8(tensor: np.ndarray, symmetric: bool = True):
    """
    Quantize a float32 tensor to INT8.

    Symmetric:
        scale      = max(|x|) / 127
        zero_point = 0
        quantized  = round(x / scale).clip(-128, 127)

    Asymmetric:
        scale      = (max - min) / 255
        zero_point = round(-min / scale)   [clamped to [0, 255]]
        quantized  = round(x / scale + zero_point).clip(0, 255)  [uint8]

    Returns
    -------
    quantized   : np.ndarray  (int8 or uint8)
    scale       : float
    zero_point  : int
    """
    tensor = np.asarray(tensor, dtype=np.float32)

    if symmetric:
        max_abs = np.max(np.abs(tensor))
        if max_abs == 0:
            scale = 1.0
        else:
            scale = float(max_abs) / 127.0
        zero_point = 0
        q = np.round(tensor / scale).clip(-128, 127).astype(np.int8)
    else:
        t_min = float(np.min(tensor))
        t_max = float(np.max(tensor))
        if t_max == t_min:
            scale = 1.0
            zero_point = 0
        else:
            scale = (t_max - t_min) / 255.0
            zero_point = int(np.round(-t_min / scale))
            zero_point = int(np.clip(zero_point, 0, 255))
        q = np.round(tensor / scale + zero_point).clip(0, 255).astype(np.uint8)

    return q, scale, zero_point


def dequantize_int8(quantized: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    """
    Reconstruct float32 from quantized INT8/UINT8.

        x_approx = (q - zero_point) * scale
    """
    return (quantized.astype(np.float32) - zero_point) * scale


# ---------------------------------------------------------------------------
# FP16 quantization
# ---------------------------------------------------------------------------

def quantize_tensor_fp16(tensor: np.ndarray):
    """
    Cast tensor to float16 and measure precision loss.

    Returns
    -------
    fp16_tensor     : np.ndarray (float16)
    max_abs_error   : float
    mean_abs_error  : float
    relative_error  : float   (mean |err| / mean |x|)
    """
    tensor = np.asarray(tensor, dtype=np.float32)
    fp16 = tensor.astype(np.float16)
    diff = np.abs(tensor - fp16.astype(np.float32))

    max_abs = float(np.max(np.abs(tensor)))
    max_abs_error  = float(np.max(diff))
    mean_abs_error = float(np.mean(diff))
    relative_error = mean_abs_error / (max_abs + 1e-12)

    return fp16, max_abs_error, mean_abs_error, relative_error


# ---------------------------------------------------------------------------
# Per-channel quantization (for weight matrices)
# ---------------------------------------------------------------------------

def per_channel_quantize(weight_matrix: np.ndarray, symmetric: bool = True):
    """
    Quantize each output channel (row) of a weight matrix independently.

    This gives better accuracy than per-tensor quantization because each
    channel has its own scale — channels with large weights don't force
    small-weight channels to lose precision.

    Parameters
    ----------
    weight_matrix : shape (out_features, in_features)
    symmetric     : use symmetric INT8 (scale per row, zero_point = 0)

    Returns
    -------
    q_weights   : np.ndarray int8, same shape as weight_matrix
    scales      : np.ndarray float32, shape (out_features,)
    zero_points : np.ndarray int32,   shape (out_features,)
    """
    weight_matrix = np.asarray(weight_matrix, dtype=np.float32)
    out_features, in_features = weight_matrix.shape

    q_weights   = np.zeros_like(weight_matrix, dtype=np.int8)
    scales      = np.zeros(out_features, dtype=np.float32)
    zero_points = np.zeros(out_features, dtype=np.int32)

    for i in range(out_features):
        row = weight_matrix[i]
        q_row, s, zp = quantize_tensor_int8(row, symmetric=symmetric)
        q_weights[i] = q_row
        scales[i]      = s
        zero_points[i] = zp

    return q_weights, scales, zero_points


def dequantize_per_channel(q_weights: np.ndarray,
                           scales: np.ndarray,
                           zero_points: np.ndarray) -> np.ndarray:
    """Reconstruct float32 weight matrix from per-channel quantized weights."""
    out = np.zeros(q_weights.shape, dtype=np.float32)
    for i in range(q_weights.shape[0]):
        out[i] = (q_weights[i].astype(np.float32) - zero_points[i]) * scales[i]
    return out


# ---------------------------------------------------------------------------
# Quantization error metrics
# ---------------------------------------------------------------------------

def quantization_snr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """
    Signal-to-Noise Ratio (dB) for quantization.
    Higher is better.  > 40 dB is typically acceptable.
    """
    signal_power = float(np.mean(original ** 2))
    noise_power  = float(np.mean((original - reconstructed) ** 2))
    if noise_power == 0:
        return float('inf')
    return 10.0 * np.log10(signal_power / (noise_power + 1e-30))


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    x = rng.standard_normal(1000).astype(np.float32) * 3.0  # typical weight range

    print("=" * 60)
    print("INT8 SYMMETRIC QUANTIZATION")
    q, s, zp = quantize_tensor_int8(x, symmetric=True)
    x_hat = dequantize_int8(q, s, zp)
    print(f"  scale={s:.6f}  zero_point={zp}")
    print(f"  max |error| = {np.max(np.abs(x - x_hat)):.6f}")
    print(f"  SNR         = {quantization_snr(x, x_hat):.2f} dB")

    print()
    print("INT8 ASYMMETRIC QUANTIZATION")
    x_asym = rng.uniform(0, 6, 1000).astype(np.float32)  # ReLU output
    q, s, zp = quantize_tensor_int8(x_asym, symmetric=False)
    x_hat = dequantize_int8(q, s, zp)
    print(f"  scale={s:.6f}  zero_point={zp}")
    print(f"  max |error| = {np.max(np.abs(x_asym - x_hat)):.6f}")
    print(f"  SNR         = {quantization_snr(x_asym, x_hat):.2f} dB")

    print()
    print("FP16 QUANTIZATION")
    fp16, max_err, mean_err, rel_err = quantize_tensor_fp16(x)
    print(f"  max |error|    = {max_err:.2e}")
    print(f"  mean |error|   = {mean_err:.2e}")
    print(f"  relative error = {rel_err:.2e}")

    print()
    print("PER-CHANNEL QUANTIZATION (vs per-tensor)")
    W = rng.standard_normal((64, 128)).astype(np.float32)
    # Simulate different channel magnitudes (common in real networks)
    W[0] *= 10.0   # one large channel
    W[1] *= 0.01   # one tiny channel

    # Per-tensor
    q_pt, s_pt, zp_pt = quantize_tensor_int8(W.flatten(), symmetric=True)
    W_hat_pt = dequantize_int8(q_pt, s_pt, zp_pt).reshape(W.shape)
    snr_pt = quantization_snr(W, W_hat_pt)

    # Per-channel
    q_pc, scales_pc, zps_pc = per_channel_quantize(W, symmetric=True)
    W_hat_pc = dequantize_per_channel(q_pc, scales_pc, zps_pc)
    snr_pc = quantization_snr(W, W_hat_pc)

    print(f"  Per-tensor  SNR = {snr_pt:.2f} dB")
    print(f"  Per-channel SNR = {snr_pc:.2f} dB  (should be higher)")

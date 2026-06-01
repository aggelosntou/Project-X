"""
quantized_linear.py — INT8-quantized linear layer.

The weights are stored as INT8.  During forward:
  1. Dequantize weights back to float32
  2. Quantize the input activations using calibrated statistics
  3. Multiply (simulating INT8 arithmetic precision)
  4. Dequantize the output

This matches the PTQ (post-training quantization) pattern used in production
frameworks like TFLite and ONNX Runtime.
"""

import numpy as np
from quantize import (
    quantize_tensor_int8,
    dequantize_int8,
    per_channel_quantize,
    dequantize_per_channel,
    quantization_snr,
)


class QuantizedLinear:
    """
    Linear layer y = xW^T + b stored with INT8 weights.

    Parameters
    ----------
    in_features   : int
    out_features  : int
    per_channel   : bool   — use per-channel weight quantization (more accurate)
    symmetric     : bool   — symmetric INT8 (zero_point=0) for weights
    """

    def __init__(self, in_features: int, out_features: int,
                 per_channel: bool = True, symmetric: bool = True):
        self.in_features  = in_features
        self.out_features = out_features
        self.per_channel  = per_channel
        self.symmetric    = symmetric

        # Will be filled by from_fp32() or set directly
        self.q_weight    = None   # int8 quantized weights
        self.weight_scales     = None
        self.weight_zero_points = None
        self.bias        = None   # kept as float32 (bias quantization rarely needed)

        # Activation quantization parameters (filled by calibrate())
        self.act_scale      = None
        self.act_zero_point = None
        self.act_min        = None
        self.act_max        = None
        self.act_percentile_min = None
        self.act_percentile_max = None
        self._calibration_data  = []   # accumulate during calibration

    # ------------------------------------------------------------------
    # Construction from float weights
    # ------------------------------------------------------------------

    @classmethod
    def from_fp32(cls, weight: np.ndarray, bias=None,
                  per_channel: bool = True, symmetric: bool = True):
        """
        Create a QuantizedLinear from a float32 weight matrix.

        weight : shape (out_features, in_features)
        bias   : shape (out_features,) or None
        """
        layer = cls(weight.shape[1], weight.shape[0],
                    per_channel=per_channel, symmetric=symmetric)

        if per_channel:
            q_w, scales, zps = per_channel_quantize(weight, symmetric=symmetric)
            layer.q_weight          = q_w
            layer.weight_scales     = scales
            layer.weight_zero_points = zps
        else:
            q_w, s, zp = quantize_tensor_int8(weight.flatten(), symmetric=symmetric)
            layer.q_weight           = q_w.reshape(weight.shape)
            layer.weight_scales      = np.array([s], dtype=np.float32)
            layer.weight_zero_points = np.array([zp], dtype=np.int32)

        layer.bias = np.array(bias, dtype=np.float32) if bias is not None else None
        return layer

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate(self, data: np.ndarray):
        """
        Accumulate activation statistics from calibration data.

        data : shape (N, in_features) — a representative sample of inputs.
        Typical practice: 100–1000 batches from the training distribution.
        """
        self._calibration_data.append(data.astype(np.float32))

    def finalize_calibration(self, method: str = "minmax"):
        """
        Compute activation quantization parameters from accumulated data.

        method : 'minmax'      — use min/max (simple, common)
                 'percentile'  — use 0.1th/99.9th percentile (more robust)
        """
        if not self._calibration_data:
            raise RuntimeError("No calibration data. Call calibrate() first.")

        all_data = np.concatenate(self._calibration_data, axis=0)

        self.act_min = float(np.min(all_data))
        self.act_max = float(np.max(all_data))
        self.act_percentile_min = float(np.percentile(all_data, 0.1))
        self.act_percentile_max = float(np.percentile(all_data, 99.9))

        if method == "percentile":
            lo, hi = self.act_percentile_min, self.act_percentile_max
        else:
            lo, hi = self.act_min, self.act_max

        # Compute asymmetric scale and zero_point for activations
        # (activations after ReLU are non-negative → asymmetric is better)
        if hi == lo:
            self.act_scale      = 1.0
            self.act_zero_point = 0
        else:
            self.act_scale      = (hi - lo) / 255.0
            self.act_zero_point = int(np.clip(np.round(-lo / self.act_scale), 0, 255))

        self._calibration_data = []  # free memory

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Quantized linear forward pass.

        x : shape (..., in_features)
        Returns y : shape (..., out_features)

        Simulates INT8 arithmetic: all intermediate values are quantized
        so that we accurately model the precision loss that INT8 hardware
        would incur.
        """
        x = np.asarray(x, dtype=np.float32)

        # 1. Dequantize weights
        if self.per_channel:
            W = dequantize_per_channel(self.q_weight,
                                       self.weight_scales,
                                       self.weight_zero_points)
        else:
            W = dequantize_int8(self.q_weight,
                                float(self.weight_scales[0]),
                                int(self.weight_zero_points[0]))

        # 2. Quantize activations (if calibrated), then immediately dequantize
        #    This simulates the precision loss of INT8 activation quantization.
        if self.act_scale is not None:
            x_q = np.round(x / self.act_scale + self.act_zero_point)
            x_q = np.clip(x_q, 0, 255).astype(np.uint8)
            x   = (x_q.astype(np.float32) - self.act_zero_point) * self.act_scale

        # 3. Matrix multiply (simulates INT8 GEMM — real hardware accumulates
        #    in INT32 accumulators then re-quantizes output)
        y = x @ W.T

        # 4. Add bias (float32 bias is added after dequantization — standard practice)
        if self.bias is not None:
            y = y + self.bias

        return y

    def __call__(self, x):
        return self.forward(x)

    # ------------------------------------------------------------------
    # Memory / stats
    # ------------------------------------------------------------------

    def weight_memory_bytes(self) -> dict:
        """Report memory footprint for weights in various precisions."""
        n = self.in_features * self.out_features
        return {
            "fp32_bytes" : n * 4,
            "fp16_bytes" : n * 2,
            "int8_bytes" : n * 1,
            "compression_vs_fp32": 4.0,
        }

    def __repr__(self):
        return (f"QuantizedLinear({self.in_features} → {self.out_features}, "
                f"per_channel={self.per_channel}, "
                f"calibrated={self.act_scale is not None})")


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(0)

    in_f, out_f = 128, 64
    W_fp32 = rng.standard_normal((out_f, in_f)).astype(np.float32) * 0.1
    b_fp32 = rng.standard_normal(out_f).astype(np.float32) * 0.01

    # Reference output (float32)
    x = rng.standard_normal((32, in_f)).astype(np.float32)
    y_ref = x @ W_fp32.T + b_fp32

    # Per-channel quantized layer
    layer_pc = QuantizedLinear.from_fp32(W_fp32, b_fp32, per_channel=True)
    print(layer_pc)
    mem = layer_pc.weight_memory_bytes()
    print(f"  Weight memory: FP32={mem['fp32_bytes']/1024:.1f}KB  "
          f"INT8={mem['int8_bytes']/1024:.1f}KB  "
          f"({mem['compression_vs_fp32']}x compression)")

    # Calibrate on some data
    for _ in range(10):
        calib_x = rng.standard_normal((16, in_f)).astype(np.float32)
        layer_pc.calibrate(calib_x)
    layer_pc.finalize_calibration(method="minmax")

    y_pc = layer_pc(x)
    snr = quantization_snr(y_ref, y_pc)
    print(f"  Per-channel SNR vs FP32: {snr:.2f} dB")
    print(f"  Max |error|: {np.max(np.abs(y_ref - y_pc)):.6f}")

    # Per-tensor for comparison
    layer_pt = QuantizedLinear.from_fp32(W_fp32, b_fp32, per_channel=False)
    for _ in range(10):
        calib_x = rng.standard_normal((16, in_f)).astype(np.float32)
        layer_pt.calibrate(calib_x)
    layer_pt.finalize_calibration(method="minmax")
    y_pt = layer_pt(x)
    snr_pt = quantization_snr(y_ref, y_pt)
    print(f"  Per-tensor  SNR vs FP32: {snr_pt:.2f} dB")
    print(f"  (Per-channel is better ↑)")

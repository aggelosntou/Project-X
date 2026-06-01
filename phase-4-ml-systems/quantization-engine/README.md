# Quantization Engine

Pure-numpy implementation of INT8 and FP16 quantization from scratch.
Covers symmetric/asymmetric quantization, per-channel quantization, and
a complete Post-Training Quantization (PTQ) pipeline.

## Files

| File | Description |
|------|-------------|
| `src/quantize.py` | Core primitives: INT8 sym/asym, FP16, per-channel |
| `src/quantized_linear.py` | Quantized linear layer with calibration |
| `src/benchmark.py` | Speed / accuracy / memory comparison |
| `src/ptq.py` | End-to-end PTQ pipeline on a trained MLP |

## Running

```bash
pip install -r requirements.txt

python3 src/quantize.py          # Test quantization primitives
python3 src/quantized_linear.py  # Test quantized linear layer
python3 src/benchmark.py         # Full benchmark (FP32/FP16/INT8)
python3 src/ptq.py               # PTQ demo
```

## Concepts

### Quantization Error
The quantization error is the difference between the original float value
and its reconstructed approximation after quantizing to a lower bit-width.
For INT8: `error = x - dequantize(quantize(x))`. The error is bounded by
`scale/2` (rounding to nearest integer).

### Symmetric vs Asymmetric
- **Symmetric**: `scale = max(|x|)/127`, `zero_point = 0`.
  The quantization grid is symmetric around zero. Best for weights
  (which are typically zero-centered).
- **Asymmetric**: `scale = (max-min)/255`, `zero_point = round(-min/scale)`.
  Uses the full [0, 255] range for non-negative values.
  Best for activations after ReLU (range [0, ∞)).

### Why Per-Channel is More Accurate
In a weight matrix, different output channels often have very different
magnitude ranges (e.g., channel 0 has max weight 0.01, channel 1 has
max weight 5.0). Per-tensor quantization must accommodate the largest
channel, wasting bits on the small channel. Per-channel quantization
gives each channel its own scale factor, using the full INT8 range for
every channel. This typically improves SNR by 10–30 dB when magnitudes vary.

### When Quantization Hurts Accuracy Most
- **First and last layers**: These handle raw inputs/logits. Quantization
  error here propagates directly into accuracy.
- **Attention layers**: Softmax outputs are in [0,1] with high dynamic
  range; quantizing them with a coarse grid loses important distinctions.
- **Layer norm / batch norm**: The normalized values have small variance;
  a coarse quantization grid causes significant precision loss.
- **Embedding layers**: Discrete lookup tables quantize well if the
  embedding values are bounded.

### PTQ vs QAT
- **PTQ (Post-Training Quantization)**: Quantize a pre-trained FP32 model
  using only a small calibration dataset. Fast (no retraining), but may
  lose 1–3% accuracy on hard tasks.
- **QAT (Quantization-Aware Training)**: Simulate quantization during
  training using "fake quantize" operations (quantize then dequantize in
  the forward pass, but backprop through them). Recovers accuracy at the
  cost of retraining. Uses straight-through estimator (STE) for gradients
  through the non-differentiable round() operation.

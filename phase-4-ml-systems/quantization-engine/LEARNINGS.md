# Learnings — Quantization Engine

## Key Mathematical Insights

### 1. The quantization map is a step function
For symmetric INT8:
```
Q(x) = round(x / s) * s,   s = max|x| / 127
```
The maximum error is `s/2`. The signal-to-noise ratio is:
```
SNR ≈ 6.02 * n_bits + 1.76 dB
```
So INT8 gives about 50 dB and FP16 gives about 98 dB. This is why
FP16 is almost transparent and INT8 requires careful calibration.

### 2. Why per-channel quantization helps
If channel 0 has max weight 0.01 and channel 1 has max weight 5.0,
per-tensor scale = 5.0/127 ≈ 0.04.
Channel 0's weight 0.01 / 0.04 ≈ 0.25 → quantizes to 0 or 1 — massive relative error.
Per-channel gives channel 0 scale = 0.01/127 ≈ 0.00008, so it uses the full range.

### 3. Asymmetric quantization for ReLU outputs
After ReLU, all values are non-negative. Symmetric quantization wastes half
the INT8 range (the negative half). Asymmetric quantization maps [min, max]
to [0, 255], using all 256 levels efficiently.

### 4. Why PTQ works most of the time
Modern over-parameterized networks have enough redundancy that small
perturbations (quantization noise) don't change the argmax of the output.
PTQ accuracy drop < 1% for most vision models at INT8.

### 5. When to use QAT instead of PTQ
- Small models (MobileNet-scale): more sensitive to perturbation
- Tasks requiring very high accuracy (medical imaging)
- INT4 quantization: much larger error, needs training to adapt
- Recurrent models: quantization errors accumulate over time steps

## Surprising Findings
- FP16 is almost lossless: SNR > 90 dB in practice
- Per-channel improves SNR by 15–25 dB when channel magnitudes vary 10–100x
- The first/last layer is most sensitive: many frameworks skip quantizing them
- Calibration dataset size matters little beyond ~100 samples for minmax

## Connection to Information Theory
Quantization is lossy compression. The rate-distortion tradeoff says:
for a Gaussian source, using n bits gives MSE ≈ σ²/2^(2n).
Going from FP32 (32 bits) to INT8 (8 bits) means 24 fewer bits,
so distortion increases by 2^48 in theory — but in practice the signal
has much less entropy than a full Gaussian, so INT8 works well.

# Learnings — Edge Deployment

## What Is ONNX?

**ONNX** (Open Neural Network Exchange) is a framework-agnostic format for
representing ML models.  The computation graph is serialised as a
**Protobuf** message; each node in the graph corresponds to a standard
operator (Conv, MatMul, ReLU, Softmax, …).

Why it matters:
- Train in PyTorch or TensorFlow, deploy in onnxruntime, TensorRT, or
  CoreML without rewriting the model.
- Hardware vendors can optimise once for ONNX operators and every
  framework benefits automatically.
- `torch.onnx.export()` traces the forward pass and records the operator
  graph; `onnxruntime.InferenceSession` then runs it with its own
  optimised kernels.

## Operator Fusion

When two operators appear consecutively (e.g. Conv → ReLU), a naive
runtime launches two GPU kernels:
1. Conv writes its output to DRAM.
2. ReLU reads from DRAM, applies max(0,x), writes back to DRAM.

**Fused Conv+ReLU** combines both into a single kernel that keeps the
intermediate result in registers or L1 cache — no round-trip to DRAM.

Typical savings:
- 1 memory read + 1 memory write eliminated per fused pair.
- Each DRAM round-trip costs ~400 GPU cycles (memory wall).
- For transformer blocks, fusing QKV projection + bias + dropout can
  reduce kernel-launch overhead by 5–10×.

Common fused patterns:
- Conv + BatchNorm + ReLU
- Linear + GELU / SiLU
- Attention score + Softmax + Dropout
- LayerNorm (mean + variance + normalise + scale + shift in one pass)

## Why C Inference Is Fastest at batch_size = 1

At very small batch sizes the arithmetic is trivial — a single 784→256
GEMV (matrix-vector multiply) takes roughly 400 µs in Python/NumPy
because:
- The Python call stack (NumPy dispatcher → BLAS) has ~5–10 µs overhead.
- NumPy checks dtype, shape, and strides on every call.
- The GIL serialises any concurrent Python work.

The compiled C binary does none of that.  At batch_size=1 the C engine
is typically **10–50× faster** than Python because the Python overhead
dominates the actual arithmetic.

At batch_size=128 the BLAS GEMM dominates and NumPy catches up to within
2–3× of C.

## TFLite and CoreML: What They Add for Mobile

| Feature | TFLite | CoreML |
|---------|--------|--------|
| Target hardware | ARM CPU, GPU, DSP, EdgeTPU | Apple Neural Engine, GPU, CPU |
| Quantisation | INT8, INT4, float16 | INT8, float16 |
| Hardware acceleration | Android NNAPI, Hexagon DSP | ANE (automatic) |
| Power constraints | Manual power profiles | Automatic throttling |
| Model format | FlatBuffer (.tflite) | mlpackage / mlmodel |

**Quantisation** reduces model size 4× (float32 → int8) and typically
runs 2–4× faster with only 1–2% accuracy loss.  On mobile this matters
because:
- Battery life: INT8 SIMD does 4 ops per cycle vs 1 for float32.
- Memory bandwidth: smaller activations → more fits in cache.
- Thermal throttling: lower power draw avoids clock-speed reduction.

## The Float16 Trade-off

float16 has:
- Half the memory footprint → better cache utilisation.
- Possible 2× SIMD throughput on CPUs that support FP16 arithmetic
  (Apple M-series, newer x86 with AVX-512 FP16).
- Reduced precision (3 decimal digits vs 7 for float32) — usually fine
  for inference, never use for loss computation during training.

On older x86 CPUs NumPy emulates float16 with float32 internally, so
the conversion overhead can actually make fp16 *slower* than fp32.
Always benchmark on your target hardware.

# Edge Deployment — Model Export & CPU Inference

This project demonstrates how to export a trained neural network and run
efficient CPU inference, comparing Python/NumPy and hand-written C.

## Contents

| File | Purpose |
|------|---------|
| `src/model_definition.py` | Self-contained MLP (NumPy) |
| `src/export_weights.py` | Export to binary / NumPy .npz |
| `src/c_inference.c` | C inference engine (forward pass + benchmark) |
| `src/benchmark_inference.py` | Side-by-side Python vs C comparison |
| `Makefile` | Build and run targets |

## Quick Start

```bash
# 1. Install Python deps
pip install -r requirements.txt

# 2. Export model weights to model.bin and model.npz
make export

# 3. Compile the C inference engine
make

# 4. Run the full benchmark (Python float32, float16, and C)
make bench
```

## Binary Format

The custom `.bin` file is self-describing — no external schema needed:

```
[int32]  n_layers
For each layer:
    [int32]  rows
    [int32]  cols
    [float32 × rows×cols]  weights  (row-major)
    [int32]  bias_len
    [float32 × bias_len]   biases
```

This is intentionally simple: no versioning, no checksums.  A
production format would add a magic number and version field.

## Why C Is Faster

- **No interpreter overhead** — Python bytecode dispatch costs ~50–100 ns
  per instruction even before NumPy is called.
- **Compiler optimisations** — `-O3` enables auto-vectorisation (AVX/SSE),
  loop unrolling, and constant-propagation that NumPy cannot apply at
  runtime.
- **Memory layout control** — the C code pre-allocates exactly two scratch
  buffers and ping-pongs, producing zero GC pressure.
- **No Python object boxing** — NumPy arrays carry reference counts,
  shape metadata, and dtype checks; raw C arrays do not.

The gap is most pronounced at **batch_size = 1** where Python overhead
dominates.  At large batch sizes NumPy's BLAS back-end (OpenBLAS / MKL)
closes the gap because the GEMM itself dominates runtime.

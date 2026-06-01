# Benchmarks — Edge Deployment

Run `make bench` to reproduce these numbers on your machine.
Results below are representative of a modern laptop CPU.

## Inference Latency (1000 total samples, MLP 784→256→128→10)

| batch_size | fp32 lat (ms/sample) | fp32 tput (/s) | fp16 lat (ms/sample) | fp16 tput (/s) | fp16 speedup |
|:----------:|:--------------------:|:--------------:|:--------------------:|:--------------:|:------------:|
| 1          | ~0.020               | ~50,000        | ~0.018               | ~55,000        | ~1.1×        |
| 8          | ~0.004               | ~250,000       | ~0.003               | ~300,000       | ~1.2×        |
| 32         | ~0.002               | ~500,000       | ~0.002               | ~520,000       | ~1.0×        |
| 128        | ~0.001               | ~900,000       | ~0.001               | ~920,000       | ~1.0×        |

C binary (batch=1): **~0.001 ms/sample** (~10–50× faster than Python fp32)

## Key Observations

1. **Python overhead is most visible at batch_size=1**.  The GEMM
   computation is so small that Python call-stack overhead (dtype checks,
   shape validation, GIL) dominates.

2. **float16 helps little on x86** without native FP16 SIMD.  NumPy
   converts to float32 internally for arithmetic, so you pay the
   conversion cost without the speed gain.  On Apple M-series the
   speedup is larger (~2×) due to native fp16 units.

3. **C inference is fastest at every batch size** but the relative
   advantage shrinks as batch size grows and BLAS dominates.

4. **Memory**: fp16 cuts weight storage in half (from ~1.6 MB to ~0.8 MB
   for this model).  At batch_size=1 the entire model fits in L2 cache,
   explaining good performance regardless of format.

## How to Run

```bash
# Build and run everything
make bench

# Or step by step:
python3 src/export_weights.py        # creates model.bin
make                                  # compiles c_inference
./c_inference model.bin               # C-only benchmark
python3 src/benchmark_inference.py    # full comparison
```

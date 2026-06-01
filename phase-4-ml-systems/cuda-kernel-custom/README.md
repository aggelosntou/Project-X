# CUDA Kernel Custom — Attention Implementations

Three progressively optimised implementations of scaled dot-product
multi-head attention, plus a fused LayerNorm kernel.

## Files

| File | Description |
|------|-------------|
| `src/attention_naive.cu` | 3 separate kernel launches, O(N²) scores tensor |
| `src/attention_shared_mem.cu` | Tiled Q/K/V loads via shared memory |
| `src/flash_attention.cu` | FlashAttention: online softmax, O(N) memory |
| `src/layer_norm_kernel.cu` | Fused mean+var+norm+scale in one kernel |
| `src/benchmark.cu` | Benchmark harness, prints comparison table |
| `Makefile` | Builds everything with nvcc |

## Requirements

- CUDA toolkit ≥ 11.0 with `nvcc`
- GPU with compute capability ≥ 7.0 (Volta/Turing/Ampere)

## Quick Start

```bash
# Build all kernels together (benchmark binary)
make

# Run benchmark
./benchmark

# Or compile and test individual kernels
make naive   && ./naive_test
make shared  && ./shared_test
make flash   && ./flash_test
make layernorm && ./layernorm_test
```

## Expected Output

```
Device: NVIDIA A100-SXM4  (sm_80)
Global memory: 40536 MB

=== Attention Benchmark  batch=1  heads=8  d_head=64 ===
seq_len    naive_ms  shared_ms   flash_ms   naive_MB   flash_MB
--------  ----------  ----------  ----------  ----------  ----------
      64      0.0231      0.0180      0.0145        1.25       0.25
     128      0.0612      0.0420      0.0310        4.75       0.50
     256      0.2134      0.1420      0.0892       17.75       1.00
     512      0.8012      0.5231      0.2140       68.75       2.00
    1024      3.2010      2.0140      0.6230      268.75       4.00
```

## Memory Comparison (seq_len=1024, 8 heads)

| Implementation | Score tensor | Total |
|---------------|-------------|-------|
| Naive          | 268 MB      | 272 MB |
| Shared memory  | 268 MB      | 272 MB |
| FlashAttention | 0 MB        | 4 MB  |

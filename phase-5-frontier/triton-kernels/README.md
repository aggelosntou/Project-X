# triton-kernels

**Phase:** 5 — Frontier ML Systems  
**Language:** Triton (OpenAI), Python  
**Mirrors:** How ML engineers at Anthropic, OpenAI, and Meta write custom GPU kernels today  
**Difficulty:** 9 / 10

---

## What this is

A collection of production-quality GPU kernels written in Triton — the language that lets you write CUDA-level performance kernels in Python-like syntax by describing tile-level operations rather than thread-level ones. You will implement four kernels:

1. **Fused softmax** — compute softmax over a row in a single kernel pass, avoiding the two-pass (max then sum) pattern that requires two global memory reads
2. **Fused LayerNorm / RMSNorm** — compute mean, variance, normalize, and scale in one pass; implement the backward pass including the gradient through the normalization
3. **FlashAttention-style fused attention in Triton** — the same tiled algorithm as `flash-attention-cuda` but written in Triton, to compare the abstraction cost against hand-written CUDA
4. **Fused cross-entropy** — the logit → softmax → log → NLL computation in one kernel, critical for training because storing logits for 32k-vocab softmax at batch×seq_len is the dominant memory cost in a training step

Each kernel is benchmarked against PyTorch eager, `torch.compile`, and where applicable the CUDA implementation from the adjacent project.

---

## Why this matters

Triton is how frontier labs write custom kernels in 2024–2026. The reason: Triton handles the tile-loop and shared memory orchestration automatically, letting you focus on the algorithm rather than `__syncthreads()` placement. This is why the FlashAttention-2 reference implementation is in Triton. Understanding both Triton and raw CUDA makes you dangerous: you can prototype in Triton, profile, and drop to CUDA only where necessary. Engineers who understand both are rare.

The fused cross-entropy kernel alone saves ~30% of the memory footprint during LLM training — this is not a toy optimization.

---

## Papers and references to read first

| Reference | What to extract |
|---|---|
| Tillet et al., *Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations* (2019) | Read §3 (language design) and §4 (compilation). Understand what `tl.load`, `tl.store`, and `tl.dot` compile to and why the tile abstraction enables auto-tuning. |
| Dao et al., *FlashAttention-2* (2024) | The Triton reference implementation is in the supplementary code. Read it *after* you have written your CUDA version — the contrast is the lesson. |
| NVIDIA, *Nsight Compute Documentation* | You will profile with `ncu`. Read the sections on occupancy, warp efficiency, and memory throughput. |
| OpenAI Triton tutorials (triton-lang.org) | The vector addition, fused softmax, and matrix multiplication tutorials. Do all three before starting this project. |

---

## Architecture

```
triton-kernels/
├── kernels/
│   ├── softmax.py              # Fused softmax (online max + sum in one pass)
│   ├── layernorm.py            # Fused LayerNorm forward + backward
│   ├── rmsnorm.py              # RMSNorm (no mean subtraction — used in LLaMA)
│   ├── flash_attention.py      # Tiled causal attention in Triton
│   └── cross_entropy.py        # Fused logit → log-softmax → NLL
├── autotune/
│   ├── softmax_configs.py      # Block size sweep: BLOCK_SIZE ∈ {64, 128, 256, 512, 1024}
│   ├── attention_configs.py    # Br × Bc grid search for the attention kernel
│   └── run_autotune.py         # Run all configs, save best, print roofline position
├── benchmarks/
│   ├── bench_softmax.py        # vs torch eager, vs torch.compile, vs torch.nn.Softmax
│   ├── bench_layernorm.py      # vs torch.nn.LayerNorm (forward and backward)
│   ├── bench_attention.py      # vs PyTorch SDPA, vs your flash_attn_cuda
│   ├── bench_cross_entropy.py  # vs F.cross_entropy — memory profile is the key benchmark
│   └── bench_e2e.py            # Full transformer layer (attn + norm + FFN) all fused
└── tests/
    ├── test_softmax.py         # atol=1e-4 vs torch reference, test large rows (N=65536)
    ├── test_layernorm.py       # Forward + backward correctness, test edge: batch=1
    ├── test_attention.py       # Match SDPA output, test causal masking
    └── test_cross_entropy.py   # Match F.cross_entropy, test vocab=128256 (LLaMA vocab size)
```

---

## Implementation plan

### Kernel 1 — Fused softmax

**The problem with naïve softmax:** Computing `softmax(x)` row-wise requires:
1. Pass 1: find `max(x)` — one read of x
2. Pass 2: compute `exp(x - max) / sum(exp(x - max))` — another read of x

Two passes = two global memory reads per row. For large sequence lengths this is the bottleneck.

**The Triton solution:** Load the entire row into registers (if it fits in SRAM) in one pass and compute the online max and sum simultaneously. This is only possible because Triton's block abstraction lets you describe "operate on a tile of BLOCK_SIZE elements" and the compiler manages the register allocation.

Your kernel signature:
```python
@triton.jit
def softmax_kernel(
    output_ptr, input_ptr,
    input_row_stride, output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr
):
```

Key decisions to make and justify:
- What BLOCK_SIZE is optimal for your GPU? (Run the autotune sweep.)
- When does the row not fit in a single tile? (N > BLOCK_SIZE.) How do you handle multi-tile rows with online max?
- How do you handle numerical stability identically to PyTorch (subtract max before exp)?

**Gate:** Output matches `torch.softmax(x, dim=-1)` to atol=1e-4. Benchmark shows ≥ 2× speedup over eager for N=32768 (a typical vocabulary-sized row in cross-entropy).

### Kernel 2 — Fused LayerNorm and RMSNorm

LayerNorm computes:
```
y = (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta
```

RMSNorm (used in LLaMA, Mistral, Gemma) computes:
```
y = x / sqrt(mean(x²) + eps) * gamma
```

The forward pass can be fused: load x once, accumulate mean and sum-of-squares, normalize, scale, store. The backward pass is harder: `dL/dx` involves a sum over the hidden dimension that requires knowing the normalized value for every element, which means either storing the normalized intermediate or recomputing it during the backward.

Implement the backward pass using the technique in the Triton LayerNorm tutorial — store the per-row mean and inverse-std in the forward pass, use them in the backward to avoid recomputation.

**Gate:** Forward matches `torch.nn.LayerNorm` to atol=1e-4. Backward passes `torch.autograd.gradcheck`. Benchmark shows ≥ 2× speedup over eager on hidden_size=4096, batch=32 (realistic LLaMA dimensions).

### Kernel 3 — FlashAttention in Triton

Implement the same tiled causal attention as `flash-attention-cuda`, but in Triton. You should find:
- The algorithm is identical
- The code is significantly shorter (no `__shared__` declarations, no `__syncthreads()`, no warp reductions)
- The performance is within 10–20% of hand-written CUDA for most use cases

The learning is in the *comparison*, not just the implementation. After writing both, document concretely what Triton handles for you and what you lose control over. This is the tradeoff every engineer using Triton must understand.

**Gate:** Output matches PyTorch SDPA to atol=1e-2 (FP16). Causal masking is correct. Benchmark comparison table: Triton vs your CUDA FA-2 vs PyTorch SDPA across seq_len ∈ {512, 1024, 2048, 4096}.

### Kernel 4 — Fused cross-entropy

Standard `F.cross_entropy(logits, labels)` in PyTorch:
1. Materializes logits: `[batch × seq_len, vocab_size]` — for LLaMA with vocab=128k, batch=8, seq_len=2048, this is 8×2048×128256×2 bytes = ~4 GB
2. Computes softmax over that materialized tensor
3. Takes log
4. Gathers the label probabilities

Fused cross-entropy never materializes the full softmax matrix in HBM. It computes the log-softmax and NLL reduction in one kernel pass, reading logits once and writing only the scalar loss (and per-token losses for the backward pass).

This is the single highest-impact kernel for LLM training memory usage. Profile the memory difference empirically — it should be nearly 2× reduction in peak memory for a training step.

**Gate:** Loss matches `F.cross_entropy` to atol=1e-3. Gradient matches to atol=1e-3. Memory profile shows the expected reduction vs PyTorch eager at vocab=128256, batch×seq=16384.

### Step 5 — Autotuning

Triton's `@triton.autotune` decorator sweeps over configurations and caches the best one. For each kernel, define a grid of configurations (block sizes, number of warps, number of stages for software pipelining) and run the autotune sweep. Document:
- The winning configuration for your GPU
- The sensitivity: how much does performance degrade at the second-best configuration?
- Whether the optimal configuration matches the theoretical prediction from the roofline model

### Step 6 — End-to-end transformer layer benchmark

Assemble a complete transformer layer using your fused kernels:
```
x → RMSNorm → FlashAttention → RMSNorm → FFN (fused GEMM + SwiGLU)
```

Compare end-to-end throughput (tokens/sec) against:
- PyTorch eager
- `torch.compile` (the compiler should fuse many of these automatically)
- Your fused kernels

`torch.compile` is the honest baseline — it is what you are competing against in practice. Your fused kernels should win on memory, and match or beat it on speed for large batch sizes.

---

## What you will understand when this is done

- How Triton compiles to PTX and why it can match CUDA performance for memory-bound kernels
- The correct abstraction level for writing GPU kernels in 2024: Triton for most things, raw CUDA for the last 10%
- Why fused kernels win: fewer global memory round-trips, not fewer FLOPs
- How to autotune GPU kernels systematically using empirical sweep rather than guessing
- The actual memory cost of LLM training at vocabulary scale and why fused cross-entropy matters

---

## Completion criteria

- [ ] All four kernels pass numerical correctness tests
- [ ] LayerNorm backward passes `gradcheck`
- [ ] Autotune sweep run and results documented
- [ ] FlashAttention Triton vs CUDA comparison table complete
- [ ] Fused cross-entropy memory reduction measured and documented (should be ≥ 40% reduction in peak memory for LLaMA-scale vocabulary)
- [ ] End-to-end transformer layer benchmark vs `torch.compile` documented in `BENCHMARKS.md`

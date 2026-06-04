# flash-attention-cuda

**Phase:** 5 — Frontier ML Systems  
**Language:** CUDA C++, Python (benchmarking harness)  
**Mirrors:** FlashAttention-2 (Dao et al.), FlashAttention-3 (Tri Dao, 2024)  
**Difficulty:** 10 / 10

---

## What this is

A from-scratch CUDA implementation of the FlashAttention algorithm: tiled matrix multiplication over Q, K, V with an online softmax computation that avoids materializing the full N×N attention matrix in HBM (high-bandwidth memory).

This is not a wrapper. This is the actual algorithm, written in CUDA, with the memory access pattern designed explicitly around the GPU memory hierarchy. The insight is not about FLOPs — standard attention and FlashAttention compute the same number of floating-point operations. The insight is that HBM reads/writes are the bottleneck, and tiling reduces them by a factor of N/block_size.

You will implement this, benchmark it against PyTorch's eager attention and the Triton version from the adjacent project, and reproduce the roofline analysis from the paper to prove you understand *why* it is fast.

---

## Why this matters

FlashAttention is the single most impactful algorithmic contribution to transformer inference and training since the transformer itself. Every production LLM today uses it. Reading the paper and understanding the algorithm are table stakes; being able to write the CUDA kernel is the engineering proof that you actually understand it. This project is the difference between knowing FlashAttention and being the person who could have written it.

For your SDE/sampling background: the online softmax trick is a numerically stable algorithm for a running maximum — it is the log-sum-exp stabilization you already know from probabilistic computing, applied recursively within a tiled computation. The analysis of numerical error as block size varies is exactly discretization error analysis in a different guise.

---

## Papers to read before writing a single line

| Paper | What to extract |
|---|---|
| Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness* (NeurIPS 2022) | Read §3 (algorithm) and §4 (analysis) with a pencil. Reproduce the IO complexity derivation: O(N² d / M) reads, where M is SRAM size. This is the proof you are optimizing. |
| Dao, *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning* (ICLR 2024) | The key change: partition across sequence length on the forward pass, eliminating synchronization across warps. Read the work decomposition in §3.2. |
| Williams et al., *Roofline: An Insightful Visual Performance Model* (2009) | The analytical framework you will use to understand whether your kernel is compute-bound or memory-bound. Plot your kernel on the roofline before claiming it is fast. |
| NVIDIA, *CUDA Programming Guide*, §5 (Memory Hierarchy) | You cannot write correct tiled kernels without internalizing the L1/shared memory / L2 / HBM hierarchy. This is required reading before writing a single `__shared__` declaration. |
| Dao et al., *FlashAttention-3* (2024) | The current state of the art — asynchronous memory transfers, WGMMA instructions on H100. You will not implement FA3, but you should understand why FA2 is not the ceiling. |

---

## Architecture

```
flash-attention-cuda/
├── kernels/
│   ├── flash_attn_fwd.cu      # Forward pass: tiled Q×K, online softmax, tiled ×V
│   ├── flash_attn_bwd.cu      # Backward pass: recompute attention from tiles (no storing attn matrix)
│   ├── flash_attn_fwd_v2.cu   # FA-2 variant: query-parallel tiling
│   └── utils.cuh              # Shared: warp reductions, tile load/store helpers, numerical guards
├── python/
│   ├── flash_attn.py          # Python bindings via torch.utils.cpp_extension
│   ├── reference.py           # Pure PyTorch reference for numerical correctness checks
│   └── roofline.py            # Measure arithmetic intensity and plot on the roofline
├── benchmarks/
│   ├── bench_forward.py       # Forward pass: wall time, GB/s, TFLOPS vs seq length
│   ├── bench_backward.py      # Backward pass: same metrics
│   ├── bench_compare.py       # Your CUDA vs Triton FA vs PyTorch eager vs sdpa
│   └── memory_profile.py      # Peak HBM usage vs N — confirm O(N) not O(N²)
└── tests/
    ├── test_fwd_correctness.py # atol=1e-2 vs reference (FP16 accumulation drift is expected)
    ├── test_bwd_correctness.py # Gradient check via torch.autograd.gradcheck
    └── test_causal_mask.py     # Causal variant: lower-triangular mask correctness
```

---

## Implementation plan

### Step 1 — Understand the algorithm mathematically, before any CUDA

Write out the standard attention computation:

```
S = Q K^T / sqrt(d)        [N × N]
P = softmax(S, dim=-1)     [N × N]
O = P V                    [N × d]
```

Now write out what FlashAttention does instead. For a tile of Q with rows [i:i+Br] and a tile of K,V with columns [j:j+Bc]:

```
S_ij = Q_i K_j^T                              (Br × Bc matmul, in SRAM)
m_i^new = max(m_i, rowmax(S_ij))              (running max, for softmax stability)
l_i^new = exp(m_i - m_i^new) * l_i + rowsum(exp(S_ij - m_i^new))   (running sum)
O_i = diag(exp(m_i - m_i^new))^{-1} * O_i + exp(S_ij - m_i^new) * V_j
```

Write this derivation in your notebook, not from memory of the paper — derive it. Then verify that when all tiles are summed, O_i equals the correct attention output. This is the math that makes the algorithm correct. You cannot write the CUDA until you can write the math.

### Step 2 — Naïve blocked attention in CUDA (no online softmax)

Before implementing online softmax, implement blocked attention that materializes the full S matrix but uses shared memory for the Q, K, V tiles. This confirms you can write a correct tiled GEMM in CUDA.

Key CUDA concepts you must use:
- `__shared__ float smem_Q[Br][d], smem_K[Bc][d]` — tile staging in SRAM
- `__syncthreads()` placement — after every tile load, before any computation that reads the loaded data
- Coalesced global memory access — thread i in block loads element i, not element i*stride
- Warp-level reduction for the softmax denominator

**Gate:** Output matches `torch.nn.functional.scaled_dot_product_attention` to 1e-3 on FP32 inputs, sequence length 512, d=64.

### Step 3 — Online softmax

Now remove the materialized S matrix. The tile loop over K,V blocks must maintain running statistics (m_i, l_i) and update O_i incrementally using the derivation from Step 1. The S tile is computed in registers, not stored to global memory.

This is the step that makes the algorithm memory-IO efficient: the N×N matrix never touches HBM.

**Gate:** Output matches reference to 1e-2 on FP16 inputs (acceptable due to FP16 accumulation). Memory profiling shows peak HBM usage is O(N·d), not O(N²). This is the key correctness check.

### Step 4 — Causal masking

For autoregressive LLMs, attention is causal: position i can only attend to positions j ≤ i. In the tiled implementation, this means:
- Tiles where j_block > i_block are skipped entirely (zero contribution)
- Tiles on the diagonal require a within-tile mask: `S_ij[i, j] = -inf if j > i`

The causal mask halves the compute for a sequence of length N (you only compute the lower triangle). Measure that your TFLOPS counter reflects this.

**Gate:** Output matches causal reference. Verify that for a causal sequence, position 0 cannot see position 1 (set K[1] to garbage and assert O[0] is unchanged).

### Step 5 — Backward pass

The FlashAttention backward pass recomputes the attention weights from Q, K, V during the backward pass rather than storing them — this is the memory win for training. The backward pass is significantly more complex than the forward.

The gradient flow:
```
dV = P^T dO
dP = dO V^T
dS = P * (dP - rowsum(dP * P))   (the "delta" term)
dQ = dS K / sqrt(d)
dK = dS^T Q / sqrt(d)
```

All of these must be computed without materializing P (the N×N attention matrix). You will tile over the same blocks as the forward pass and recompute P_ij from the saved m_i and l_i statistics.

**Gate:** `torch.autograd.gradcheck` passes on your CUDA backward with eps=1e-2 (FP16 numerical differentiation tolerance).

### Step 6 — FlashAttention-2 parallelism improvement

FA-1 tiles over the K,V dimension in the outer loop and Q in the inner loop, which forces serialization over the Q dimension within a block. FA-2 swaps the loop order: outer loop over Q tiles, inner loop over K,V tiles. This allows different thread blocks to independently compute output rows, eliminating the need for atomic adds across blocks.

Implement the FA-2 tiling strategy and measure the speedup. On an A100, FA-2 should be 1.5–2× faster than FA-1 at sequence length 2048+.

### Step 7 — Roofline analysis and benchmarks

For every kernel variant, measure:
- **Arithmetic intensity** (FLOPs / bytes transferred to/from HBM): `2Nd²` FLOPs, `O(Nd)` bytes — compute the exact ratio for your block sizes
- **Achieved memory bandwidth** (GB/s): from `nvprof` or `ncu` — should approach 80% of peak A100 HBM bandwidth (2 TB/s)
- **Achieved compute** (TFLOPS): should be close to the memory bandwidth limit (you are memory-bound, not compute-bound, until d is very large)

Plot your kernels on the roofline diagram. If you are not approaching the memory bandwidth roofline, find out why — common causes are misaligned loads, bank conflicts in shared memory, or insufficient occupancy.

| Benchmark | Seq len | Expected result |
|---|---|---|
| Your CUDA FA-1 vs PyTorch eager | 1024 | ≥ 3× speedup |
| Your CUDA FA-1 vs PyTorch eager | 4096 | ≥ 5× speedup |
| Your CUDA FA-2 vs FA-1 | 2048 | ≥ 1.5× speedup |
| Peak HBM memory, N=4096 | — | O(N·d), not O(N²) |
| Backward pass vs PyTorch | 2048 | ≥ 2× speedup |

---

## What you will understand when this is done

- Why FlashAttention is IO-bound, not compute-bound, and why that distinction is everything
- How online softmax works numerically and why it is equivalent to standard softmax
- How to write a production-quality tiled GEMM in CUDA, including warp reductions and shared memory layout
- How to use `ncu` (nsight compute) to measure arithmetic intensity, occupancy, and memory bandwidth
- Why the backward pass of attention is harder than the forward and how recomputation trades compute for memory
- The roofline model as a practical tool, not a theoretical concept

---

## Completion criteria

- [ ] Forward pass matches reference to atol=1e-2 on FP16
- [ ] Backward pass passes `gradcheck` with eps=1e-2
- [ ] Causal masking is correct (verified by isolation test)
- [ ] FA-2 variant is measurably faster than FA-1 at seq_len ≥ 2048
- [ ] Memory usage is O(N·d), confirmed by profiler
- [ ] Roofline plot generated and kernel is within 80% of the memory bandwidth roofline
- [ ] Benchmark table reproduced in `BENCHMARKS.md` with `ncu` logs attached

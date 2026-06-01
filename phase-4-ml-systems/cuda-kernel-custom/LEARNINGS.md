# Learnings — CUDA Kernel Custom

## Why Naive Attention Is Memory-Bandwidth Bound

**Arithmetic intensity** = FLOPs / bytes_of_memory_traffic.

For computing one element of the Q @ K^T score matrix:
- FLOPs: `2 * d_head` (d_head multiplies + d_head adds)
- Memory reads: `2 * d_head * 4 bytes` (one row of Q, one row of K)
- Arithmetic intensity: `(2 * d_head) / (2 * d_head * 4)` = **0.25 FLOPs/byte**

A modern GPU (e.g., A100) has:
- Peak compute: ~312 TFLOPS (BF16)
- Peak memory bandwidth: ~2 TB/s

Roofline crossover ≈ 312 / 2000 ≈ **156 FLOPs/byte**.

Since attention's intensity (0.25) is far below 156, it is **heavily
memory-bandwidth bound** — adding more CUDA cores does nothing; the
bottleneck is reading Q, K, V from DRAM.

## How FlashAttention Achieves O(N) Memory

The naïve approach writes the full N×N score matrix to DRAM so that the
softmax kernel can read it back.  FlashAttention observes:

1. SRAM (shared memory) is ~100× faster to access than DRAM.
2. You can compute softmax **incrementally** using the log-sum-exp trick
   without seeing the entire row at once.
3. Therefore: process queries in blocks, loop over key/value blocks,
   keep running statistics `(m_i, l_i, O_i)` in SRAM, and never write
   the N×N matrix to DRAM.

Memory complexity:
- Naive:         O(N² · heads)    — the score matrix
- FlashAttention: O(N · d · heads) — only Q, K, V, O

At seq_len=1024, d=64, 8 heads, fp32:
- Naive:          8 * 1024² * 4 = **32 MB** score matrix
- FlashAttention: ~0 MB score matrix

## IO Complexity

Let M = SRAM capacity per SM.

- Naive IO:        O(N² · d)          — write/read the score matrix once
- FlashAttention:  O(N · d · ⌈N/M⌉)  = O(N² · d / M)

Since M > d in practice, FlashAttention does fewer total memory ops.
On A100: M ≈ 192 KB, d = 64 floats = 256 bytes → speedup factor ≈ M/d ≈ 750.
In practice the speedup is 2–4× because other factors (compute, softmax
overhead) limit the gain.

## What Fused Kernels Save

Without fusion, a Conv → ReLU → LayerNorm chain:
1. Conv kernel: reads input, writes output to DRAM.
2. ReLU kernel: reads conv output, writes to DRAM.
3. LayerNorm: reads relu output, writes to DRAM.

Each DRAM round-trip: ~400 GPU cycles (≈ 400 ns on A100).
Three kernels = 4 reads + 3 writes = 7 DRAM accesses per element.

Fused Conv+ReLU+LayerNorm:
- Input read once, output written once = 2 DRAM accesses per element.
- ReLU and LayerNorm computed in registers / L1 cache.
- Savings: ~5 DRAM accesses × 400 cycles = ~2000 cycles per element.

For a 768-dim transformer layer on 2048 tokens (≈1.5M elements):
- Savings: ~3 billion cycles ≈ ~1 ms on A100 (3 GHz base clock).
- That is significant when the entire attention layer takes ~5 ms!

## Shared Memory Tiling

The tiled attention kernel loads a `TILE × d_head` block of Q and K into
shared memory.  Within the block, every thread accesses shared memory
(~4 ns latency) instead of DRAM (~400 ns latency).

Global memory reads: reduced by factor `TILE` for the Q/K loads.
Score matrix writes: unchanged — still O(N²).

This is why tiling helps vs naive for compute-dominant regimes but
FlashAttention is ultimately better: it eliminates the O(N²) score
matrix entirely.

## The Online Softmax Trick

Standard softmax requires two passes over a row:
1. Find max (for numerical stability).
2. Compute exp(x - max) and sum.

Online softmax (Milakov & Gimelshein, 2018) merges these into one pass
by maintaining a running max `m` and rescaling the accumulator whenever
a new maximum is found:

```
For each new score s:
    m_new = max(m_old, s)
    l_new = l_old * exp(m_old - m_new) + exp(s - m_new)
    O_new = O_old * exp(m_old - m_new) + exp(s - m_new) * V[s]
```

This is the core of FlashAttention's one-pass approach.

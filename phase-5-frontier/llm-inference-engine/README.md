# llm-inference-engine

**Phase:** 5 — Frontier ML Systems  
**Language:** Python (core), C extensions where needed  
**Mirrors:** vLLM, TensorRT-LLM, SGLang internals  
**Difficulty:** 10 / 10

---

## What this is

A from-scratch LLM inference engine that implements the three algorithmic ideas that define how every frontier lab (Google, Anthropic, OpenAI, Meta) serves language models at scale:

1. **Paged KV cache** — the PagedAttention insight: treat the KV cache like virtual memory, allocating in fixed-size blocks rather than per-sequence. Eliminates fragmentation and lets you serve far more concurrent sequences than the naïve pre-allocated approach.
2. **Continuous batching** — instead of waiting for an entire batch to finish, evict completed sequences and slot in new ones dynamically. This is why production throughput is 3–10× higher than the HuggingFace `generate()` baseline.
3. **Speculative decoding** — use a small draft model to generate k tokens in parallel, verify with the target model in one forward pass, accept the longest valid prefix. Latency per output token drops without changing output distribution.

You will implement all three from first principles, connecting them into a single serving loop, and benchmark each component against a naïve baseline.

---

## Why this matters

The gap between a model that fits on a GPU and a model that *serves* at production quality is 90% these three ideas. Every ML Systems role at a frontier lab — inference optimization, serving infrastructure, model deployment — lives here. Understanding PagedAttention at the implementation level is now a baseline expectation for research engineers at labs working on inference.

The mathematical depth required (online softmax, block-sparse attention, stochastic acceptance in speculative decoding) is non-trivial. This is not a wrapper around vLLM — it is the thing vLLM is.

---

## Papers to read before writing a single line

Read these in order. Each one must be understood at the algorithmic level before you implement it.

| Paper | Why it is required |
|---|---|
| Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention* (vLLM, 2023) | The core idea: paged KV cache. Read the memory fragmentation analysis in §3. Reproduce Figure 3 in your head before you code. |
| Leviathan et al., *Fast Inference from Transformers via Speculative Decoding* (2023) | The acceptance criterion proof (Theorem 1) is critical — speculative decoding is correct by construction, not an approximation. Write out the proof before implementing. |
| Chen et al., *Accelerating Large Language Model Decoding with Speculative Sampling* (2023) | The Google variant; read alongside Leviathan for the full picture. |
| Chetlur et al., *cuDNN: Efficient Primitives for Deep Learning* (2014) | Background on memory layout and batching that the KV cache implementation will depend on. |
| Agrawal et al., *Sarathi-Serve: Efficient LLM Serving by Piggybacking Decodes with Chunked Prefills* (2024) | The current state of the art beyond vLLM — this is where continuous batching goes next. |

---

## Architecture

The engine is organized around a central scheduler that owns the KV cache and routes requests to the attention kernels.

```
llm-inference-engine/
├── engine/
│   ├── scheduler.py          # Continuous batching loop — the main serving loop
│   ├── kv_cache.py           # Paged block allocator + block table management
│   ├── sequence.py           # Sequence state machine (WAITING → RUNNING → FINISHED)
│   └── block_manager.py      # Map logical KV slots to physical memory blocks
├── attention/
│   ├── paged_attention.py    # Attention over non-contiguous KV blocks
│   └── flash_attention.py    # Tiled attention (or call the CUDA impl from project 2)
├── models/
│   ├── gpt2.py               # A small GPT-2 wired to your KV cache (not HF)
│   └── llama.py              # LLaMA-2 7B wired similarly, for benchmarking
├── speculative/
│   ├── draft_model.py        # Small draft network (GPT-2 small or 3-layer MLP)
│   ├── verifier.py           # Token acceptance loop (the Theorem 1 implementation)
│   └── spec_scheduler.py     # Speculative decoding wired into the main scheduler
├── benchmarks/
│   ├── baseline_hf.py        # HuggingFace generate() as the comparison baseline
│   ├── throughput.py         # Requests/sec, tokens/sec across concurrency levels
│   ├── latency.py            # Time-to-first-token and per-token latency distributions
│   └── memory.py             # Peak memory usage vs batch size vs sequence length
└── tests/
    ├── test_kv_correctness.py # Assert paged attention == dense attention numerically
    ├── test_spec_correctness.py # Assert speculative output matches greedy output
    └── test_scheduler.py      # Sequence lifecycle: admission, preemption, completion
```

---

## Implementation plan

Work through these in strict order. Each step has a correctness gate before you move to the next.

### Step 1 — Sequence state machine and scheduler skeleton

Define a `Sequence` class with states: `WAITING`, `PREFILL`, `DECODE`, `PREEMPTED`, `FINISHED`. The scheduler holds a priority queue of waiting sequences and a running set. Implement round-robin admission first, then FCFS, then priority-by-arrival. Do not touch the KV cache yet — use a dummy that always returns zeros.

**Gate:** A scheduler test that admits 10 sequences, runs them for 50 steps, and verifies every sequence eventually reaches FINISHED.

### Step 2 — Paged block allocator

The KV cache is a large pre-allocated GPU tensor: `[num_blocks, 2, num_heads, block_size, head_dim]`. The `2` is for K and V. The block manager hands out integer block IDs. Each sequence owns a *block table* — a list of block IDs, one per block_size tokens. When a sequence needs more KV storage, request a new block from the free list. When it finishes, return all blocks to the free list.

The critical invariant: two different sequences can share a prefix's blocks (copy-on-write). You do not need to implement prefix sharing in step 2 — just the allocator and the correctness property that block contents match what was written.

**Gate:** Allocate blocks for 100 sequences with random lengths. Assert that no two sequences share a block they wrote different content to. Assert that all blocks are returned on sequence completion and the free list is full again.

### Step 3 — Paged attention kernel

Standard attention reads from a contiguous KV buffer. Yours does not — the KV for a sequence is scattered across non-contiguous blocks. Implement a gather: for each query token in the decode step, collect K and V from the block table, then compute attention over the gathered tensors. Start in pure PyTorch (gather + matmul). This is correct but slow — the benchmark will quantify the gap vs a fused CUDA version.

**Gate:** Run paged attention and standard (contiguous) attention on the same input. Assert outputs match to 1e-4 absolute tolerance. This correctness check must pass before any performance work.

### Step 4 — Continuous batching loop

Wire the scheduler, block allocator, and paged attention into a serving loop. The loop runs forever:
1. Admit new sequences from the waiting queue (if free blocks exist).
2. Collect all PREFILL sequences, run a single batched forward pass for their prompt tokens.
3. Collect all DECODE sequences, run a single batched forward pass for one new token each.
4. Update block tables, append new tokens to sequences, mark finished sequences.
5. Release their blocks, admit more waiting sequences.

The key is that PREFILL and DECODE share the same GPU forward pass in a single batch — this is what continuous batching means. You mix sequences at different stages in the same call.

**Gate:** Serve 50 concurrent sequences of mixed lengths with no OOM and no incorrect output relative to greedy single-sequence decoding.

### Step 5 — Speculative decoding

Load a draft model (GPT-2 small works). For each decode step:
1. Run the draft model autoregressively for `k=4` steps to get 4 draft tokens and their probability distributions `q(x)`.
2. Run the target model on all 4 draft tokens in parallel (one forward pass) to get `p(x)` for each.
3. Accept token i with probability `min(1, p(x_i) / q(x_i))`. On rejection, sample a corrected token from `max(0, p - q)` normalized.
4. The number of tokens accepted in one round is geometrically distributed — measure this empirically.

**Gate:** Run speculative decoding and greedy decoding on 100 prompts. Assert outputs are statistically identical (same distribution, not the same tokens — this is a probabilistic test). Measure mean accepted tokens per round — should be > 2.5 for a well-matched draft/target pair.

### Step 6 — Benchmarks

Every benchmark runs on the same hardware with the same model and reports median ± p99 across 1000 requests.

| Benchmark | Baseline | Target |
|---|---|---|
| Throughput (tokens/sec) | HF `generate()`, batch=1 | Your engine at concurrency=32 |
| Time-to-first-token (ms) | HF `generate()`, batch=1 | Your engine under load |
| Memory efficiency | Naïve pre-allocated KV (max_seq_len × batch) | Paged allocation at the same throughput |
| Speculative speedup | Target model, greedy, no draft | Speculative, k=4, same draft model |

Expected results: throughput should be 4–8× higher than the HF baseline at high concurrency. Speculative decoding should reduce per-token latency by 1.5–2.5× on a CPU draft / GPU target setup.

---

## What you will understand when this is done

- Why memory fragmentation kills LLM serving and exactly how paging solves it
- Why continuous batching works and why it requires changing the attention computation, not just the scheduler
- The acceptance criterion for speculative decoding and why it preserves the target distribution exactly
- The real bottleneck in LLM serving: it is not compute, it is memory bandwidth — and you will have measured this
- Why vLLM is fast and what its actual bottlenecks are in 2024+

---

## Completion criteria

- [ ] Paged attention output matches standard attention to 1e-4 on all test cases
- [ ] Continuous batching scheduler runs 50 concurrent sequences with no OOM
- [ ] Speculative decoding output is statistically indistinguishable from greedy
- [ ] Throughput benchmark shows ≥ 4× improvement over HF baseline at concurrency=32
- [ ] Memory benchmark shows ≥ 2× improvement in memory efficiency vs naïve allocation
- [ ] All benchmark numbers are in `BENCHMARKS.md` with hardware spec, model name, and exact commands to reproduce

# distributed-training-internals

**Phase:** 5 — Frontier ML Systems  
**Language:** Python, C (socket primitives), CUDA  
**Mirrors:** Megatron-LM, DeepSpeed ZeRO, PyTorch FSDP internals  
**Difficulty:** 10 / 10

---

## What this is

A from-scratch implementation of the three parallelism strategies that define how frontier models are trained at scale:

1. **Ring-AllReduce** — the collective communication primitive that synchronizes gradients across data-parallel workers. Implemented from raw TCP sockets, then from MPI, then via `torch.distributed`. You will understand what `nccl` is abstracting.
2. **Tensor parallelism** — split individual weight matrices across GPUs so that a single transformer layer runs in parallel across N devices. Implemented from scratch on a small GPT model.
3. **ZeRO-stage-2 optimizer state sharding** — instead of replicating the optimizer state (Adam's m, v buffers) across all data-parallel workers, shard it. On 8 GPUs with Adam, this reduces memory per GPU by 8× for the optimizer state. Implemented from scratch using `torch.distributed`.

Each component is benchmarked for scaling efficiency (achieved throughput / linear-scaling ideal throughput) and communication overhead.

---

## Why this matters

Distributed training is not an advanced topic — it is the foundation of every model above ~10B parameters. Every research engineer at a frontier lab writes distributed training code. Understanding Ring-AllReduce at the socket level means you can debug hangs, identify stragglers, and reason about communication/computation overlap. ZeRO and tensor parallelism are in every modern training run. Building them from scratch is the only way to understand the tradeoffs between the three parallelism strategies — and those tradeoffs determine whether a training run is efficient or not.

For your mathematical background: the ZeRO analysis is a combinatorial optimization problem (minimize replicated state subject to communication constraints), and the scaling efficiency analysis is directly Amdahl's Law with a communication term added.

---

## Papers to read before writing a single line

| Paper | What to extract |
|---|---|
| Rajbhandari et al., *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models* (2020) | Read Table 1 and §3 carefully. The memory formula per GPU as a function of number of workers and ZeRO stage is the key result. Reproduce it analytically before implementing. |
| Shoeybi et al., *Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism* (2019) | Read §3 on tensor parallelism. The specific AllReduce placement in the transformer block (two per layer in the forward pass) is not obvious — draw the communication graph before implementing. |
| Huang et al., *GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism* (2019) | Pipeline parallelism and the bubble analysis: with K pipeline stages, bubble overhead = (K-1)/(K-1+M) where M is microbatches. Compute the optimal M for your setup. |
| Zhao et al., *PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel* (2023) | The FSDP paper is the production realization of ZeRO-3. Read the prefetch and resharding strategy — it is the key to hiding communication latency. |
| Li et al., *PyTorch Distributed: Experiences on Accelerating Data Parallel Training* (2020) | The DDP design paper. Read the bucket gradients section — this is how gradient communication is overlapped with backward computation. |

---

## Architecture

```
distributed-training-internals/
├── collective/
│   ├── ring_allreduce_socket.py   # Ring-AllReduce from raw TCP (no MPI, no NCCL)
│   ├── ring_allreduce_mpi.py      # Same algorithm via mpi4py
│   ├── ring_allreduce_torch.py    # Via torch.distributed, for comparison
│   └── test_allreduce.py          # Correctness: all ranks end with the same sum
├── tensor_parallel/
│   ├── tp_linear.py               # Column-parallel and row-parallel Linear layers
│   ├── tp_attention.py            # Multi-head attention with head-parallel sharding
│   ├── tp_mlp.py                  # Fused GEMM + AllReduce placement for MLP
│   ├── tp_transformer.py          # Full transformer block with tensor parallelism
│   └── test_tp_correctness.py     # TP output == single-GPU output to atol=1e-4
├── zero/
│   ├── zero_stage1.py             # Shard optimizer state across ranks
│   ├── zero_stage2.py             # Shard optimizer state + gradients
│   ├── zero_optimizer.py          # ZeRO-wrapped Adam: step, reduce-scatter, all-gather
│   └── test_zero_memory.py        # Measure per-GPU memory vs number of ranks
├── pipeline/
│   ├── pipeline_stage.py          # One pipeline stage: forward + backward for K layers
│   ├── gpipe_scheduler.py         # Microbatch scheduler: compute bubble fraction
│   └── test_pipeline.py           # End-to-end training on a 4-stage pipeline
├── benchmarks/
│   ├── bench_allreduce.py         # Bandwidth (GB/s) vs message size vs number of ranks
│   ├── bench_scaling.py           # Weak scaling: throughput vs number of GPUs (1,2,4,8)
│   ├── bench_tp.py                # TP throughput vs tensor-parallel degree
│   ├── bench_zero_memory.py       # Memory per GPU vs ZeRO stage vs num ranks
│   └── bench_communication.py     # Communication overhead: compute/comms overlap fraction
└── model/
    ├── gpt_single.py              # Reference single-GPU GPT (for correctness comparison)
    └── gpt_distributed.py         # GPT wired to all three parallelism strategies
```

---

## Implementation plan

### Part 1 — Ring-AllReduce from sockets

Ring-AllReduce is the algorithm that allows N workers to sum their gradient tensors, such that every worker ends with the global sum, using only point-to-point messages (no central coordinator). It runs in two phases:

**Reduce-Scatter phase (N-1 rounds):**
Each worker has a tensor partitioned into N chunks. In round r, each worker sends chunk (rank - r) % N to the next worker and receives a chunk from the previous worker. The received chunk is summed with the local chunk. After N-1 rounds, each worker holds the correct partial sum for one chunk.

**AllGather phase (N-1 rounds):**
Each worker sends its correct chunk to the next worker. After N-1 rounds, every worker holds all N chunks — the global sum.

Implement this with raw Python `socket` module (TCP). Start with 2 processes on localhost. The implementation must handle arbitrary tensor sizes (the last chunk may be smaller than the others).

Message format: length-prefixed binary: first 8 bytes are the tensor byte count as a uint64, then the raw float32 bytes.

**Gate:** 4 processes, each holding a random tensor of shape (1024,). After AllReduce, all 4 processes have tensors equal to the sum to atol=1e-4. Bandwidth measured: should approach theoretical peak for loopback on localhost (typically 10+ GB/s).

### Part 2 — Tensor parallelism on a transformer

In tensor parallelism, each attention layer is split across GPUs at the head level. If there are 16 heads and 2 GPUs, each GPU handles 8 heads. The Q, K, V projections are column-parallel (each GPU computes its own heads' projections from the full input). The output projection is row-parallel (each GPU holds the rows corresponding to its heads, and the outputs are summed with AllReduce).

This requires exactly one AllReduce per transformer layer in the forward pass (after the output projection) and one in the backward pass (before the input gradient is used by the previous layer). The placement is specific — misplacing the AllReduce is a common bug that produces correct outputs but incorrect gradients.

Implement `ColumnParallelLinear` and `RowParallelLinear` as drop-in replacements for `nn.Linear`. They must:
- Initialize with the weight shard for this rank (not the full weight)
- Perform local matmul, then AllReduce (RowParallel) or Scatter/Gather (ColumnParallel) as needed
- Pass `gradcheck`

**Gate:** A 4-layer transformer running on 2 GPUs with tensor_parallel_degree=2 produces output identical to the same model on 1 GPU, for both forward pass and gradients, to atol=1e-4.

### Part 3 — ZeRO-stage-2 optimizer state and gradient sharding

In standard data parallelism (DDP), every GPU holds:
- Parameters: Ψ bytes
- Gradients: Ψ bytes
- Optimizer state (Adam): 2Ψ bytes (m and v)
- Total: 4Ψ bytes per GPU

With ZeRO stage 2, gradients and optimizer state are sharded across N GPUs:
- Parameters: Ψ bytes (still replicated — you gather before forward)
- Gradients: Ψ/N bytes per GPU (reduce-scatter after backward)
- Optimizer state: 2Ψ/N bytes per GPU (each rank owns 1/N of the Adam state)
- Total: Ψ + 3Ψ/N bytes per GPU

For N=8, this is ~Ψ + 0.375Ψ ≈ 1.375Ψ vs 4Ψ — a 2.9× memory reduction.

Implement `ZeROAdam`:
1. After backward, call `reduce_scatter` on gradients — each rank ends with its shard of the gradient
2. Each rank runs Adam on its parameter shard using the local gradient shard
3. Each rank calls `all_gather` on its updated parameter shard so all ranks have the full updated parameters before the next forward pass

**Gate:** ZeROAdam produces the same parameter updates as standard Adam to atol=1e-5 across 100 training steps on a small model. Memory profile shows per-GPU memory matches the ZeRO formula above.

### Part 4 — Scaling efficiency benchmark

Run a GPT-2 scale model (12 layers, d=768, 12 heads) training loop at:
- 1 GPU (baseline)
- 2 GPUs (data parallel)
- 4 GPUs (data parallel)
- 4 GPUs (2× data parallel, 2× tensor parallel)

Measure tokens/sec for each configuration. Compute scaling efficiency = (N × single_GPU_tokens_per_sec) / (N_GPU_tokens_per_sec). Document:
- Where efficiency degrades and why (communication overhead, pipeline bubble, load imbalance)
- What fraction of wall time is spent in AllReduce vs compute (use `torch.profiler`)
- Whether the ZeRO memory savings allow larger batch sizes that recover some efficiency

---

## What you will understand when this is done

- Ring-AllReduce at the algorithm level: why it is communication-optimal for gradient synchronization
- The three axes of model parallelism (data, tensor, pipeline) and the tradeoffs between them
- ZeRO's memory formula and why it is the key to training models that do not fit on a single GPU
- How NCCL is implemented underneath and what `torch.distributed` is abstracting
- How to profile distributed training to find the communication bottleneck
- Scaling efficiency as a measurable quantity, not just a concept

---

## Completion criteria

- [ ] Ring-AllReduce from sockets produces correct sum across 4 processes
- [ ] Tensor parallel transformer matches single-GPU output (forward + backward)
- [ ] ZeROAdam produces identical updates to standard Adam
- [ ] ZeRO memory reduction is measured and matches the theoretical formula
- [ ] Scaling efficiency benchmark run on 1, 2, 4 GPUs with results in `BENCHMARKS.md`
- [ ] Communication overhead fraction measured with `torch.profiler` at each scale point
- [ ] `BENCHMARKS.md` contains: hardware spec, model size, batch size, tokens/sec, scaling efficiency, and the command to reproduce every number

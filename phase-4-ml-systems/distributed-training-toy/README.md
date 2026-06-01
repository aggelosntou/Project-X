# Distributed Training Toy

Data-parallel distributed training implemented from scratch using Python
multiprocessing — no PyTorch, no NCCL, no torch.distributed.

## What's Implemented

- **All-reduce** (gather-scatter and ring-allreduce)
- **Broadcast** from rank 0
- **Barrier** synchronization
- **Data-parallel training**: N workers, same model, different data shards
- **Gradient verification**: final losses are identical across workers

## Running

```bash
pip install -r requirements.txt

# Train with 2 workers (default)
python3 src/train_distributed.py

# Train with 4 workers
python3 src/train_distributed.py --world-size 4

# Use ring-allreduce
python3 src/train_distributed.py --world-size 4 --ring

# Compare strategies
python3 src/compare_strategies.py
```

## Key Concepts

### Why Gradient Averaging, Not Model Averaging

In data-parallel training, each worker computes gradients on its local shard.
We average the gradients (all-reduce with mean), then each worker applies
the same update. This is mathematically equivalent to computing the gradient
on the full dataset:

```
(1/N) * sum_i ∇L_i(θ) ≈ ∇L_full(θ)
```

If we averaged the models instead, we'd get the same result only for linear
models. For non-linear models, gradient averaging is the correct approach
(and it's what SGD with effective batch size = N * local_batch_size does).

### Ring-Allreduce: Bandwidth-Optimal

In naïve gather-scatter: rank 0 receives from all others → rank 0's inbound
bandwidth is O(N * data_size), a bottleneck.

In ring-allreduce:
- **Reduce-scatter phase**: N-1 rounds, each worker sends 1/N of data to its
  right neighbor, accumulates received chunks.
- **All-gather phase**: N-1 rounds, each worker sends its fully-reduced chunk.

Bandwidth per worker = 2*(N-1)/N * data_size → approaches 2*data_size as N grows.
Every network link carries the same load. This scales perfectly with N.

### Communication Bottleneck

For a model with P parameters (float32 = 4 bytes each):
- All-reduce volume = 2 * P * 4 bytes (send + receive, after ring optimization)
- For GPT-3 (175B params): 2 * 175B * 4 = 1.4 TB per step!
- At 200 GB/s NVLink bandwidth: 7 seconds per step... just for communication.
- This is why gradient compression (1-bit SGD, PowerSGN) matters at scale.

### Parallelism Strategies

| Strategy | Split | When to use |
|----------|-------|-------------|
| Data parallel | Data shards | Most common, easy to implement |
| Model parallel | Layers across devices | When model doesn't fit in one GPU |
| Pipeline parallel | Stages of layers | Large transformers (GPT, PaLM) |
| Tensor parallel | Individual weight matrices | Extreme scale (Megatron-LM) |

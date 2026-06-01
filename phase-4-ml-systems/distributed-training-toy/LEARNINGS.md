# Learnings — Distributed Training Toy

## Mathematical Foundation

### Gradient Averaging = Larger Effective Batch Size
With N workers each processing batch B:
- Each worker computes g_i = (1/B) * sum_{j in shard_i} ∇ℓ(x_j, θ)
- After averaging: g = (1/N) * sum_i g_i = (1/(N*B)) * sum_j ∇ℓ(x_j, θ)
- This equals the gradient of the full batch of size N*B

So data-parallel training with gradient averaging is exactly equivalent to
training with batch size N*B — a clean mathematical equivalence.
Linear scaling rule: lr should scale proportionally with N.

### Ring-Allreduce Bandwidth Analysis

For N workers each holding a vector of size D floats:

**Reduce-scatter phase:**
- N-1 rounds, each worker sends D/N floats
- Bytes sent per worker: (N-1) * D/N * 4

**All-gather phase:**
- Same as reduce-scatter: (N-1) * D/N * 4

**Total bytes per worker:** 2*(N-1)/N * D * 4

**Ratio vs optimal (just sending D floats once):** 2*(N-1)/N → 2 as N→∞

Compare naïve gather at rank 0: rank 0 receives N*D floats = N× worse!
Ring-allreduce is bandwidth-optimal to within a factor of 2.

## Implementation Insights

### Barrier via Shared Counter
Our barrier implementation:
1. Each worker increments a shared counter
2. The last worker (counter == N) wakes all others via Events
3. Everyone waits on their own Event

Race condition note: resetting the barrier is tricky. We have rank 0 reset
the counter, but only after all workers have passed the wait(). This requires
careful ordering — a real implementation uses a generation counter or
semaphores.

### Queue-Based Communication
Python multiprocessing.Queue uses pipes internally. For our toy implementation:
- send: pickle the array and push to pipe
- recv: unpickle from pipe

Real NCCL uses CUDA-aware MPI to transfer data directly between GPU memory
without CPU involvement (GPU Direct RDMA).

## Surprising Findings

1. **Communication dominates for small models**: 30–70% of step time is
   gradient synchronization for our toy model. Real GPUs with compute-heavy
   models spend <5% on communication.

2. **Same loss across workers**: After all-reduce, all workers apply
   identical gradient updates, so they stay perfectly synchronized.
   This is the beauty of synchronous data parallelism.

3. **Python multiprocessing overhead**: Process startup (~0.1s) dominates
   for short training runs. Real distributed training runs for hours/days,
   making this amortized negligible.

## What We Didn't Implement

- **Asynchronous SGD**: workers don't wait for each other → stale gradients,
  but higher throughput
- **Gradient compression**: 1-bit quantization, Top-K sparsification
- **Overlap compute and communication**: start all-reduce while backward
  still running (partial gradient ready-to-send)
- **Fault tolerance**: checkpoint and recover from worker failures

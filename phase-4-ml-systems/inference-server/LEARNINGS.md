# Learnings — Inference Server

## Dynamic Batching

The key insight: a neural network forward pass has fixed overhead (kernel
launch, memory allocation, Python interpreter) that doesn't scale with
batch size — only the compute scales. So batching N requests together costs
roughly the same as 1 request, giving N× throughput.

The tradeoff is added latency (up to `max_wait_ms`). This is acceptable
for throughput-oriented workloads but problematic for real-time systems.

## Optimal Batch Size

- **Memory bound**: The bottleneck is loading weights from memory.
  Larger batches reuse the same weights more, improving efficiency.
  Optimal batch = fill memory bandwidth.
- **Compute bound**: The bottleneck is FLOPS. Larger batches don't help
  as much once all compute units are busy.
- In practice: measure throughput vs batch size; it plateaus when you
  hit either compute or memory saturation.

## Why We Use Async/Threading

Inference servers must handle many concurrent clients without blocking.
The BatchAccumulator uses a background daemon thread to drain the queue,
while the FastAPI async handlers yield control to the event loop while
waiting. This allows thousands of concurrent connections on one server.

## Queue Theory Insights

- If arrival rate > service rate, the queue grows without bound → OOM
- With max_queue_size, we drop excess requests with 503 → bounded latency
- Mean queue length = throughput × mean latency (Little's Law)
- At 90% utilization, queue length ≈ 9× the single-server value (heavy-tailed)

## Why p99 Latency Is the Right Metric

A service making 100 downstream calls: probability that ALL complete within
p50 latency = 0.50^100 ≈ 10^-30. So the fanout makes the tail latency
dominate. Google's engineering blog ("The Tail at Scale") shows that
even small tail-latency improvements can dramatically improve user experience.

## Monitoring in Production

Key signals to watch:
1. **Error rate**: sudden spike → model bug or overload
2. **p99 latency**: degradation → saturation or slow requests
3. **Queue depth**: sustained > 0 → under-provisioned
4. **Batch size**: small batches → requests not concurrent enough

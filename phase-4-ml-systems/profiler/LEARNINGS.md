# Learnings — Training Loop Profiler

## 1. Why backward takes 2–3x longer than forward

### The forward pass

The forward pass computes layer outputs sequentially:

```
x → [W1, b1] → h1 → [W2, b2] → h2 → ... → logits
```

Each layer requires one matrix multiply `h_{i+1} = h_i @ W_i.T + b_i`.
The total FLOPs are exactly `sum_i 2 * B * n_in_i * n_out_i`.

### The backward pass

The backward pass must compute **two** quantities per layer:

1. **Gradient w.r.t. weights** (`dL/dW_i = delta_i.T @ h_i`):
   same shape matrix multiply as forward.

2. **Gradient w.r.t. input** (`dL/dh_i = delta_i @ W_i`):
   another matrix multiply, needed to propagate the gradient to layer i-1.

This means the backward pass performs approximately **2x the matrix
multiplications** of the forward pass, plus:

- **Storing activations** `h_i` during the forward pass (extra memory traffic)
- **ReLU mask reuse** (`delta *= (h_i > 0)`) — extra read of the activation cache
- **Accumulation**: gradients may need to be summed over batch dimension

So the expected ratio is approximately **2–3x** forward time.
This benchmark typically shows 2.5–3.5x depending on layer sizes and hardware.

### Why this matters

When profiling a training step and seeing backward dominate, it is **expected**.
Optimisations to consider:
- **Mixed precision** (float16 for forward, float32 for gradient accumulation)
  halves memory bandwidth for activations.
- **Gradient checkpointing**: trade recompute for memory (recompute some
  activations during backward instead of storing them — costs 30% more compute
  but saves significant memory for deep models).

---

## 2. The roofline model

The **roofline model** characterises whether a computation is:
- **Compute-bound**: the bottleneck is floating-point throughput (TFLOPS)
- **Memory-bound**: the bottleneck is memory bandwidth (GB/s)

### The ridge point

A device has two peak specs:

| Metric | Example (A100 GPU) |
|--------|--------------------|
| Peak compute | 312 TFLOPS (FP16) |
| Memory bandwidth | 2000 GB/s |
| Ridge point = compute/bandwidth | 156 FLOPs/byte |

If your operation has **arithmetic intensity** (FLOPs/byte) **above 156**,
it is compute-bound — you can improve by using faster hardware.
If it is **below 156**, it is memory-bound — more compute won't help;
you need better memory bandwidth or cache efficiency.

### Arithmetic intensity formula

For a linear layer (y = x @ W.T):

```
FLOPs  = 2 * B * in_features * out_features
Bytes  = (W_size + X_size + Y_size) * itemsize
       = (in*out + B*in + B*out) * 4   (float32)
AI     = FLOPs / Bytes
```

Example: batch=1, 784→256:
```
FLOPs = 2 * 1 * 784 * 256 = 401 408
Bytes = (784*256 + 1*784 + 1*256) * 4 = ~807 KB
AI    = 401408 / 807168 ≈ 0.5 FLOPs/byte   ← extremely memory-bound
```

Example: batch=512, 784→256:
```
FLOPs = 2 * 512 * 784 * 256 = 205 520 896
Bytes = (784*256 + 512*784 + 512*256) * 4 = ~3.5 MB
AI    = 205M / 3.5MB ≈ 58 FLOPs/byte   ← still memory-bound on A100
```

### Practical implication

Most ML operations in standard training are **memory-bound** at typical
batch sizes. This is why:
- Larger batches increase arithmetic intensity and improve hardware utilisation.
- Flash Attention re-orders computation to fuse operations and reduce memory
  traffic (increasing effective arithmetic intensity).
- Quantisation (INT8) reduces memory bandwidth pressure even if FLOPs are
  similar, because 4x smaller operands fit better in cache.

---

## 3. How to compute arithmetic intensity for a linear layer

Given: batch B, input dimension m, output dimension n.

```
FLOPs = 2 * B * m * n       (each of B*n outputs is a dot product of length m)

Bytes read:
  - Weight matrix W:  m * n * 4  bytes  (read once)
  - Input X:          B * m * 4  bytes
  - Output Y (write): B * n * 4  bytes

Total bytes = (m*n + B*m + B*n) * 4

Arithmetic intensity (AI) = FLOPs / Bytes
                           = (2 * B * m * n) / ((m*n + B*m + B*n) * 4)
```

For large batches (B >> 1) and large matrices (m, n >> B):

```
AI ≈ 2 * B * m * n / (m * n * 4) = B/2   FLOPs/byte
```

So arithmetic intensity scales **linearly with batch size** for large matrices.
A batch of 64 gives AI ≈ 32 FLOPs/byte; batch 256 gives AI ≈ 128 FLOPs/byte.
The A100 ridge point is ~156 FLOPs/byte, so even batch 256 with 784→256 is
memory-bound.

---

## 4. Why the profiler uses wall-clock time

`time.perf_counter()` measures elapsed real time (wall clock), not CPU time.
This means:
- It includes time spent waiting for memory (cache misses, DRAM latency).
- It is affected by OS scheduling and other processes.
- For profiling ML code on a single thread it is a good proxy for what the
  user actually experiences.

For precise micro-benchmarking, use Python's `timeit` module with `number=1000`
repetitions to average out OS noise. The Profiler is designed for
**coarse-grained phase analysis** (which phase dominates?), not nanosecond
precision.

---

## 5. Memory: what tracemalloc captures and what it misses

`tracemalloc` intercepts Python's memory allocator (`pymalloc`) and NumPy's
`np.empty/zeros/ones` calls (which go through `malloc`). It **does not** capture:

- BLAS library scratch space (OpenBLAS, MKL allocate temporary buffers outside
  Python's allocator).
- Memory-mapped files (`np.memmap`).
- GPU memory (CUDA allocations are invisible to Python's allocator).
- Memory allocated by C extensions that bypass `PyMem_Malloc`.

As a result, `tracemalloc` peak values are a **lower bound** on true process
memory. For accurate system-level memory, use `psutil.Process().memory_info().rss`
(Resident Set Size) sampled at regular intervals.

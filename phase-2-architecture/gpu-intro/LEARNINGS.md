# GPU Architecture — Key Concepts

## 1. CUDA Thread Hierarchy

```
Grid
└── Blocks  (up to 65535 per dimension)
    └── Warps  (hardware unit: 32 threads that execute in lockstep)
        └── Threads  (up to 1024 per block)
```

- **Thread**: the smallest unit of work. Has its own registers and a unique
  (threadIdx, blockIdx) identity.
- **Warp**: 32 threads scheduled together as a single hardware unit by the SM.
  All threads in a warp execute the same instruction at the same time (SIMT).
  If threads diverge (different `if` branches), the warp serialises both paths.
- **Block**: a group of threads that share the same shared memory and can
  synchronise with `__syncthreads()`. Assigned to exactly one SM for its lifetime.
- **Grid**: the full set of blocks launched by one kernel call. Blocks may
  execute in any order on any SM.

### Why warps matter

The GPU hides memory latency by switching between warps. While warp A waits for
a global memory load (~400 cycles), the SM runs warp B, C, D, … When the data
for warp A arrives, it resumes. This is called **latency hiding** and is the
reason GPUs need thousands of live threads to stay busy.

---

## 2. Memory Hierarchy

| Memory type      | Location     | Latency     | Size per SM  | Visible to     |
|------------------|--------------|-------------|--------------|----------------|
| Register         | On-chip      | 0 cycles    | 256 KiB      | 1 thread       |
| Shared memory    | On-chip SRAM | ~4 cycles   | 48–96 KiB    | 1 block        |
| L1 cache         | On-chip      | ~28 cycles  | (shared w/ shmem) | 1 SM     |
| L2 cache         | On-chip      | ~100 cycles | 4–40 MiB     | all SMs        |
| Global memory    | GDDR/HBM     | ~400 cycles | GiB          | all threads    |
| Constant memory  | GDDR + cache | ~4 cycles*  | 64 KiB       | all (read-only)|
| Texture memory   | GDDR + cache | ~100 cycles | GiB          | all (read-only)|

*Constant memory is fast only when all threads in a warp read the same address.

### Practical rule

Anything computed from global memory should be written to shared memory first,
used many times from shared memory, then (optionally) written back once.

---

## 3. Memory Coalescing

When 32 threads in a warp issue a memory load, the GPU hardware attempts to
merge all 32 addresses into as few 128-byte cache-line transactions as possible.

**Coalesced** (stride = 1):
```
thread 0 → byte 0
thread 1 → byte 4
...
thread 31 → byte 124
```
One 128-byte transaction covers all 32 threads. Effective bandwidth = peak.

**Strided** (stride = 2):
```
thread 0 → byte 0
thread 1 → byte 8
...
thread 31 → byte 248
```
Two cache lines needed (bytes 0–127 and 128–255), but only half the bytes in
each line are used. Effective bandwidth ≈ 50% of peak.

**Stride = 32**:
```
thread 0 → byte 0
thread 1 → byte 128
...
```
Each thread touches a separate cache line. 32 transactions. Effective bandwidth
≈ 1/32 of peak — the worst case.

Rule: always try to make threads in a warp access consecutive memory addresses.

---

## 4. Why Shared-Memory Tiling Works

Consider naive matrix multiply C = A × B where C[i][j] = Σ_k A[i][k] * B[k][j].

For an N×N matrix, every output element requires N reads from A and N reads from B.
Across the entire N×N output, that is **2 N³ global memory reads**.

With 16×16 tiling:
- A block of 16×16 threads cooperatively loads a 16×16 tile of A (256 elements)
  and a 16×16 tile of B (256 elements) into shared memory.
- The 256 threads then compute 256 partial dot products reusing those 256+256
  shared-memory values.
- Each element of A and B is loaded from global memory only **N/16 times**
  instead of **N times** — a 16× reduction in global memory traffic.

General rule: if an element participates in T operations and you can stage it
into shared memory once, you reduce global memory traffic by T×.

---

## 5. Occupancy

**Occupancy** = active warps on an SM / maximum warps an SM can hold.

A V100 SM can hold at most 64 warps (2048 threads). If your kernel only has
16 warps active (512 threads), occupancy = 25%.

What limits occupancy:
1. **Registers per thread**: V100 has 65536 registers per SM. If a kernel uses
   64 registers/thread, at most 65536/64 = 1024 threads (32 warps) can be live
   at once — half of maximum.
2. **Shared memory per block**: V100 has 96 KiB per SM. If a block uses 48 KiB,
   at most 2 blocks can run simultaneously, capping occupancy.
3. **Block size**: very small blocks (e.g., 32 threads = 1 warp) limit the
   parallelism available to hide latency.

Higher occupancy generally improves performance by giving the scheduler more
warps to switch to while others wait for memory, but the relationship is not
always linear — a well-optimised kernel with 50% occupancy can outperform a
naive kernel at 100% occupancy if the efficient kernel does fewer memory ops.

Use `nvcc --ptxas-options=-v` to see register usage per thread.

---

## 6. PCIe Bandwidth vs GPU Memory Bandwidth

| Path                    | Bandwidth        |
|-------------------------|------------------|
| CPU ↔ GPU (PCIe 3.0 x16)| ~16 GB/s         |
| CPU ↔ GPU (PCIe 4.0 x16)| ~32 GB/s         |
| GPU global memory (V100)| ~900 GB/s        |
| GPU shared memory       | ~13,000 GB/s     |

The PCIe bottleneck means: always minimise data transfers between CPU and GPU.
Keep data on the GPU across multiple kernels. Use pinned (page-locked) host
memory (`cudaMallocHost`) for faster transfers when transfers are unavoidable.

---

## 7. Parallel Reduction

Reduction (sum, max, min) is a fundamental pattern. The naive approach
(every thread does an `atomicAdd` to a single global counter) serialises all
additions. The tree approach logarithmically reduces the problem:

```
Round 1: N → N/2 (in shared memory, in parallel)
Round 2: N/2 → N/4
...
Round log2(N): 2 → 1
```

For N=1024 threads, this is 10 rounds vs 1024 sequential steps — 100× fewer
synchronisation points. One final `atomicAdd` per block instead of per thread.

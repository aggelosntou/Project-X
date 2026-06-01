# GPU Introduction — CUDA Kernels

Five progressively deeper CUDA programs that build intuition for GPU programming.

## Requirements

- NVIDIA GPU (Volta / Turing / Ampere recommended; minimum Kepler sm_30)
- CUDA Toolkit 10.0 or later
- `nvcc` on your PATH
- Linux or Windows (WSL2 works well)

To target a different GPU architecture, edit `NVCCFLAGS` in the Makefile:
- Kepler (GTX 780, K80): `-arch=sm_35`
- Maxwell (GTX 980): `-arch=sm_52`
- Pascal (GTX 1080, P100): `-arch=sm_60`
- Volta (V100): `-arch=sm_70`
- Turing (RTX 2080): `-arch=sm_75`
- Ampere (A100, RTX 3090): `-arch=sm_80`

## Build

```bash
make          # builds all five programs
make clean    # removes binaries
```

## Programs

### 01_hello_gpu — Device Properties + Hello Kernel

Queries `cudaGetDeviceProperties` and prints a full hardware profile, then
launches a small kernel where every thread prints its global thread ID.

**Expected output** (V100):
```
GPUs found: 1

=== Device 0: Tesla V100-SXM2-16GB ===
  Compute capability       : 7.0
  Streaming Multiprocessors: 80
  Global memory            : 15.8 GiB
  Shared memory per block  : 48 KiB
  Warp size                : 32 threads
  Max threads per block    : 1024
  Max threads per SM       : 2048
  Peak memory bandwidth    : 900.1 GB/s

Launching hello_kernel<<<4, 32>>> (128 total threads)

  Hello from block 0, thread 0  →  global id = 0
  Hello from block 0, thread 1  →  global id = 1
  ... (128 lines total, order may vary between blocks)
```

### 02_vector_add — CPU vs GPU Benchmark

Adds two float arrays of increasing size and times both approaches including
PCIe data transfer.

**Expected output** (V100):
```
  N            | cpu_ms   | gpu_ms   | speedup
  1000         |    0.003 |    0.310 |   0.01x
  10000        |    0.031 |    0.315 |   0.10x
  100000       |    0.304 |    0.342 |   0.89x
  1000000      |    3.121 |    0.483 |   6.46x
  10000000     |   31.450 |    2.021 |  15.56x
```

The crossover point is typically around 100K–500K elements depending on PCIe
generation and GPU memory bandwidth.

### 03_matrix_multiply — Naive vs Tiled Shared Memory

Two kernels for N×N matrix multiplication. The tiled version uses 16×16 shared
memory tiles to reduce global memory traffic by ~16×.

**Expected output** (V100):
```
  N      | naive_ms | tiled_ms | naive_GFLOPS | tiled_GFLOPS
  128    |     0.10 |     0.04 |         42.1 |        104.9
  256    |     0.72 |     0.18 |         46.9 |        187.3
  512    |     5.50 |     0.93 |         48.9 |        289.8
  1024   |    43.20 |     5.10 |         49.8 |        422.1
```

Peak theoretical FLOPS on V100: ~14 TFLOPS FP32. cuBLAS achieves ~12 TFLOPS.
The tiled kernel demonstrates the shared-memory principle at ~422 GFLOPS
(~3% of peak) — already 8× better than the naive version.

### 04_memory_patterns — Memory Coalescing

Reads a 128 MiB array with varying strides and measures achieved bandwidth.

**Expected output** (V100, peak bandwidth 900 GB/s):
```
  stride | bandwidth (GB/s)
       1 |      890.3
       2 |      452.1
       4 |      228.7
       8 |      114.9
      16 |       57.3
      32 |       28.8
```

Each doubling of stride halves effective bandwidth because only half the bytes
in each fetched cache line carry useful data.

### 05_reduction — Parallel Sum Reduction

Sums 1M floats using two strategies: naive global atomics vs. tree reduction
in shared memory.

**Expected output** (V100):
```
  CPU result    : 524287.8750
  Naive result  : 524287.8125  (error 6.25e-02)  time: 2.814 ms
  Tree result   : 524287.8125  (error 6.25e-02)  time: 0.041 ms

  Speedup (tree vs naive): 68.6x
```

The small floating-point error relative to CPU is expected: the GPU uses a
different summation order, and single-precision arithmetic is not associative.

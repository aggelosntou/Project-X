/*
 * 04_memory_patterns.cu
 *
 * Learning goal: understand memory coalescing.
 *
 * A "warp" is a group of 32 threads that execute in lockstep on the GPU.
 * When a warp issues a memory load, the hardware merges the 32 addresses
 * into as few 128-byte cache-line transactions as possible — but only
 * when the addresses are contiguous (coalesced).
 *
 * Coalesced (stride=1):   threads 0-31 read bytes 0-127 → ONE transaction
 * Strided   (stride=2):   threads read bytes 0,8,16,… → multiple cache lines
 * Stride=32:              each thread touches a separate cache line → 32×
 *                         the memory transactions of the coalesced case
 *
 * Compile:  nvcc -O2 -arch=sm_70 -o 04_memory_patterns src/04_memory_patterns.cu
 * Run:      ./04_memory_patterns
 *
 * Expected output (bandwidth in GB/s, ~900 GB/s peak on V100):
 *   stride |  bandwidth (GB/s)
 *        1 |   890.3   ← near-peak (coalesced)
 *        2 |   452.1
 *        4 |   228.7
 *        8 |   114.9
 *       16 |    57.3
 *       32 |    28.8   ← 32x worse (one cache line per thread)
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(call)                                                 \
    do {                                                                 \
        cudaError_t _e = (call);                                         \
        if (_e != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error %s:%d — %s\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(_e));          \
            exit(EXIT_FAILURE);                                          \
        }                                                                \
    } while (0)

/*
 * Kernel: every thread reads one float from d_in and writes to d_out.
 * The access pattern is controlled by 'stride':
 *   stride=1  → thread i reads d_in[i]        (coalesced)
 *   stride=s  → thread i reads d_in[i*s]      (scattered)
 *
 * We use a grid-stride loop to process the entire array of size 'n'.
 * The array is allocated large enough that no thread goes out of bounds.
 */
__global__ void stride_kernel(const float *d_in, float *d_out, int n, int stride)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = blockDim.x * gridDim.x;
    for (int i = tid; i < n; i += total) {
        /* Strided read, safe write (no stride on output so write does
         * not itself become a bottleneck) */
        d_out[i] = d_in[(long)i * stride % n];
    }
}

static void benchmark_stride(int stride, float *d_in, float *d_out, int n, int reps)
{
    int threads = 256;
    int blocks  = (n + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    /* Warm-up pass */
    stride_kernel<<<blocks, threads>>>(d_in, d_out, n, stride);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(t0));
    for (int r = 0; r < reps; ++r)
        stride_kernel<<<blocks, threads>>>(d_in, d_out, n, stride);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    ms /= reps;

    /* Bytes moved: n reads × 4 bytes + n writes × 4 bytes */
    double bytes = 2.0 * n * sizeof(float);
    double bandwidth_gbs = bytes / (ms * 1e-3) / 1e9;

    printf("  %6d | %9.1f\n", stride, bandwidth_gbs);

    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
}

int main(void)
{
    /* Array large enough to fill L2 cache and stress the memory system */
    int n = 1 << 25;   /* 32M floats = 128 MiB */
    int reps = 10;

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in,  (long)n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, (long)n * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_in, 0, (long)n * sizeof(float)));

    printf("Memory Coalescing Benchmark\n");
    printf("Array size: %d floats (%.0f MiB)\n\n", n, n * 4.0 / (1 << 20));
    printf("  stride | bandwidth (GB/s)\n");
    printf("  %s\n", "-------------------------------");

    int strides[] = {1, 2, 4, 8, 16, 32};
    for (int i = 0; i < 6; ++i)
        benchmark_stride(strides[i], d_in, d_out, n, reps);

    printf("\nKey insight: stride=1 (coalesced) achieves near-peak bandwidth.\n");
    printf("Each 2x increase in stride roughly halves effective bandwidth\n");
    printf("because the same number of cache lines are fetched but only\n");
    printf("half the data in each line is actually used.\n");

    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}

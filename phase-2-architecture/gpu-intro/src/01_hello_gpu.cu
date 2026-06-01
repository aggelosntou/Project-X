/*
 * 01_hello_gpu.cu
 *
 * Learning goal: understand the CUDA device model.
 * - Query cudaGetDeviceProperties to see what the GPU looks like
 * - Launch a small kernel so every thread prints its global ID
 *
 * Compile:  nvcc -O2 -arch=sm_70 -o 01_hello_gpu src/01_hello_gpu.cu
 * Run:      ./01_hello_gpu
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

/* ---------------------------------------------------------------
 * Utility: check CUDA errors
 * --------------------------------------------------------------- */
#define CUDA_CHECK(call)                                                 \
    do {                                                                 \
        cudaError_t _e = (call);                                         \
        if (_e != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error %s:%d — %s\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(_e));          \
            exit(EXIT_FAILURE);                                          \
        }                                                                \
    } while (0)

/* ---------------------------------------------------------------
 * Kernel: each thread announces its identity.
 *
 * Thread hierarchy reminders:
 *   threadIdx.x  — thread index within its block  (0 … blockDim.x-1)
 *   blockIdx.x   — block index within the grid    (0 … gridDim.x-1)
 *   blockDim.x   — number of threads per block
 *
 * Global thread ID = blockIdx.x * blockDim.x + threadIdx.x
 * --------------------------------------------------------------- */
__global__ void hello_kernel(void)
{
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    printf("  Hello from block %d, thread %d  →  global id = %d\n",
           blockIdx.x, threadIdx.x, global_id);
}

/* ---------------------------------------------------------------
 * Host: query device properties then launch the kernel
 * --------------------------------------------------------------- */
int main(void)
{
    /* --- 1. Count available GPUs -------------------------------- */
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    printf("GPUs found: %d\n\n", device_count);

    /* --- 2. Print properties for every GPU --------------------- */
    for (int dev = 0; dev < device_count; ++dev) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

        printf("=== Device %d: %s ===\n", dev, prop.name);
        printf("  Compute capability       : %d.%d\n",
               prop.major, prop.minor);
        printf("  Streaming Multiprocessors: %d\n",
               prop.multiProcessorCount);
        printf("  Global memory            : %.1f GiB\n",
               (double)prop.totalGlobalMem / (1 << 30));
        printf("  Shared memory per block  : %zu KiB\n",
               prop.sharedMemPerBlock / 1024);
        printf("  Warp size                : %d threads\n",
               prop.warpSize);
        printf("  Max threads per block    : %d\n",
               prop.maxThreadsPerBlock);
        printf("  Max threads per SM       : %d\n",
               prop.maxThreadsPerMultiProcessor);
        printf("  L2 cache size            : %d KiB\n",
               prop.l2CacheSize / 1024);
        printf("  Memory bus width         : %d bits\n",
               prop.memoryBusWidth);
        printf("  Memory clock rate        : %.1f GHz\n",
               prop.memoryClockRate * 1e-6);
        printf("  Peak memory bandwidth    : %.1f GB/s\n",
               2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) * 1e-6);
        printf("\n");
    }

    /* --- 3. Launch hello kernel: 4 blocks × 32 threads = 128 threads */
    int blocks  = 4;
    int threads = 32;
    printf("Launching hello_kernel<<<%d, %d>>> (%d total threads)\n\n",
           blocks, threads, blocks * threads);

    hello_kernel<<<blocks, threads>>>();

    /* Wait for all GPU threads to finish before we read stdout */
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("\nDone.\n");
    return 0;
}

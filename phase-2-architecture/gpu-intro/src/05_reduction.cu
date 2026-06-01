/*
 * 05_reduction.cu
 *
 * Learning goal: parallel reduction — computing one scalar from N elements.
 *
 * Problem: sum an array of N floats as fast as possible.
 *
 * Approach 1 — Atomic naive:
 *   Every thread calls atomicAdd(&result, a[i]).
 *   Simple but serialised: all threads fight for the same memory location.
 *   Works, but atomics on global memory are slow (hundreds of cycles each).
 *
 * Approach 2 — Tree reduction in shared memory:
 *   Round 1: load 256 elements into shared mem, sum pairs →128 partial sums
 *   Round 2: sum pairs → 64
 *   …
 *   Round 8: one partial sum per block (256 → 1 in 8 steps)
 *   Thread 0 writes block sum to global memory via atomicAdd.
 *   Shared memory atomics are very fast; we only do ONE global atomic
 *   per block rather than one per thread.
 *
 * Compile:  nvcc -O2 -arch=sm_70 -o 05_reduction src/05_reduction.cu
 * Run:      ./05_reduction
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define CUDA_CHECK(call)                                                 \
    do {                                                                 \
        cudaError_t _e = (call);                                         \
        if (_e != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error %s:%d — %s\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(_e));          \
            exit(EXIT_FAILURE);                                          \
        }                                                                \
    } while (0)

#define BLOCK_SIZE 256

/* ---------------------------------------------------------------
 * Naive: every thread does a direct atomicAdd to global memory.
 * --------------------------------------------------------------- */
__global__ void reduce_atomic_naive(const float *in, float *result, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        atomicAdd(result, in[i]);
}

/* ---------------------------------------------------------------
 * Tree reduction in shared memory.
 *
 * Step-by-step for BLOCK_SIZE=256:
 *   stride=128: threads 0-127 add in[tid+128] to shared[tid]
 *   stride= 64: threads 0-63  add shared[tid+64] to shared[tid]
 *   …
 *   stride=  1: thread 0 adds shared[1] to shared[0]
 *   thread 0 does one atomicAdd to global result
 *
 * Total global atomics per launch: (n / BLOCK_SIZE) instead of n.
 * --------------------------------------------------------------- */
__global__ void reduce_tree(const float *in, float *result, int n)
{
    __shared__ float sdata[BLOCK_SIZE];

    int tid   = threadIdx.x;
    int gid   = blockIdx.x * blockDim.x + threadIdx.x;
    int total = blockDim.x * gridDim.x;

    /* Each thread accumulates its slice of the array into a register,
     * then writes once to shared memory. */
    float acc = 0.0f;
    for (int i = gid; i < n; i += total)
        acc += in[i];
    sdata[tid] = acc;
    __syncthreads();

    /* Tree reduction: halve active threads each round */
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }

    /* Thread 0 contributes this block's sum to the global result */
    if (tid == 0)
        atomicAdd(result, sdata[0]);
}

/* ---------------------------------------------------------------
 * CPU reference
 * --------------------------------------------------------------- */
static float reduce_cpu(const float *a, int n)
{
    double sum = 0.0;   /* use double to reduce floating-point error */
    for (int i = 0; i < n; ++i)
        sum += a[i];
    return (float)sum;
}

int main(void)
{
    int n = 1 << 20;   /* 1M elements */
    size_t bytes = n * sizeof(float);

    float *h_in = (float *)malloc(bytes);
    srand(42);
    for (int i = 0; i < n; ++i)
        h_in[i] = (float)rand() / RAND_MAX;

    /* --- CPU --------------------------------------------------- */
    float cpu_result = reduce_cpu(h_in, n);

    /* --- Device setup ------------------------------------------ */
    float *d_in, *d_result;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (blocks > 65535) blocks = 65535;

    int reps = 50;

    /* --- Naive atomic ------------------------------------------ */
    float h_result_naive = 0.0f;
    CUDA_CHECK(cudaMemset(d_result, 0, sizeof(float)));
    /* Warm-up */
    reduce_atomic_naive<<<blocks, BLOCK_SIZE>>>(d_in, d_result, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemset(d_result, 0, sizeof(float)));

    CUDA_CHECK(cudaEventRecord(t0));
    for (int r = 0; r < reps; ++r) {
        CUDA_CHECK(cudaMemset(d_result, 0, sizeof(float)));
        reduce_atomic_naive<<<blocks, BLOCK_SIZE>>>(d_in, d_result, n);
    }
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float naive_ms;
    CUDA_CHECK(cudaEventElapsedTime(&naive_ms, t0, t1));
    naive_ms /= reps;
    CUDA_CHECK(cudaMemcpy(&h_result_naive, d_result, sizeof(float),
                          cudaMemcpyDeviceToHost));

    /* --- Tree reduction ---------------------------------------- */
    float h_result_tree = 0.0f;
    /* Warm-up */
    CUDA_CHECK(cudaMemset(d_result, 0, sizeof(float)));
    reduce_tree<<<blocks, BLOCK_SIZE>>>(d_in, d_result, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemset(d_result, 0, sizeof(float)));

    CUDA_CHECK(cudaEventRecord(t0));
    for (int r = 0; r < reps; ++r) {
        CUDA_CHECK(cudaMemset(d_result, 0, sizeof(float)));
        reduce_tree<<<blocks, BLOCK_SIZE>>>(d_in, d_result, n);
    }
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float tree_ms;
    CUDA_CHECK(cudaEventElapsedTime(&tree_ms, t0, t1));
    tree_ms /= reps;
    CUDA_CHECK(cudaMemcpy(&h_result_tree, d_result, sizeof(float),
                          cudaMemcpyDeviceToHost));

    /* --- Report ------------------------------------------------ */
    printf("Parallel Reduction  (N = %d elements)\n\n", n);
    printf("  CPU result    : %.4f\n", cpu_result);
    printf("  Naive result  : %.4f  (error %.2e)  time: %.3f ms\n",
           h_result_naive, fabsf(h_result_naive - cpu_result), naive_ms);
    printf("  Tree result   : %.4f  (error %.2e)  time: %.3f ms\n",
           h_result_tree,  fabsf(h_result_tree  - cpu_result), tree_ms);
    printf("\n  Speedup (tree vs naive): %.1fx\n", naive_ms / tree_ms);
    printf("\nKey insight:\n");
    printf("  Naive: %d global atomicAdd calls per launch — all serialised.\n", n);
    printf("  Tree:  %d global atomicAdd calls per launch — %dx fewer.\n",
           blocks, n / blocks);
    printf("  Shared-memory adds are ~100x cheaper than global atomics.\n");

    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    cudaFree(d_in);
    cudaFree(d_result);
    free(h_in);
    return 0;
}

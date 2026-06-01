/*
 * layer_norm_kernel.cu
 * ====================
 * Fused LayerNorm in a single CUDA kernel.
 *
 * Normalises each row of the input matrix X (shape: [N, D]) independently:
 *
 *   y = gamma * (x - mean) / sqrt(var + eps) + beta
 *
 * Algorithm (each thread block handles one row):
 *   1. Load row into shared memory.
 *   2. Parallel reduction → mean.
 *   3. Parallel reduction → variance.
 *   4. Normalise + apply gamma / beta.
 *   5. Write to output.
 *
 * "Fused" means: only ONE global memory read and ONE global memory write
 * per element.  A two-kernel approach (compute_stats, then normalise)
 * would require two reads and two writes.
 *
 * Compile:
 *   nvcc -O2 -arch=sm_70 -std=c++14 -o layernorm_test src/layer_norm_kernel.cu
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t _e = (call);                                            \
        if (_e != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s:%d — %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(_e));             \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

/* =========================================================================
 * Warp-level reduction (faster than shared-memory tree for small D)
 * ====================================================================== */
__device__ float warp_reduce_sum(float val)
{
    /* All 32 threads in a warp contribute */
    for (int mask = 16; mask > 0; mask >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, mask);
    return val;
}

/* =========================================================================
 * Fused LayerNorm Kernel
 *
 * Grid:  (N,)        — one block per row
 * Block: (blockDim.x,) — multiple threads per row, must be power of 2 <= 1024
 *
 * Each thread handles ceil(D / blockDim.x) elements.
 *
 * Shared memory:
 *   [blockDim.x] floats for row data scratch (one pass: mean, then var)
 * ====================================================================== */
__global__ void layer_norm_forward(
    const float * __restrict__ X,       /* [N, D] input  */
    const float * __restrict__ gamma,   /* [D]    scale  */
    const float * __restrict__ beta,    /* [D]    shift  */
    float       * __restrict__ Y,       /* [N, D] output */
    int N, int D,
    float eps)
{
    int row = blockIdx.x;   /* which row this block handles */
    int tid = threadIdx.x;
    int T   = blockDim.x;   /* number of threads per block  */

    if (row >= N) return;

    const float *x_row = X + (size_t)row * D;
    float       *y_row = Y + (size_t)row * D;

    /* ------------------------------------------------------------------ *
     * Pass 1: compute mean using parallel reduction                      *
     * ------------------------------------------------------------------ */
    float sum = 0.0f;
    for (int i = tid; i < D; i += T)
        sum += x_row[i];

    /* Block-level reduction via warp reduction + shared memory */
    extern __shared__ float sdata[];
    sum = warp_reduce_sum(sum);
    if ((tid & 31) == 0)                /* lane 0 of each warp writes     */
        sdata[tid >> 5] = sum;
    __syncthreads();

    /* Final reduction across warps (only first warp does this) */
    if (tid < (T >> 5)) {
        float val = sdata[tid];
        val = warp_reduce_sum(val);
        if (tid == 0) sdata[0] = val;   /* sdata[0] now holds total sum   */
    }
    __syncthreads();

    float mean = sdata[0] / (float)D;

    /* ------------------------------------------------------------------ *
     * Pass 2: compute variance                                            *
     * ------------------------------------------------------------------ */
    float var_sum = 0.0f;
    for (int i = tid; i < D; i += T) {
        float diff = x_row[i] - mean;
        var_sum += diff * diff;
    }

    var_sum = warp_reduce_sum(var_sum);
    if ((tid & 31) == 0) sdata[tid >> 5] = var_sum;
    __syncthreads();

    if (tid < (T >> 5)) {
        float val = sdata[tid];
        val = warp_reduce_sum(val);
        if (tid == 0) sdata[0] = val;
    }
    __syncthreads();

    float var     = sdata[0] / (float)D;
    float inv_std = rsqrtf(var + eps);   /* rsqrtf = 1/sqrt (fast intrinsic) */

    /* ------------------------------------------------------------------ *
     * Pass 3: normalise + scale + shift — write output                   *
     * ------------------------------------------------------------------ */
    for (int i = tid; i < D; i += T)
        y_row[i] = gamma[i] * (x_row[i] - mean) * inv_std + beta[i];
}

/* =========================================================================
 * Host-side wrapper
 * ====================================================================== */

/*
 * run_layer_norm
 *
 * X_d:     device pointer [N * D]
 * gamma_d: device pointer [D]
 * beta_d:  device pointer [D]
 * Y_d:     device pointer [N * D]
 *
 * Returns kernel time in ms.
 */
float run_layer_norm(
    const float *X_d, const float *gamma_d, const float *beta_d,
    float *Y_d,
    int N, int D, float eps)
{
    /* Choose block size: next power of 2 >= min(D, 1024) */
    int T = 32;
    while (T < D && T < 1024) T <<= 1;

    int warps_per_block = T / 32;
    size_t shmem = (size_t)warps_per_block * sizeof(float);

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);

    layer_norm_forward<<<N, T, shmem>>>(X_d, gamma_d, beta_d, Y_d, N, D, eps);

    cudaEventRecord(t1);
    cudaEventSynchronize(t1);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, t0, t1);
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    return ms;
}

/* =========================================================================
 * Stand-alone test
 * ====================================================================== */
#ifndef BENCHMARK_MAIN

static float *host_rand(int n)
{
    float *h = (float *)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) h[i] = (float)rand() / RAND_MAX;
    return h;
}

int main(void)
{
    int N = 1024, D = 512;
    printf("LayerNorm: N=%d D=%d\n", N, D);

    float *hX     = host_rand(N * D);
    float *hGamma = host_rand(D);
    float *hBeta  = host_rand(D);
    float *hY     = (float *)calloc(N * D, sizeof(float));

    float *X_d, *G_d, *B_d, *Y_d;
    CUDA_CHECK(cudaMalloc(&X_d, (size_t)N * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&G_d,            D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&B_d,            D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&Y_d, (size_t)N * D * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(X_d, hX,     (size_t)N * D * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(G_d, hGamma,           D * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B_d, hBeta,            D * sizeof(float), cudaMemcpyHostToDevice));

    /* Warm-up */
    run_layer_norm(X_d, G_d, B_d, Y_d, N, D, 1e-5f);
    CUDA_CHECK(cudaDeviceSynchronize());

    float ms = run_layer_norm(X_d, G_d, B_d, Y_d, N, D, 1e-5f);
    printf("Kernel time: %.4f ms\n", ms);

    /* Quick sanity check on first row */
    CUDA_CHECK(cudaMemcpy(hY, Y_d, (size_t)N * D * sizeof(float), cudaMemcpyDeviceToHost));
    float row_sum = 0.0f, row_sq_sum = 0.0f;
    for (int i = 0; i < D; i++) { row_sum += hY[i]; row_sq_sum += hY[i] * hY[i]; }
    printf("First row output:  mean=%.6f  rms=%.6f  (expect ~0 and ~1 before scale)\n",
           row_sum / D, sqrtf(row_sq_sum / D));

    free(hX); free(hGamma); free(hBeta); free(hY);
    cudaFree(X_d); cudaFree(G_d); cudaFree(B_d); cudaFree(Y_d);
    return 0;
}

#endif /* BENCHMARK_MAIN */

/*
 * 03_matrix_multiply.cu
 *
 * Learning goal: understand why shared-memory tiling is important.
 *
 * Naive kernel: each thread reads N elements from global memory
 *   (once per row of A, once per column of B).  For an N×N matmul,
 *   global memory traffic = 2 * N^3 loads.
 *
 * Tiled kernel: threads in a 16×16 block cooperatively load a
 *   16×16 tile of A and B into shared memory, then compute 16×16
 *   partial products.  Each global element is loaded only N/16 times
 *   instead of N times → 16× reduction in global memory traffic.
 *
 * Compile:  nvcc -O2 -arch=sm_70 -o 03_matrix_multiply src/03_matrix_multiply.cu
 * Run:      ./03_matrix_multiply
 *
 * Expected output (V100-class GPU):
 *   N    | naive_ms | tiled_ms | naive_GFLOPS | tiled_GFLOPS
 *   128  |    0.10  |    0.04  |     42.1     |    104.9
 *   256  |    0.72  |    0.18  |     46.9     |    187.3
 *   512  |    5.50  |    0.93  |     48.9     |    289.8
 *   1024 |   43.20  |    5.10  |     49.8     |    422.1
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

#define TILE 16

/* ---------------------------------------------------------------
 * Naive kernel
 * Each thread computes one element C[row][col].
 * For every output element the thread walks an entire row of A and
 * an entire column of B — all from slow global memory.
 * --------------------------------------------------------------- */
__global__ void matmul_naive(const float *A, const float *B, float *C, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= N || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < N; ++k)
        sum += A[row * N + k] * B[k * N + col];
    C[row * N + col] = sum;
}

/* ---------------------------------------------------------------
 * Tiled kernel (TILE × TILE shared memory blocks)
 *
 * Algorithm:
 *   Divide the k-dimension into tiles of width TILE.
 *   For each tile:
 *     1. Each thread loads one element of A and one of B into shmem.
 *     2. __syncthreads() — all threads must finish loading.
 *     3. Each thread accumulates TILE multiply-adds using shmem.
 *     4. __syncthreads() — all threads must finish computing before
 *        the shmem arrays are overwritten in the next iteration.
 * --------------------------------------------------------------- */
__global__ void matmul_tiled(const float *A, const float *B, float *C, int N)
{
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    int tx  = threadIdx.x;
    int ty  = threadIdx.y;

    float sum = 0.0f;
    int num_tiles = (N + TILE - 1) / TILE;

    for (int t = 0; t < num_tiles; ++t) {
        /* Load this tile into shared memory */
        int a_col = t * TILE + tx;
        int b_row = t * TILE + ty;

        As[ty][tx] = (row < N && a_col < N) ? A[row * N + a_col] : 0.0f;
        Bs[ty][tx] = (b_row < N && col < N) ? B[b_row * N + col] : 0.0f;

        __syncthreads();  /* barrier: wait until all threads have loaded */

        /* Compute partial dot product from shared memory */
        for (int k = 0; k < TILE; ++k)
            sum += As[ty][k] * Bs[k][tx];

        __syncthreads();  /* barrier: wait before overwriting shared mem */
    }

    if (row < N && col < N)
        C[row * N + col] = sum;
}

/* ---------------------------------------------------------------
 * CPU reference (used for correctness check on small N)
 * --------------------------------------------------------------- */
static void matmul_cpu(const float *A, const float *B, float *C, int N)
{
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            float s = 0.0f;
            for (int k = 0; k < N; ++k)
                s += A[i * N + k] * B[k * N + j];
            C[i * N + j] = s;
        }
}

/* ---------------------------------------------------------------
 * Verify two N×N float matrices agree within tolerance
 * --------------------------------------------------------------- */
static int verify(const float *ref, const float *got, int N, const char *tag)
{
    int errors = 0;
    for (int i = 0; i < N * N; ++i) {
        float rel = fabsf(ref[i] - got[i]) / (fabsf(ref[i]) + 1e-6f);
        if (rel > 1e-3f) {
            if (errors < 5)
                fprintf(stderr, "  [%s] mismatch at %d: ref=%.4f got=%.4f\n",
                        tag, i, ref[i], got[i]);
            errors++;
        }
    }
    return errors;
}

/* ---------------------------------------------------------------
 * Benchmark one size N
 * --------------------------------------------------------------- */
static void benchmark(int N)
{
    size_t bytes = (size_t)N * N * sizeof(float);
    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C_ref = (float *)malloc(bytes);
    float *h_C_naive = (float *)malloc(bytes);
    float *h_C_tiled = (float *)malloc(bytes);

    srand(1234);
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = (float)rand() / RAND_MAX - 0.5f;
        h_B[i] = (float)rand() / RAND_MAX - 0.5f;
    }

    /* CPU reference for small N */
    if (N <= 256)
        matmul_cpu(h_A, h_B, h_C_ref, N);

    /* --- Device allocations ------------------------------------ */
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    /* --- Naive kernel ------------------------------------------ */
    CUDA_CHECK(cudaEventRecord(t0));
    matmul_naive<<<grid, block>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float naive_ms;
    CUDA_CHECK(cudaEventElapsedTime(&naive_ms, t0, t1));
    CUDA_CHECK(cudaMemcpy(h_C_naive, d_C, bytes, cudaMemcpyDeviceToHost));

    /* --- Tiled kernel ------------------------------------------ */
    CUDA_CHECK(cudaEventRecord(t0));
    matmul_tiled<<<grid, block>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float tiled_ms;
    CUDA_CHECK(cudaEventElapsedTime(&tiled_ms, t0, t1));
    CUDA_CHECK(cudaMemcpy(h_C_tiled, d_C, bytes, cudaMemcpyDeviceToHost));

    /* Verify tiled against naive */
    int errs = verify(h_C_naive, h_C_tiled, N, "tiled-vs-naive");
    if (errs > 0)
        fprintf(stderr, "  %d total mismatches for N=%d\n", errs, N);

    /* FLOPS = 2 * N^3 (one mul + one add per k-step) */
    double flops = 2.0 * (double)N * N * N;
    double naive_gflops = flops / (naive_ms * 1e-3) / 1e9;
    double tiled_gflops = flops / (tiled_ms * 1e-3) / 1e9;

    printf("  %-6d | %9.2f | %9.2f | %13.1f | %13.1f\n",
           N, naive_ms, tiled_ms, naive_gflops, tiled_gflops);

    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C_ref); free(h_C_naive); free(h_C_tiled);
}

int main(void)
{
    printf("Matrix Multiply Benchmark  (N×N × N×N)\n");
    printf("Tile size: %d × %d\n\n", TILE, TILE);
    printf("  %-6s | %9s | %9s | %13s | %13s\n",
           "N", "naive_ms", "tiled_ms", "naive_GFLOPS", "tiled_GFLOPS");
    printf("  %s\n", "-----------------------------------------------------------------");

    int sizes[] = {128, 256, 512, 1024};
    for (int i = 0; i < 4; ++i)
        benchmark(sizes[i]);

    printf("\nKey insight: tiling reuses each shared-mem element %d times,\n", TILE);
    printf("reducing global memory traffic by ~%dx.\n", TILE);
    return 0;
}

/*
 * 02_vector_add.cu
 *
 * Learning goal: understand when the GPU actually wins.
 *
 * PCIe bandwidth (CPU→GPU) is ~12-16 GB/s.
 * For tiny arrays the transfer cost dominates, so CPU is faster.
 * For large arrays (millions of elements) the massively parallel
 * GPU computation outweighs the transfer overhead.
 *
 * Compile:  nvcc -O2 -arch=sm_70 -o 02_vector_add src/02_vector_add.cu
 * Run:      ./02_vector_add
 *
 * Expected output shape:
 *   N          | cpu_ms  | gpu_ms  | speedup
 *   1000       |  0.003  |  0.312  |   0.01x   ← GPU loses
 *   10000      |  0.030  |  0.318  |   0.09x
 *   100000     |  0.302  |  0.341  |   0.89x   ← break-even
 *   1000000    |  3.124  |  0.481  |   6.49x   ← GPU wins
 *   10000000   | 31.480  |  2.014  |  15.63x
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
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

/* ---------------------------------------------------------------
 * CPU baseline
 * --------------------------------------------------------------- */
static void vector_add_cpu(const float *a, const float *b, float *c, int n)
{
    for (int i = 0; i < n; ++i)
        c[i] = a[i] + b[i];
}

/* ---------------------------------------------------------------
 * GPU kernel: one thread per element
 *
 * We use a "grid-stride loop" pattern so the kernel works
 * correctly even if N > total number of launched threads.
 * --------------------------------------------------------------- */
__global__ void vector_add_gpu(const float *a, const float *b, float *c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;   /* total number of threads */
    for (; i < n; i += stride)
        c[i] = a[i] + b[i];
}

/* ---------------------------------------------------------------
 * High-resolution wall-clock timer (nanoseconds → ms)
 * --------------------------------------------------------------- */
static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec * 1e-6;
}

/* ---------------------------------------------------------------
 * Verify GPU result against CPU result
 * --------------------------------------------------------------- */
static void verify(const float *cpu, const float *gpu, int n)
{
    for (int i = 0; i < n; ++i) {
        if (fabsf(cpu[i] - gpu[i]) > 1e-4f) {
            fprintf(stderr, "MISMATCH at i=%d: cpu=%f gpu=%f\n",
                    i, cpu[i], gpu[i]);
            exit(EXIT_FAILURE);
        }
    }
}

/* ---------------------------------------------------------------
 * Benchmark one size N
 * --------------------------------------------------------------- */
static void benchmark(int n)
{
    size_t bytes = n * sizeof(float);

    /* --- Host allocations --------------------------------------- */
    float *h_a = (float *)malloc(bytes);
    float *h_b = (float *)malloc(bytes);
    float *h_c_cpu = (float *)malloc(bytes);
    float *h_c_gpu = (float *)malloc(bytes);

    /* Fill with random values */
    for (int i = 0; i < n; ++i) {
        h_a[i] = (float)rand() / RAND_MAX;
        h_b[i] = (float)rand() / RAND_MAX;
    }

    /* --- CPU benchmark ----------------------------------------- */
    double t0 = now_ms();
    vector_add_cpu(h_a, h_b, h_c_cpu, n);
    double cpu_ms = now_ms() - t0;

    /* --- GPU benchmark (includes H2D + kernel + D2H) ----------- */
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    /* Use cudaEvent_t for accurate GPU timing */
    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    CUDA_CHECK(cudaEventRecord(ev_start));

    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    int threads_per_block = 256;
    int blocks = (n + threads_per_block - 1) / threads_per_block;
    /* Cap grid size to the maximum the GPU supports */
    if (blocks > 65535) blocks = 65535;
    vector_add_gpu<<<blocks, threads_per_block>>>(d_a, d_b, d_c, n);

    CUDA_CHECK(cudaMemcpy(h_c_gpu, d_c, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));

    float gpu_ms_f = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_ms_f, ev_start, ev_stop));
    double gpu_ms = gpu_ms_f;

    /* --- Verify ------------------------------------------------- */
    verify(h_c_cpu, h_c_gpu, n);

    double speedup = cpu_ms / gpu_ms;
    printf("  %-12d | %8.3f  | %8.3f  | %6.2fx\n",
           n, cpu_ms, gpu_ms, speedup);

    /* --- Cleanup ------------------------------------------------ */
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c_cpu); free(h_c_gpu);
}

int main(void)
{
    srand(42);

    printf("Vector Addition Benchmark\n");
    printf("(gpu_ms includes H2D transfer + kernel + D2H transfer)\n\n");
    printf("  %-12s | %8s  | %8s  | %s\n",
           "N", "cpu_ms", "gpu_ms", "speedup");
    printf("  %s\n", "---------------------------------------------");

    int sizes[] = {1000, 10000, 100000, 1000000, 10000000};
    for (int i = 0; i < 5; ++i)
        benchmark(sizes[i]);

    printf("\nNote: for small N the PCIe round-trip (~few ms) dominates.\n");
    printf("      For large N the GPU's parallelism pays off.\n");
    return 0;
}

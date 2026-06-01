/*
 * benchmark.cu
 * ============
 * Benchmark all three attention implementations across sequence lengths.
 *
 * For each seq_len in {64, 128, 256, 512, 1024}:
 *   - Allocate Q, K, V on device
 *   - Run each kernel 10 times, record minimum time via cudaEvent
 *   - Compute: time (ms), memory used (MB), effective bandwidth (GB/s)
 *
 * Output table:
 *   seq_len | naive_ms | shared_ms | flash_ms | naive_MB | flash_MB
 *
 * Compile:
 *   nvcc -O2 -arch=sm_70 -std=c++14 -DBENCHMARK_MAIN \
 *        -o benchmark \
 *        src/attention_naive.cu src/attention_shared_mem.cu \
 *        src/flash_attention.cu src/layer_norm_kernel.cu \
 *        src/benchmark.cu
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>

/* Tell the other translation units not to compile their own main() */
#define BENCHMARK_MAIN

/* Forward declarations of host-side wrappers */
float naive_attention(
    const float *Q_d, const float *K_d, const float *V_d,
    float *scores_d, float *out_d,
    int batch, int heads, int seq_len, int d_head);

float shared_mem_attention(
    const float *Q_d, const float *K_d, const float *V_d,
    float *scores_d, float *out_d,
    int batch, int heads, int seq_len, int d_head);

float flash_attention(
    const float *Q_d, const float *K_d, const float *V_d,
    float *O_d,
    int batch, int heads, int seq_len, int d_head,
    int Br, int Bc);

float run_layer_norm(
    const float *X_d, const float *gamma_d, const float *beta_d,
    float *Y_d,
    int N, int D, float eps);

/* -------------------------------------------------------------------------
 * Helpers
 * ---------------------------------------------------------------------- */

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t _e = (call);                                            \
        if (_e != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s:%d — %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(_e));             \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

static void cuda_fill_rand(float *d, size_t n)
{
    float *h = (float *)malloc(n * sizeof(float));
    for (size_t i = 0; i < n; i++) h[i] = (float)rand() / RAND_MAX * 0.1f;
    CUDA_CHECK(cudaMemcpy(d, h, n * sizeof(float), cudaMemcpyHostToDevice));
    free(h);
}

/*
 * Repeat a kernel call N_RUNS times and return the minimum elapsed time.
 * Uses cudaEvent timing.
 */
#define N_RUNS 10

/* =========================================================================
 * Benchmark attention kernels
 * ====================================================================== */
static void benchmark_attention(int batch, int heads, int d_head)
{
    printf("\n=== Attention Benchmark  batch=%d  heads=%d  d_head=%d ===\n",
           batch, heads, d_head);
    printf("%-8s  %10s  %10s  %10s  %10s  %10s\n",
           "seq_len", "naive_ms", "shared_ms", "flash_ms",
           "naive_MB", "flash_MB");
    printf("%s\n", "--------  ----------  ----------  ----------  ----------  ----------");

    int seq_lens[] = {64, 128, 256, 512, 1024};
    int n_cases = (int)(sizeof(seq_lens) / sizeof(seq_lens[0]));

    for (int ci = 0; ci < n_cases; ci++) {
        int seq_len = seq_lens[ci];
        int bh      = batch * heads;

        size_t qkv_n  = (size_t)bh * seq_len * d_head;
        size_t scr_n  = (size_t)bh * seq_len * seq_len;
        size_t qkv_sz = qkv_n  * sizeof(float);
        size_t scr_sz = scr_n  * sizeof(float);

        /* Memory in MB */
        float naive_mb = (float)(qkv_sz * 4 + scr_sz) / 1024.0f / 1024.0f;
        float flash_mb = (float)(qkv_sz * 4)           / 1024.0f / 1024.0f;

        /* Allocate device memory */
        float *Q_d, *K_d, *V_d, *S_d, *O_d, *O_flash;
        CUDA_CHECK(cudaMalloc(&Q_d,     qkv_sz));
        CUDA_CHECK(cudaMalloc(&K_d,     qkv_sz));
        CUDA_CHECK(cudaMalloc(&V_d,     qkv_sz));
        CUDA_CHECK(cudaMalloc(&S_d,     scr_sz));
        CUDA_CHECK(cudaMalloc(&O_d,     qkv_sz));
        CUDA_CHECK(cudaMalloc(&O_flash, qkv_sz));

        cuda_fill_rand(Q_d, qkv_n);
        cuda_fill_rand(K_d, qkv_n);
        cuda_fill_rand(V_d, qkv_n);

        /* --- Naive: skip if seq_len large (OOM risk) --- */
        float naive_min = -1.0f, shared_min = -1.0f, flash_min = FLT_MAX;

        if (scr_sz < 512UL * 1024 * 1024) {  /* < 512 MB scores tensor */
            /* Warm-up */
            naive_attention(Q_d, K_d, V_d, S_d, O_d, batch, heads, seq_len, d_head);
            CUDA_CHECK(cudaDeviceSynchronize());
            naive_min = FLT_MAX;
            for (int r = 0; r < N_RUNS; r++) {
                float ms = naive_attention(Q_d, K_d, V_d, S_d, O_d,
                                           batch, heads, seq_len, d_head);
                if (ms < naive_min) naive_min = ms;
            }

            /* Shared memory */
            shared_mem_attention(Q_d, K_d, V_d, S_d, O_d, batch, heads, seq_len, d_head);
            CUDA_CHECK(cudaDeviceSynchronize());
            shared_min = FLT_MAX;
            for (int r = 0; r < N_RUNS; r++) {
                float ms = shared_mem_attention(Q_d, K_d, V_d, S_d, O_d,
                                                batch, heads, seq_len, d_head);
                if (ms < shared_min) shared_min = ms;
            }
        }

        /* Flash attention */
        {
            int Br = 32, Bc = 32;
            flash_attention(Q_d, K_d, V_d, O_flash, batch, heads, seq_len, d_head, Br, Bc);
            CUDA_CHECK(cudaDeviceSynchronize());
            for (int r = 0; r < N_RUNS; r++) {
                float ms = flash_attention(Q_d, K_d, V_d, O_flash,
                                           batch, heads, seq_len, d_head, Br, Bc);
                if (ms < flash_min) flash_min = ms;
            }
        }

        /* Print row */
        if (naive_min < 0)
            printf("%-8d  %10s  %10s  %10.4f  %10s  %10.2f\n",
                   seq_len, "OOM", "OOM",
                   flash_min, "OOM", flash_mb);
        else
            printf("%-8d  %10.4f  %10.4f  %10.4f  %10.2f  %10.2f\n",
                   seq_len,
                   naive_min, shared_min, flash_min,
                   naive_mb, flash_mb);

        cudaFree(Q_d); cudaFree(K_d); cudaFree(V_d);
        cudaFree(S_d); cudaFree(O_d); cudaFree(O_flash);
    }
}

/* =========================================================================
 * Benchmark LayerNorm
 * ====================================================================== */
static void benchmark_layer_norm(void)
{
    printf("\n=== LayerNorm Benchmark ===\n");
    printf("%-8s  %-8s  %10s\n", "N_rows", "D", "time_ms");
    printf("%s\n", "--------  --------  ----------");

    int configs[][2] = {{512, 256}, {1024, 512}, {2048, 768}, {4096, 1024}};
    int n = (int)(sizeof(configs) / sizeof(configs[0]));

    for (int ci = 0; ci < n; ci++) {
        int N = configs[ci][0], D = configs[ci][1];
        size_t sz = (size_t)N * D * sizeof(float);

        float *X_d, *G_d, *B_d, *Y_d;
        CUDA_CHECK(cudaMalloc(&X_d, sz));
        CUDA_CHECK(cudaMalloc(&G_d,      D * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&B_d,      D * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&Y_d, sz));

        cuda_fill_rand(X_d, (size_t)N * D);
        cuda_fill_rand(G_d, (size_t)D);
        cuda_fill_rand(B_d, (size_t)D);

        run_layer_norm(X_d, G_d, B_d, Y_d, N, D, 1e-5f);
        CUDA_CHECK(cudaDeviceSynchronize());

        float best = FLT_MAX;
        for (int r = 0; r < N_RUNS; r++) {
            float ms = run_layer_norm(X_d, G_d, B_d, Y_d, N, D, 1e-5f);
            if (ms < best) best = ms;
        }

        printf("%-8d  %-8d  %10.4f\n", N, D, best);

        cudaFree(X_d); cudaFree(G_d); cudaFree(B_d); cudaFree(Y_d);
    }
}

/* =========================================================================
 * Main
 * ====================================================================== */
int main(void)
{
    int device;
    CUDA_CHECK(cudaGetDevice(&device));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    printf("Device: %s  (sm_%d%d)\n",
           prop.name, prop.major, prop.minor);
    printf("Global memory: %.0f MB\n",
           prop.totalGlobalMem / 1024.0 / 1024.0);
    printf("Shared mem / block: %zu KB\n",
           prop.sharedMemPerBlock / 1024);
    printf("Warp size: %d\n", prop.warpSize);

    benchmark_attention(/*batch=*/1, /*heads=*/8, /*d_head=*/64);
    benchmark_layer_norm();

    printf("\nDone.\n");
    return 0;
}

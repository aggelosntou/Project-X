/*
 * benchmark.c — Benchmark all matrix multiply implementations
 *
 * Metrics reported:
 *   - Time (ms)
 *   - GFLOPS (billion floating point ops per second)
 *   - % of theoretical peak
 *
 * FLOP count for N×N matmul: 2×N³ (N² dot products of length N, each with N
 * multiplies and N-1 adds ≈ 2N per output element).
 *
 * Theoretical peak calculation:
 *   peak_gflops = clock_freq_ghz × cores × avx_width × fma_factor
 *   - Single core (our test)
 *   - AVX2: 8 float32 per 256-bit register
 *   - FMA: multiply + add = 2 ops per cycle
 *   - 1 AVX2 FMA unit per cycle at 3 GHz: 3 × 8 × 2 = 48 GFLOPS
 *
 * For Apple Silicon (no AVX2): scalar FMA = 3 × 1 × 2 = 6 GFLOPS peak.
 *
 * We also verify correctness by comparing output against the naive result.
 */

#include "matmul.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void fill_random(float *m, int N) {
    for (int i = 0; i < N * N; i++)
        m[i] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
}

/* Max absolute difference between two matrices */
static float max_diff(const float *A, const float *B, int N) {
    float d = 0.0f;
    for (int i = 0; i < N*N; i++) {
        float diff = fabsf(A[i] - B[i]);
        if (diff > d) d = diff;
    }
    return d;
}

typedef void (*MatmulFn)(const float*, const float*, float*, int);

/* Get theoretical peak GFLOPS for this machine */
static double theoretical_peak_gflops(void) {
#ifdef __AVX2__
    /* AVX2 FMA: 8 floats × 2 ops/cycle × 3 GHz (estimate) */
    return 48.0;
#else
    /* Scalar FMA or 2-way SIMD on Apple Silicon (NEON is available but we
     * don't use it here). Rough estimate: 2 ops/cycle × 3 GHz = 6 GFLOPS */
    return 6.0;
#endif
}

static void benchmark_one(const char *name, MatmulFn fn,
                           const float *A, const float *B,
                           float *C, float *C_ref,
                           int N, double peak_gflops) {
    int repeats = 3;

    /* Warm up */
    fn(A, B, C, N);

    double best_time = 1e9;
    for (int r = 0; r < repeats; r++) {
        double t0 = now_sec();
        fn(A, B, C, N);
        double t1 = now_sec();
        if (t1 - t0 < best_time) best_time = t1 - t0;
    }

    double flops    = 2.0 * (double)N * N * N;
    double gflops   = flops / best_time / 1e9;
    double pct_peak = gflops / peak_gflops * 100.0;

    /* Verify correctness (skip for reference) */
    float err = 0.0f;
    if (C_ref) {
        err = max_diff(C, C_ref, N);
    }

    printf("%-20s | %8.2f ms | %8.3f GFLOPS | %6.1f%% peak",
           name,
           best_time * 1000.0,
           gflops,
           pct_peak);

    if (C_ref) {
        printf(" | max_err=%.2e %s\n", (double)err, (err < 1e-2f) ? "OK" : "FAIL");
    } else {
        printf(" | (reference)\n");
    }
}

int main(void) {
    srand(42);

    double peak = theoretical_peak_gflops();

    printf("Matrix Multiply Benchmark\n");
    printf("=========================\n");
    printf("Theoretical peak: %.0f GFLOPS (single core)\n", peak);
#ifdef __AVX2__
    printf("SIMD: AVX2 (8×float32) + FMA enabled\n");
#else
    printf("SIMD: AVX2 not available (scalar fallback)\n");
#endif
    printf("\n");

    int sizes[] = {64, 128, 256, 512, 1024};
    int nsizes  = (int)(sizeof(sizes) / sizeof(sizes[0]));

    for (int si = 0; si < nsizes; si++) {
        int N = sizes[si];

        float *A     = malloc((size_t)N * N * sizeof(float));
        float *B     = malloc((size_t)N * N * sizeof(float));
        float *C     = malloc((size_t)N * N * sizeof(float));
        float *C_ref = malloc((size_t)N * N * sizeof(float));

        if (!A || !B || !C || !C_ref) { perror("malloc"); return 1; }

        fill_random(A, N);
        fill_random(B, N);

        /* Compute reference */
        matmul_naive(A, B, C_ref, N);

        printf("N = %4d  (2×N³ = %.1f MFLOPS)\n",
               N, 2.0 * N * N * N / 1e6);
        printf("%-20s | %-12s | %-18s | %-12s | %s\n",
               "Implementation", "Time (ms)", "GFLOPS", "% Peak", "Correctness");
        printf("%-20s-+-%-12s-+-%-18s-+-%-12s-+-%s\n",
               "--------------------", "------------", "------------------",
               "------------", "-----------");

        benchmark_one("naive (ikj)",      matmul_naive,       A, B, C, NULL,  N, peak);
        benchmark_one("transposed",       matmul_transposed,  A, B, C, C_ref, N, peak);
        benchmark_one("tiled (T=32)",     matmul_tiled,       A, B, C, C_ref, N, peak);
        benchmark_one("simd (AVX2/scal)", matmul_simd,        A, B, C, C_ref, N, peak);
        benchmark_one("tiled+simd",       matmul_tiled_simd,  A, B, C, C_ref, N, peak);
        printf("\n");

        free(A); free(B); free(C); free(C_ref);
    }

    return 0;
}

/*
 * benchmark.c — Driver that measures all four matmul implementations.
 *
 * OUTPUT FORMAT:
 *   Part 1: For each N in {128, 256, 512, 1024} × each implementation,
 *           print: N | impl | time_ms | GFLOPS
 *
 *   Part 2: Amdahl's Law table for N=512, varying thread count 1..max.
 *           Print: threads | time_ms | speedup | efficiency | Amdahl_limit
 *
 * GFLOPS FORMULA:
 *   A matrix multiply C = A*B for N×N matrices performs 2*N^3 floating-
 *   point operations (N^2 dot products × N multiply-adds × 2 ops each).
 *   GFLOPS = 2 * N^3 / time_seconds / 1e9
 *
 * AMDAHL'S LAW:
 *   If fraction s of the algorithm is serial (cannot be parallelised),
 *   the maximum speedup with p threads is:
 *     S(p) = 1 / (s + (1-s)/p)
 *
 *   In our matmul the serial fraction is very small (just the memset and
 *   loop bookkeeping), so we expect near-linear scaling until memory
 *   bandwidth becomes the bottleneck.
 *
 *   Efficiency = speedup / p.  When efficiency < 50%, adding more threads
 *   is counterproductive.
 */

#include "matmul.h"

#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* -------------------------------------------------------------------------
 * Timing
 * ---------------------------------------------------------------------- */

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* -------------------------------------------------------------------------
 * Matrix allocation helpers
 * ---------------------------------------------------------------------- */

static float *mat_alloc(int N)
{
    float *m = malloc((size_t)N * N * sizeof(float));
    if (!m) { fprintf(stderr, "malloc failed\n"); exit(1); }
    return m;
}

/* Fill matrix with small random floats in [0, 1). */
static void mat_rand(float *m, int N)
{
    for (int i = 0; i < N * N; i++)
        m[i] = (float)rand() / (float)RAND_MAX;
}

/* -------------------------------------------------------------------------
 * Verify two result matrices are close (sanity check)
 * ---------------------------------------------------------------------- */

static int mat_close(const float *A, const float *B, int N, float tol)
{
    for (int i = 0; i < N * N; i++) {
        if (fabsf(A[i] - B[i]) > tol * (fabsf(A[i]) + 1.0f))
            return 0;
    }
    return 1;
}

/* -------------------------------------------------------------------------
 * Run an implementation multiple times and return the minimum time (seconds)
 * ---------------------------------------------------------------------- */

#define TRIALS 3
#define TILE_SIZE 32   /* default tile size for tiled_openmp */

typedef enum { NAIVE, TRANSPOSED, OPENMP, TILED_OPENMP } ImplID;

static double measure(ImplID impl, const float *A, const float *B,
                      float *C, int N)
{
    double best = 1e18;
    for (int t = 0; t < TRIALS; t++) {
        double t0 = now_sec();
        switch (impl) {
        case NAIVE:        matmul_naive(A, B, C, N);                 break;
        case TRANSPOSED:   matmul_transposed(A, B, C, N);            break;
        case OPENMP:       matmul_openmp(A, B, C, N);                break;
        case TILED_OPENMP: matmul_tiled_openmp(A, B, C, N, TILE_SIZE); break;
        }
        double elapsed = now_sec() - t0;
        if (elapsed < best) best = elapsed;
    }
    return best;
}

static const char *impl_name(ImplID impl)
{
    switch (impl) {
    case NAIVE:        return "naive        ";
    case TRANSPOSED:   return "transposed   ";
    case OPENMP:       return "openmp       ";
    case TILED_OPENMP: return "tiled_openmp ";
    }
    return "?";
}

/* -------------------------------------------------------------------------
 * Part 1: sweep over N and implementations
 * ---------------------------------------------------------------------- */

static void part1_sweep(void)
{
    static const int Ns[] = { 128, 256, 512, 1024 };
    static const int nN   = 4;

    printf("=== Part 1: Implementation Comparison ===\n");
    printf("%-6s  %-15s  %10s  %10s\n",
           "N", "Implementation", "time (ms)", "GFLOPS");
    printf("%-6s  %-15s  %10s  %10s\n",
           "------", "---------------", "----------", "----------");

    for (int ni = 0; ni < nN; ni++) {
        int N = Ns[ni];

        float *A  = mat_alloc(N);
        float *B  = mat_alloc(N);
        float *C  = mat_alloc(N);
        float *C0 = mat_alloc(N);  /* reference result from naive */

        mat_rand(A, N);
        mat_rand(B, N);

        /* Compute reference once. */
        matmul_naive(A, B, C0, N);

        /* Skip naive for N=1024 to avoid a very long wait (optional). */
        int run_naive = (N <= 512);

        static const ImplID impls[] = { NAIVE, TRANSPOSED, OPENMP, TILED_OPENMP };
        int n_impls = 4;

        for (int ii = 0; ii < n_impls; ii++) {
            ImplID impl = impls[ii];
            if (impl == NAIVE && !run_naive) {
                printf("%-6d  %-15s  %10s  %10s  (skipped: too slow)\n",
                       N, impl_name(impl), "-", "-");
                continue;
            }

            double t = measure(impl, A, B, C, N);
            double gflops = 2.0 * (double)N * N * N / t / 1e9;
            double ms     = t * 1000.0;

            /* Verify against reference (except for the reference itself). */
            const char *ok = "";
            if (impl != NAIVE) {
                ok = mat_close(C, C0, N, 1e-3f) ? "" : " [MISMATCH!]";
            }

            printf("%-6d  %-15s  %10.2f  %10.3f%s\n",
                   N, impl_name(impl), ms, gflops, ok);
        }

        printf("\n");
        free(A); free(B); free(C); free(C0);
    }
}

/* -------------------------------------------------------------------------
 * Part 2: Amdahl's Law — vary thread count for N=512
 * ---------------------------------------------------------------------- */

static void part2_amdahl(void)
{
    const int N = 512;
    int max_threads = omp_get_max_threads();

    printf("=== Part 2: Amdahl's Law (N=%d, tiled_openmp, tile=%d) ===\n",
           N, TILE_SIZE);
    printf("%-8s  %10s  %10s  %10s  %14s\n",
           "threads", "time (ms)", "speedup", "efficiency", "Amdahl limit");
    printf("%-8s  %10s  %10s  %10s  %14s\n",
           "--------", "----------", "----------", "----------", "--------------");

    float *A = mat_alloc(N);
    float *B = mat_alloc(N);
    float *C = mat_alloc(N);
    mat_rand(A, N);
    mat_rand(B, N);

    double t1 = 0.0;  /* single-thread baseline */

    /* Estimate serial fraction s from single vs double thread timing. */
    double s_estimate = 0.0;  /* will be computed after 1 and 2 thread runs */
    double t2 = 0.0;

    int printed_efficiency_warning = 0;

    for (int p = 1; p <= max_threads; p++) {
        omp_set_num_threads(p);

        double t = measure(TILED_OPENMP, A, B, C, N);
        double ms = t * 1000.0;

        if (p == 1) {
            t1 = t;
        }
        if (p == 2) {
            t2 = t;
            /*
             * Estimate the serial fraction s using Amdahl's Law solved for s:
             *   speedup(2) = 1/(s + (1-s)/2)
             *   => s = (2 - 2*speedup(2)) / (speedup(2)*(2-2) + ... )
             * Simpler: s = (2/speedup - 1) / (2-1)  [standard formula]
             */
            double sp2 = t1 / t2;
            if (sp2 > 1.0) {
                s_estimate = (2.0 / sp2 - 1.0);  /* approximate */
                if (s_estimate < 0.0) s_estimate = 0.0;
            }
        }

        double speedup    = t1 / t;
        double efficiency = speedup / (double)p;

        /*
         * Amdahl's Law theoretical limit for this thread count, using
         * the estimated serial fraction.
         */
        double amdahl_limit = (s_estimate > 0.0)
            ? 1.0 / (s_estimate + (1.0 - s_estimate) / (double)p)
            : (double)p;  /* fully parallel: limit = p */

        printf("%-8d  %10.2f  %10.3f  %10.1f%%  %14.3f\n",
               p, ms, speedup, efficiency * 100.0, amdahl_limit);

        if (!printed_efficiency_warning && efficiency < 0.5 && p > 1) {
            printf("  ^ efficiency dropped below 50%% at %d threads"
                   " (Amdahl ceiling reached)\n", p);
            printed_efficiency_warning = 1;
        }
    }

    printf("\n");
    if (s_estimate > 0.0) {
        printf("Estimated serial fraction s ≈ %.4f  (%.2f%%)\n",
               s_estimate, s_estimate * 100.0);
        printf("Theoretical max speedup (p→∞) ≈ %.1fx  [= 1/s]\n",
               1.0 / s_estimate);
    }
    printf("\n");

    free(A); free(B); free(C);
}

/* -------------------------------------------------------------------------
 * Main
 * ---------------------------------------------------------------------- */

int main(void)
{
    srand(42);  /* reproducible random matrices */

    printf("Parallel Matrix Multiplication Benchmark\n");
    printf("OpenMP max threads: %d\n", omp_get_max_threads());
    printf("Tile size: %d\n", TILE_SIZE);
    printf("Trials per measurement: %d (minimum taken)\n\n", TRIALS);

    part1_sweep();
    part2_amdahl();

    return 0;
}

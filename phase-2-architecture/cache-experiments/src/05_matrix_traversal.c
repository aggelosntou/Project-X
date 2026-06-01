/*
 * 05_matrix_traversal.c — Row-major vs column-major matrix traversal
 *
 * C stores 2D arrays in ROW-MAJOR order: A[i][j] and A[i][j+1] are adjacent.
 * Row-major traversal (i outer, j inner) accesses consecutive memory → cache-friendly.
 * Column-major traversal (j outer, i inner) accesses A[0][j], A[1][j], ... which
 * are N elements apart → stride-N → cache-UNFRIENDLY.
 *
 * For an N×N matrix of floats, stride = N * sizeof(float) bytes.
 * With N=2048 and float=4 bytes, stride = 8192 bytes ≫ 64-byte cache line.
 *
 * This is directly relevant to ML: a matrix multiply B×A where B is stored
 * column-major (or you access A's columns) will thrash the cache.
 * BLAS solves this by transposing or blocking.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <string.h>

static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

/* Use a large matrix to exceed L3 cache.
 * N=2048 → 2048*2048*4 = 16MB. L3 on Apple M-series = 8-12MB.
 * For most laptops this exceeds L3. Adjust N if needed.        */
#define N 2048

int main(void) {
    printf("Matrix Traversal: Row-major vs Column-major\n");
    printf("============================================\n");
    printf("Matrix: %d x %d floats = %d MB\n\n",
           N, N, (int)(N * N * sizeof(float) / (1024*1024)));

    float *A = malloc((size_t)N * N * sizeof(float));
    if (!A) { perror("malloc"); return 1; }

    /* Initialize */
    for (int i = 0; i < N * N; i++) A[i] = (float)(i + 1);

    double row_time, col_time;

    /* ---- ROW-MAJOR traversal ---- */
    {
        volatile double sink = 0.0;
        /* Warm up */
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                sink += A[i*N + j];

        uint64_t start = now_ns();
        sink = 0.0;
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                sink += A[i*N + j];   /* sequential memory access */
            }
        }
        uint64_t end = now_ns();
        if (sink == 0.0) printf("(sink)\n");

        row_time = (double)(end - start) / 1e6;  /* ms */
        double bw = (double)N*N*sizeof(float) / (double)(end-start) * 1000.0;  /* GB/s */
        printf("Row-major (cache-friendly):    %.2f ms  |  %.2f GB/s\n", row_time, bw);
    }

    /* ---- COLUMN-MAJOR traversal ---- */
    {
        volatile double sink = 0.0;

        uint64_t start = now_ns();
        for (int j = 0; j < N; j++) {
            for (int i = 0; i < N; i++) {
                sink += A[i*N + j];   /* stride-N memory access */
            }
        }
        uint64_t end = now_ns();
        if (sink == 0.0) printf("(sink)\n");

        col_time = (double)(end - start) / 1e6;  /* ms */
        double bw = (double)N*N*sizeof(float) / (double)(end-start) * 1000.0;
        printf("Column-major (cache-hostile):  %.2f ms  |  %.2f GB/s\n", col_time, bw);
    }

    printf("\nSlowdown from column-major: %.1fx\n", col_time / row_time);

    printf("\nWhy?\n");
    printf("  In row-major, inner loop accesses A[i*N+0], A[i*N+1], A[i*N+2]...\n");
    printf("  These are consecutive in memory. Loading one cache line (64 bytes)\n");
    printf("  gives us 16 floats for free.\n\n");
    printf("  In column-major, inner loop accesses A[0*N+j], A[1*N+j], A[2*N+j]...\n");
    printf("  These are %d bytes apart (N=%d floats × 4 bytes).\n", N*4, N);
    printf("  Each access brings in a fresh cache line that we use only ONCE.\n\n");
    printf("  ML relevance: attention score matrix S = Q*K^T\n");
    printf("  If K is stored row-major, computing K^T access is column-major.\n");
    printf("  BLAS/cuBLAS handles this by accepting a 'transpose' flag.\n");

    free(A);
    return 0;
}

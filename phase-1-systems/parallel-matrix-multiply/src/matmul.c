/*
 * matmul.c — Four matrix multiplication implementations.
 *
 * All operate on square N×N matrices of floats in row-major order.
 * Element [i][j] is at offset i*N + j.
 *
 * =========================================================================
 * LOOP ORDER: WHY ikj BEATS ijk
 * =========================================================================
 *
 * For row-major storage:
 *
 *   ijk order (classic textbook):
 *     for i: for j: for k: C[i][j] += A[i][k] * B[k][j]
 *     Inner loop varies j — B[k][j] steps through a column of B.
 *     Column stride = N floats = N*4 bytes.  For N=1024 that is 4 KB per
 *     step, almost guaranteed to miss the cache on every iteration.
 *
 *   ikj order (cache-friendly):
 *     for i: for k: for j: C[i][j] += A[i][k] * B[k][j]
 *     Inner loop varies j — B[k][j] now traverses ROW k of B (sequential).
 *     C[i][j] also steps sequentially through row i of C.
 *     A[i][k] is a scalar held in a register for the entire inner loop.
 *     Result: ~N contiguous float reads per inner loop → cache-line reuse.
 *
 * =========================================================================
 * TRANSPOSE TRICK
 * =========================================================================
 *
 * Even ikj has a stride-N access in B during the middle loop (k varies,
 * accessing B[k][j] for fixed j — but since j is the inner variable this
 * is fine).  The transposed version goes one step further: copy B into
 * Bt such that Bt[j][k] = B[k][j], then compute:
 *
 *   C[i][j] += A[i][k] * Bt[j][k]
 *
 * Now both A and Bt are read row-wise.  For large N this yields measurably
 * better L1/L2 hit rates than the raw ikj version.
 *
 * =========================================================================
 * TILING (BLOCKING)
 * =========================================================================
 *
 * For an N×N matrix with tile size T, the work is split into (N/T)^3 tile
 * triples.  Within each tile, all three sub-matrices fit in L1 cache:
 *
 *   working set = 3 * T * T * sizeof(float) = 3 * T^2 * 4 bytes
 *   T=32 → 12,288 bytes ≈ 12 KB (fits in typical 32 KB L1 data cache)
 *   T=64 → 49,152 bytes ≈ 48 KB (fits in 64 KB L1 of modern cores)
 *
 * Without tiling, for each of the N^2 output elements we read an entire
 * row of A and column of B — N floats each, N^2 times → N^3 cache misses.
 * With tiling, each T×T tile of A and B is reused T times per tile triple,
 * reducing cache misses by roughly a factor of T.
 */

#include "matmul.h"

#include <stdlib.h>
#include <string.h>

/* Macro for element access — avoids index-arithmetic mistakes. */
#define A(i, k) A[(i)*N + (k)]
#define B(k, j) B[(k)*N + (j)]
#define C(i, j) C[(i)*N + (j)]

/* =========================================================================
 * 1. Naive — ikj loop order, no parallelism, no tiling
 * ====================================================================== */

void matmul_naive(const float *A, const float *B, float *C, int N)
{
    /* Zero C first (calloc would work too, but here we get a passed-in C). */
    memset(C, 0, (size_t)N * N * sizeof(float));

    for (int i = 0; i < N; i++) {
        for (int k = 0; k < N; k++) {
            /*
             * a_ik is hoisted out of the j-loop by the compiler (and by us
             * explicitly to make the intent clear).  It is held in a register
             * for all N iterations of the inner loop.
             */
            float a_ik = A(i, k);
            for (int j = 0; j < N; j++) {
                /*
                 * Inner loop: sequential access to row k of B and row i of C.
                 * Both are stride-1 → hardware prefetcher kicks in.
                 */
                C(i, j) += a_ik * B(k, j);
            }
        }
    }
}

/* =========================================================================
 * 2. Transposed-B — copy B into Bt (Bt[j][k] = B[k][j]), then ikj on Bt
 * ====================================================================== */

void matmul_transposed(const float *A, const float *B, float *C, int N)
{
    /* Allocate and fill the transposed copy of B. */
    float *Bt = malloc((size_t)N * N * sizeof(float));
    if (!Bt) abort();

    /* Transpose: Bt[j][k] = B[k][j] */
    for (int k = 0; k < N; k++) {
        for (int j = 0; j < N; j++) {
            Bt[j * N + k] = B(k, j);
        }
    }

    memset(C, 0, (size_t)N * N * sizeof(float));

    /*
     * Now compute C[i][j] += A[i][k] * Bt[j][k].
     * The outer two loops (i, j) select one element of C.
     * The inner loop (k) reads row i of A and row j of Bt — both sequential.
     *
     * Trade-off: the transpose step itself costs O(N^2) and a full extra
     * N×N allocation.  For large N (e.g. N=2048) this might not fit in L3
     * cache, but the main multiply still benefits from sequential Bt access.
     */
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A(i, k) * Bt[j * N + k];
            }
            C(i, j) = sum;
        }
    }

    free(Bt);
}

/* =========================================================================
 * 3. OpenMP — parallelise the outer i-loop, ikj body
 * ====================================================================== */

void matmul_openmp(const float *A, const float *B, float *C, int N)
{
    memset(C, 0, (size_t)N * N * sizeof(float));

    /*
     * #pragma omp parallel for schedule(static):
     *   - Distribute the i-iterations evenly across threads at compile time.
     *   - Each thread works on a contiguous block of rows of A and C →
     *     no false sharing between threads (each row is at least 512 bytes
     *     for N=128, far larger than a 64-byte cache line).
     *   - schedule(static) has lower overhead than schedule(dynamic) when
     *     all iterations take the same time (square matrix, uniform work).
     */
#pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < N; k++) {
            float a_ik = A(i, k);
            for (int j = 0; j < N; j++) {
                C(i, j) += a_ik * B(k, j);
            }
        }
    }
}

/* =========================================================================
 * 4. Tiled + OpenMP — cache blocking with tile size T, parallelised
 * ====================================================================== */

void matmul_tiled_openmp(const float *A, const float *B, float *C, int N, int T)
{
    memset(C, 0, (size_t)N * N * sizeof(float));

    /*
     * Six nested loops:
     *   Outer three (ii, jj, kk) — step through tiles.
     *   Inner three (i, j, k)   — step within a tile.
     *
     * For each tile triple (ii, jj, kk):
     *   We accumulate into C[ii..ii+T][jj..jj+T] using
     *   A[ii..ii+T][kk..kk+T] and B[kk..kk+T][jj..jj+T].
     *
     * Working set per tile triple:
     *   3 tiles × T×T floats × 4 bytes = 12 * T^2 bytes
     *   T=32 → 12,288 bytes  — fits in a 32 KB L1 data cache
     *   T=64 → 49,152 bytes  — fits in a 64 KB L1 data cache
     *
     * #pragma omp parallel for collapse(2):
     *   Collapses the ii and jj loops into one iteration space so OpenMP
     *   has (N/T)^2 independent tasks to distribute across threads.
     *   Without collapse(2), only (N/T) tasks are available from the ii
     *   loop alone — not enough to keep many threads busy for small T.
     *
     * NOTE: We guard tile boundaries with min() to handle N that is not
     * a multiple of T.
     */

#pragma omp parallel for collapse(2) schedule(static)
    for (int ii = 0; ii < N; ii += T) {
        for (int jj = 0; jj < N; jj += T) {
            /* Tile bounds (handle edge tiles). */
            int i_end = ii + T < N ? ii + T : N;
            int j_end = jj + T < N ? jj + T : N;

            for (int kk = 0; kk < N; kk += T) {
                int k_end = kk + T < N ? kk + T : N;

                /* Inner triple: accumulate one tile contribution. */
                for (int i = ii; i < i_end; i++) {
                    for (int k = kk; k < k_end; k++) {
                        float a_ik = A(i, k);
                        for (int j = jj; j < j_end; j++) {
                            C(i, j) += a_ik * B(k, j);
                        }
                    }
                }
            }
        }
    }
}

#undef A
#undef B
#undef C

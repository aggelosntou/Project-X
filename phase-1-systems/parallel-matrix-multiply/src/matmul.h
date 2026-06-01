#ifndef MATMUL_H
#define MATMUL_H

/*
 * matmul.h — Matrix multiplication: four implementations
 *
 * All functions compute C = A * B for square N×N matrices stored in
 * row-major order (C convention: element [i][j] is at index i*N + j).
 *
 * IMPLEMENTATIONS (in order of increasing performance):
 *
 *   1. matmul_naive        — Triple loop, ikj order, no special tricks.
 *                            Baseline.
 *
 *   2. matmul_transposed   — Transpose B into Bt first, then ikj on Bt.
 *                            Eliminates column-stride cache misses on B.
 *
 *   3. matmul_openmp       — Like naive but outer loop parallelised with
 *                            OpenMP.  Shows raw thread-level speedup.
 *
 *   4. matmul_tiled_openmp — Cache-tiled + OpenMP.  Maximises both data
 *                            reuse (tiling) and parallelism (OpenMP).
 *                            Tile size T is a tunable parameter.
 */

void matmul_naive(const float *A, const float *B, float *C, int N);
void matmul_transposed(const float *A, const float *B, float *C, int N);
void matmul_openmp(const float *A, const float *B, float *C, int N);
void matmul_tiled_openmp(const float *A, const float *B, float *C, int N, int T);

#endif /* MATMUL_H */

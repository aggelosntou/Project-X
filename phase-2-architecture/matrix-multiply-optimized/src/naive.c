/*
 * naive.c — Naive matrix multiply with ikj loop order
 *
 * We compute C = A * B where A, B, C are N×N matrices of floats.
 *
 * THREE LOOP ORDERS AND WHY ikj IS BEST FOR NAIVE:
 *
 * ijk order:   for i, for j, for k:  C[i][j] += A[i][k] * B[k][j]
 *   - Inner loop varies k → B[k][j] has stride N → cache-hostile
 *   - A[i][k] is sequential → fine
 *
 * ikj order:   for i, for k, for j:  C[i][j] += A[i][k] * B[k][j]
 *   - Inner loop varies j → both C[i][j] and B[k][j] are sequential
 *   - A[i][k] is loaded once per (i,k) iteration and reused N times
 *   - Best spatial locality of all naive orderings
 *
 * jki order:   for j, for k, for i:  C[i][j] += A[i][k] * B[k][j]
 *   - Inner loop varies i → A[i][k] has stride N → cache-hostile
 *
 * Result: ikj is 2-5x faster than ijk on large matrices.
 */

#include "matmul.h"
#include <string.h>

void matmul_naive(const float * restrict A,
                  const float * restrict B,
                  float       * restrict C,
                  int N) {
    /* Zero output matrix */
    memset(C, 0, (size_t)N * N * sizeof(float));

    for (int i = 0; i < N; i++) {
        for (int k = 0; k < N; k++) {
            float a_ik = A[i*N + k];   /* load A[i][k] once, reuse N times */
            for (int j = 0; j < N; j++) {
                /* C[i][j] += A[i][k] * B[k][j]
                 * Inner loop: both C[i][*] and B[k][*] accessed sequentially */
                C[i*N + j] += a_ik * B[k*N + j];
            }
        }
    }
}

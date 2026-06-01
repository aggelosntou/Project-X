/*
 * transposed.c — Matrix multiply with explicit B transpose
 *
 * Algorithm:
 *   1. Compute B^T (transpose B)
 *   2. Compute C[i][j] = dot(A[i,:], B^T[j,:])  (ijk order)
 *
 * Why this helps:
 *   Standard ijk:  inner loop accesses B[k][j] — stride N → cache-hostile
 *   With B^T:      inner loop accesses BT[j][k] — sequential → cache-friendly
 *
 * Both A[i][k] and BT[j][k] are accessed sequentially in the inner loop.
 * This gives excellent spatial locality.
 *
 * Cost: one extra O(N²) pass to transpose B.
 * Benefit: inner loop becomes a cache-friendly dot product.
 *
 * For large N this dramatically improves cache utilization.
 */

#include "matmul.h"
#include <stdlib.h>
#include <string.h>

void matmul_transposed(const float * restrict A,
                       const float * restrict B,
                       float       * restrict C,
                       int N) {
    /* Step 1: Transpose B → BT */
    float *BT = malloc((size_t)N * N * sizeof(float));
    if (!BT) return;

    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            BT[j*N + i] = B[i*N + j];

    /* Step 2: C = A * BT using ijk (now cache-friendly) */
    memset(C, 0, (size_t)N * N * sizeof(float));

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            const float *ai  = A  + i*N;  /* row i of A */
            const float *btj = BT + j*N;  /* row j of BT = column j of B */
            /* Both pointers walk sequentially — cache-friendly dot product */
            for (int k = 0; k < N; k++) {
                sum += ai[k] * btj[k];
            }
            C[i*N + j] = sum;
        }
    }

    free(BT);
}

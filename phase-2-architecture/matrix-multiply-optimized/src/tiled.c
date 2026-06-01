/*
 * tiled.c — Cache-oblivious tiled (blocked) matrix multiply
 *
 * TILING MATH:
 * For N×N matrices and tile size T:
 *
 * Without tiling, processing row i of C:
 *   - Read all N² elements of B (once per row of C) → N³/N = N² reads of B total
 *   - If N is large, B doesn't fit in cache → O(N) cache misses per B element
 *
 * With tiling (tile size T):
 *   - Divide A, B, C into T×T tiles
 *   - For each pair (tile_A, tile_B), compute their contribution to tile_C
 *   - A T×T tile uses T² × 4 bytes of memory
 *   - For T=32: 32²×4 = 4 KB per tile → 3 tiles (A_tile, B_tile, C_tile) = 12 KB → fits in L1!
 *   - Each tile of B is reused T times (once per row in tile_A)
 *   - Cache miss count: O(N³ / T) instead of O(N³)
 *
 * Choosing T:
 *   L1 cache = ~32-48 KB, cache line = 64 bytes.
 *   We need 3 tiles in L1: 3 × T² × 4 ≤ 32768  →  T² ≤ 2730  →  T ≤ 52
 *   We choose T = 32 (a power of 2, fits comfortably in L1).
 *
 * FLOP count is identical to naive: 2×N³.
 * Speedup comes purely from fewer cache misses.
 */

#include "matmul.h"
#include <string.h>

/* Tile size: 32×32 floats = 4 KB per tile, 12 KB total — fits in L1 */
#define T 32

void matmul_tiled(const float * restrict A,
                  const float * restrict B,
                  float       * restrict C,
                  int N) {
    memset(C, 0, (size_t)N * N * sizeof(float));

    /* Loop over tiles in A (rows) and B (columns) and the shared k dimension */
    for (int i0 = 0; i0 < N; i0 += T) {
        for (int k0 = 0; k0 < N; k0 += T) {
            for (int j0 = 0; j0 < N; j0 += T) {

                /* Tile boundaries (handle non-divisible N) */
                int i_end = i0 + T < N ? i0 + T : N;
                int k_end = k0 + T < N ? k0 + T : N;
                int j_end = j0 + T < N ? j0 + T : N;

                /* Compute the contribution of A[i0..i_end][k0..k_end]
                 * and B[k0..k_end][j0..j_end] to C[i0..i_end][j0..j_end].
                 * All three tiles are T×T ≤ 32×32 = 4 KB — stays in L1. */
                for (int i = i0; i < i_end; i++) {
                    for (int k = k0; k < k_end; k++) {
                        float a_ik = A[i*N + k];
                        for (int j = j0; j < j_end; j++) {
                            C[i*N + j] += a_ik * B[k*N + j];
                        }
                    }
                }
            }
        }
    }
}

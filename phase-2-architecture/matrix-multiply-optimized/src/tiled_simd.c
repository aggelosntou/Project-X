/*
 * tiled_simd.c — Combined tiling + SIMD matrix multiply
 *
 * This combines both optimizations:
 *   1. Tiling: ensures the working set (3 tiles) fits in L1 cache
 *   2. SIMD:   processes 8 floats per instruction with AVX2
 *
 * The key insight: SIMD gives you compute bandwidth, tiling gives you
 * memory bandwidth. You need both for peak performance.
 *
 * FLOP analysis (N=1024, T=32):
 *   Total FLOPs: 2 × 1024³ ≈ 2.1 billion
 *   AVX2 FMA: 8 floats × 2 (FMA counts as 2) = 16 FLOPs per instruction
 *   At 3 GHz: 3e9 × 16 = 48 GFLOPS theoretical peak (single core)
 *   With tiling: working set in L1 → memory not a bottleneck
 *   Expected: 40-80% of theoretical peak
 */

#include "matmul.h"
#include <string.h>

#define T 32  /* tile size (same as tiled.c) */

#ifdef __AVX2__
#include <immintrin.h>

void matmul_tiled_simd(const float * restrict A,
                       const float * restrict B,
                       float       * restrict C,
                       int N) {
    memset(C, 0, (size_t)N * N * sizeof(float));

    for (int i0 = 0; i0 < N; i0 += T) {
        for (int k0 = 0; k0 < N; k0 += T) {
            for (int j0 = 0; j0 < N; j0 += T) {

                int i_end = i0 + T < N ? i0 + T : N;
                int k_end = k0 + T < N ? k0 + T : N;
                int j_end = j0 + T < N ? j0 + T : N;

                for (int i = i0; i < i_end; i++) {
                    for (int k = k0; k < k_end; k++) {
                        __m256 a_vec = _mm256_set1_ps(A[i*N + k]);

                        int j = j0;
                        /* Vectorized inner loop over j */
                        for (; j <= j_end - 8; j += 8) {
                            __m256 b_vec = _mm256_loadu_ps(B + k*N + j);
                            __m256 c_vec = _mm256_loadu_ps(C + i*N + j);
#ifdef __FMA__
                            c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
#else
                            c_vec = _mm256_add_ps(c_vec, _mm256_mul_ps(a_vec, b_vec));
#endif
                            _mm256_storeu_ps(C + i*N + j, c_vec);
                        }
                        /* Scalar tail */
                        for (; j < j_end; j++) {
                            C[i*N + j] += A[i*N + k] * B[k*N + j];
                        }
                    }
                }
            }
        }
    }
}

#else
/* Scalar fallback */
#warning "AVX2 not available, using scalar tiled for matmul_tiled_simd"

void matmul_tiled_simd(const float * restrict A,
                       const float * restrict B,
                       float       * restrict C,
                       int N) {
    memset(C, 0, (size_t)N * N * sizeof(float));

    for (int i0 = 0; i0 < N; i0 += T) {
        for (int k0 = 0; k0 < N; k0 += T) {
            for (int j0 = 0; j0 < N; j0 += T) {
                int i_end = i0 + T < N ? i0 + T : N;
                int k_end = k0 + T < N ? k0 + T : N;
                int j_end = j0 + T < N ? j0 + T : N;
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
#endif /* __AVX2__ */

/*
 * simd.c — AVX2 SIMD matrix multiply
 *
 * SIMD (Single Instruction, Multiple Data): one instruction operates on
 * a vector of values simultaneously.
 *
 * AVX2 registers (ymm0..ymm15) hold 256 bits = 8 × float32.
 * Key intrinsics used:
 *   _mm256_loadu_ps(ptr)         — load 8 floats from unaligned memory
 *   _mm256_set1_ps(val)          — broadcast one float to all 8 lanes
 *   _mm256_fmadd_ps(a, b, c)     — c = a*b + c  (FMA, requires -mfma)
 *   _mm256_storeu_ps(ptr, vec)   — store 8 floats to memory
 *   _mm256_setzero_ps()          — vector of 8 zeros
 *
 * We use the ikj loop order. For each (i,k):
 *   - Broadcast A[i][k] to all 8 lanes: a_vec = [A[i][k]] × 8
 *   - Process B[k][j..j+7] and C[i][j..j+7] in one vector FMA
 *   - This computes 8 output elements per instruction
 *
 * Theoretical speedup over scalar: 8× (8 floats per AVX2 instruction).
 * With FMA: 16× over separate multiply + add.
 *
 * NOTE: This file requires AVX2 and FMA support.
 * On macOS with Apple Silicon (ARM), AVX2 is not available.
 * We provide a scalar fallback.
 * On x86-64 Macs (Intel) and Linux x86, this will use real AVX2.
 */

#include "matmul.h"
#include <string.h>

#ifdef __AVX2__
#include <immintrin.h>

void matmul_simd(const float * restrict A,
                 const float * restrict B,
                 float       * restrict C,
                 int N) {
    memset(C, 0, (size_t)N * N * sizeof(float));

    for (int i = 0; i < N; i++) {
        for (int k = 0; k < N; k++) {
            __m256 a_vec = _mm256_set1_ps(A[i*N + k]);  /* broadcast A[i][k] */

            int j;
            /* Process 8 floats at a time */
            for (j = 0; j <= N - 8; j += 8) {
                __m256 b_vec = _mm256_loadu_ps(B + k*N + j);
                __m256 c_vec = _mm256_loadu_ps(C + i*N + j);
#ifdef __FMA__
                c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
#else
                c_vec = _mm256_add_ps(c_vec, _mm256_mul_ps(a_vec, b_vec));
#endif
                _mm256_storeu_ps(C + i*N + j, c_vec);
            }

            /* Handle remaining elements (when N is not divisible by 8) */
            for (; j < N; j++) {
                C[i*N + j] += A[i*N + k] * B[k*N + j];
            }
        }
    }
}

#else
/* Scalar fallback for non-AVX2 systems (e.g., Apple Silicon) */
#warning "AVX2 not available, using scalar fallback for matmul_simd"

void matmul_simd(const float * restrict A,
                 const float * restrict B,
                 float       * restrict C,
                 int N) {
    memset(C, 0, (size_t)N * N * sizeof(float));
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < N; k++) {
            float a_ik = A[i*N + k];
            for (int j = 0; j < N; j++) {
                C[i*N + j] += a_ik * B[k*N + j];
            }
        }
    }
}
#endif /* __AVX2__ */

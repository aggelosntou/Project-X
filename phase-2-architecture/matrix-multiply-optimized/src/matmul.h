#ifndef MATMUL_H
#define MATMUL_H

/*
 * All matmul implementations share this interface:
 *   C = A * B  (N×N float matrices, row-major)
 */

void matmul_naive      (const float *A, const float *B, float *C, int N);
void matmul_transposed (const float *A, const float *B, float *C, int N);
void matmul_tiled      (const float *A, const float *B, float *C, int N);
void matmul_simd       (const float *A, const float *B, float *C, int N);
void matmul_tiled_simd (const float *A, const float *B, float *C, int N);

#endif /* MATMUL_H */

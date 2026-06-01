/*
 * matrix.h — Complete matrix library for ML from scratch.
 *
 * Design decisions:
 *   - Row-major storage: element (i,j) lives at data[i*cols + j].
 *     This matches C's native 2D-array layout and is cache-friendly for
 *     row-wise operations (each row is contiguous in memory).
 *   - All "result" parameters (c) must be pre-allocated by the caller.
 *     This avoids hidden allocations inside hot loops and makes ownership clear.
 *   - float (32-bit) rather than double: modern ML hardware is optimized for fp32.
 */

#ifndef MATRIX_H
#define MATRIX_H

#include <stddef.h>  /* size_t */

/* ------------------------------------------------------------------ */
/*  Core struct                                                         */
/* ------------------------------------------------------------------ */

typedef struct {
    float *data;   /* Flat array of rows*cols floats, row-major order  */
    int    rows;   /* Number of rows                                   */
    int    cols;   /* Number of columns                                */
} Matrix;

/* ------------------------------------------------------------------ */
/*  Lifecycle                                                           */
/* ------------------------------------------------------------------ */

/* Allocate a zero-filled matrix of shape (rows, cols).
 * Returns NULL on allocation failure. */
Matrix *mat_new(int rows, int cols);

/* Create a matrix by copying from a C array of length rows*cols.
 * The array is assumed to be in row-major order. */
Matrix *mat_from_array(const float *data, int rows, int cols);

/* Free the matrix and its data buffer. Safe to call with NULL. */
void mat_free(Matrix *m);

/* Return a deep copy (new allocation). */
Matrix *mat_copy(const Matrix *m);

/* ------------------------------------------------------------------ */
/*  Arithmetic — all write result into pre-allocated c                 */
/* ------------------------------------------------------------------ */

/* Element-wise: c = a + b  (shapes must match) */
void mat_add(const Matrix *a, const Matrix *b, Matrix *c);

/* Element-wise: c = a - b */
void mat_sub(const Matrix *a, const Matrix *b, Matrix *c);

/* Matrix product: c = a @ b   (a: m×k, b: k×n, c: m×n)
 * Uses a tiled (blocked) algorithm for better cache utilization. */
void mat_mul(const Matrix *a, const Matrix *b, Matrix *c);

/* Hadamard (element-wise) product: c = a ⊙ b */
void mat_mul_elementwise(const Matrix *a, const Matrix *b, Matrix *c);

/* Scalar multiply: c = scalar * a */
void mat_scale(const Matrix *a, float scalar, Matrix *c);

/* Transpose: c[j][i] = a[i][j]  (c must be shape cols×rows) */
void mat_transpose(const Matrix *a, Matrix *c);

/* Broadcast-add bias: c[i][j] = a[i][j] + bias[0][j]
 * a is (m×n), bias is (1×n).  Broadcasting is the key operation in
 * neural networks: every row of the batch gets the same bias vector. */
void mat_add_bias(const Matrix *a, const Matrix *bias, Matrix *c);

/* ------------------------------------------------------------------ */
/*  Activation functions                                                */
/* ------------------------------------------------------------------ */

/* ReLU: c[i][j] = max(0, a[i][j])
 * The key non-linearity that makes deep nets trainable. */
void mat_relu(const Matrix *a, Matrix *c);

/* ReLU backward pass:
 *   grad_in[i][j] = grad_out[i][j]  if a[i][j] > 0, else 0
 * We need 'a' (the pre-activation) to know which units were active. */
void mat_relu_backward(const Matrix *grad_out, const Matrix *a, Matrix *grad_in);

/* Sigmoid: c[i][j] = 1 / (1 + exp(-a[i][j])) */
void mat_sigmoid(const Matrix *a, Matrix *c);

/* Numerically-stable softmax along rows:
 *   For each row r:  subtract max(row r), exponentiate, divide by sum.
 *   Without max-shift, exp() overflows for large logits.               */
void mat_softmax(const Matrix *a, Matrix *c);

/* ------------------------------------------------------------------ */
/*  Reductions                                                          */
/* ------------------------------------------------------------------ */

float mat_sum(const Matrix *a);
float mat_mean(const Matrix *a);

/* ------------------------------------------------------------------ */
/*  Debug                                                               */
/* ------------------------------------------------------------------ */

/* Pretty-print the matrix with an optional name label. */
void mat_print(const Matrix *a, const char *name);

#endif /* MATRIX_H */

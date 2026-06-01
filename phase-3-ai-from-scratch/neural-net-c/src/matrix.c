/*
 * matrix.c — Implementation of the matrix library.
 *
 * Tiled matrix multiply:
 *   Standard O(n^3) matrix multiply has terrible cache behavior — for large
 *   matrices the inner loop reads from non-contiguous memory.  By partitioning
 *   the matrices into TILE×TILE blocks that fit in L1 cache, we dramatically
 *   reduce cache misses.  TILE=8 means each block is 8×8×4 = 256 bytes ≈ 4
 *   cache lines, which fits comfortably in a 32 KB L1 cache.
 */

#include "matrix.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>

#define TILE 8  /* Tile size for blocked matrix multiply */

/* ------------------------------------------------------------------ */
/*  Lifecycle                                                           */
/* ------------------------------------------------------------------ */

Matrix *mat_new(int rows, int cols) {
    assert(rows > 0 && cols > 0);
    Matrix *m = (Matrix *)malloc(sizeof(Matrix));
    if (!m) return NULL;
    m->rows = rows;
    m->cols = cols;
    /* calloc gives us zeroed memory, important for bias init etc. */
    m->data = (float *)calloc((size_t)(rows * cols), sizeof(float));
    if (!m->data) { free(m); return NULL; }
    return m;
}

Matrix *mat_from_array(const float *data, int rows, int cols) {
    Matrix *m = mat_new(rows, cols);
    if (!m) return NULL;
    memcpy(m->data, data, (size_t)(rows * cols) * sizeof(float));
    return m;
}

void mat_free(Matrix *m) {
    if (!m) return;
    free(m->data);
    free(m);
}

Matrix *mat_copy(const Matrix *m) {
    assert(m);
    return mat_from_array(m->data, m->rows, m->cols);
}

/* ------------------------------------------------------------------ */
/*  Convenience macro: element access                                   */
/* ------------------------------------------------------------------ */
/* A(i,j) expands to A->data[i * A->cols + j], which is the row-major
 * offset.  Row i starts at index i*cols, then we step j columns right. */
#define AT(M, i, j)  ((M)->data[(i) * (M)->cols + (j)])

/* ------------------------------------------------------------------ */
/*  Arithmetic                                                          */
/* ------------------------------------------------------------------ */

void mat_add(const Matrix *a, const Matrix *b, Matrix *c) {
    assert(a->rows == b->rows && a->cols == b->cols);
    assert(c->rows == a->rows && c->cols == a->cols);
    int n = a->rows * a->cols;
    for (int i = 0; i < n; i++)
        c->data[i] = a->data[i] + b->data[i];
}

void mat_sub(const Matrix *a, const Matrix *b, Matrix *c) {
    assert(a->rows == b->rows && a->cols == b->cols);
    assert(c->rows == a->rows && c->cols == a->cols);
    int n = a->rows * a->cols;
    for (int i = 0; i < n; i++)
        c->data[i] = a->data[i] - b->data[i];
}

/* Tiled matrix multiply: c = a @ b
 *
 * Naive triple loop: for i for j for k: c[i][j] += a[i][k] * b[k][j]
 *   Problem: b[k][j] accesses column j of B, which is strided (stride=cols_b)
 *   in memory — terrible for cache.
 *
 * Tiled approach: process TILE×TILE sub-blocks at a time.
 *   Within each block, access is still row-major for both A and B (for B we
 *   access rows within the tile, which are contiguous).  The key insight is
 *   that the TILE rows of A and TILE rows of B all fit in cache simultaneously.
 */
void mat_mul(const Matrix *a, const Matrix *b, Matrix *c) {
    assert(a->cols == b->rows);
    assert(c->rows == a->rows && c->cols == b->cols);

    int M = a->rows, K = a->cols, N = b->cols;

    /* Zero the output (c may be reused across calls) */
    memset(c->data, 0, (size_t)(M * N) * sizeof(float));

    for (int i0 = 0; i0 < M; i0 += TILE) {
        for (int k0 = 0; k0 < K; k0 += TILE) {
            for (int j0 = 0; j0 < N; j0 += TILE) {
                /* Process the tile [i0..i0+TILE) × [j0..j0+TILE) of C,
                 * accumulating contributions from columns k0..k0+TILE of A
                 * and rows k0..k0+TILE of B. */
                int i_end = i0 + TILE < M ? i0 + TILE : M;
                int k_end = k0 + TILE < K ? k0 + TILE : K;
                int j_end = j0 + TILE < N ? j0 + TILE : N;

                for (int i = i0; i < i_end; i++) {
                    for (int k = k0; k < k_end; k++) {
                        float a_ik = AT(a, i, k);  /* Load once, reuse in j-loop */
                        for (int j = j0; j < j_end; j++) {
                            AT(c, i, j) += a_ik * AT(b, k, j);
                        }
                    }
                }
            }
        }
    }
}

void mat_mul_elementwise(const Matrix *a, const Matrix *b, Matrix *c) {
    assert(a->rows == b->rows && a->cols == b->cols);
    assert(c->rows == a->rows && c->cols == a->cols);
    int n = a->rows * a->cols;
    for (int i = 0; i < n; i++)
        c->data[i] = a->data[i] * b->data[i];
}

void mat_scale(const Matrix *a, float scalar, Matrix *c) {
    assert(c->rows == a->rows && c->cols == a->cols);
    int n = a->rows * a->cols;
    for (int i = 0; i < n; i++)
        c->data[i] = a->data[i] * scalar;
}

void mat_transpose(const Matrix *a, Matrix *c) {
    /* c must be shape (a->cols × a->rows) */
    assert(c->rows == a->cols && c->cols == a->rows);
    for (int i = 0; i < a->rows; i++)
        for (int j = 0; j < a->cols; j++)
            AT(c, j, i) = AT(a, i, j);
}

void mat_add_bias(const Matrix *a, const Matrix *bias, Matrix *c) {
    /* Broadcasting: bias has shape (1 × n_cols).
     * We add the same bias vector to every row of a.
     * Mathematical justification: in a neural network layer,
     *   output = input @ W + b
     * where b is a single vector added to every sample in the batch.
     * "Broadcasting" means we don't store n_batch copies of b. */
    assert(bias->rows == 1 && bias->cols == a->cols);
    assert(c->rows == a->rows && c->cols == a->cols);
    for (int i = 0; i < a->rows; i++)
        for (int j = 0; j < a->cols; j++)
            AT(c, i, j) = AT(a, i, j) + bias->data[j];
}

/* ------------------------------------------------------------------ */
/*  Activations                                                         */
/* ------------------------------------------------------------------ */

void mat_relu(const Matrix *a, Matrix *c) {
    assert(c->rows == a->rows && c->cols == a->cols);
    int n = a->rows * a->cols;
    for (int i = 0; i < n; i++)
        c->data[i] = a->data[i] > 0.0f ? a->data[i] : 0.0f;
}

void mat_relu_backward(const Matrix *grad_out, const Matrix *a, Matrix *grad_in) {
    /* Chain rule through ReLU:
     *   d/dx max(0, x) = 1 if x > 0, else 0  (the "indicator function")
     * So: grad_in = grad_out * I[a > 0]
     * We need 'a' (pre-activation) because ReLU's derivative depends on
     * whether the neuron was "on" during the forward pass. */
    assert(grad_out->rows == a->rows && grad_out->cols == a->cols);
    assert(grad_in->rows  == a->rows && grad_in->cols  == a->cols);
    int n = a->rows * a->cols;
    for (int i = 0; i < n; i++)
        grad_in->data[i] = a->data[i] > 0.0f ? grad_out->data[i] : 0.0f;
}

void mat_sigmoid(const Matrix *a, Matrix *c) {
    /* σ(x) = 1 / (1 + e^{-x})
     * For numerical stability when x << 0, use σ(x) = e^x / (1 + e^x).
     * We handle both cases to avoid overflow. */
    assert(c->rows == a->rows && c->cols == a->cols);
    int n = a->rows * a->cols;
    for (int i = 0; i < n; i++) {
        float x = a->data[i];
        if (x >= 0.0f)
            c->data[i] = 1.0f / (1.0f + expf(-x));
        else {
            float ex = expf(x);
            c->data[i] = ex / (1.0f + ex);
        }
    }
}

void mat_softmax(const Matrix *a, Matrix *c) {
    /* Numerically-stable row-wise softmax.
     *
     * Naive:  p_j = exp(x_j) / sum_k exp(x_k)
     *   Problem: if x_j = 1000, exp(1000) = +inf → NaN.
     *
     * Stable: subtract max first.
     *   p_j = exp(x_j - max_x) / sum_k exp(x_k - max_x)
     *   This is algebraically equivalent because the max cancels:
     *     exp(x_j - m) / sum exp(x_k - m)
     *   = exp(x_j) * exp(-m) / (exp(-m) * sum exp(x_k))
     *   = exp(x_j) / sum exp(x_k)                              ✓
     *   But now all exponents are ≤ 0, so exp(·) ∈ (0, 1] — no overflow.
     */
    assert(c->rows == a->rows && c->cols == a->cols);
    for (int i = 0; i < a->rows; i++) {
        /* Step 1: find row maximum */
        float max_val = AT(a, i, 0);
        for (int j = 1; j < a->cols; j++)
            if (AT(a, i, j) > max_val) max_val = AT(a, i, j);

        /* Step 2: exponentiate shifted values and accumulate sum */
        float sum = 0.0f;
        for (int j = 0; j < a->cols; j++) {
            AT(c, i, j) = expf(AT(a, i, j) - max_val);
            sum += AT(c, i, j);
        }

        /* Step 3: normalize */
        for (int j = 0; j < a->cols; j++)
            AT(c, i, j) /= sum;
    }
}

/* ------------------------------------------------------------------ */
/*  Reductions                                                          */
/* ------------------------------------------------------------------ */

float mat_sum(const Matrix *a) {
    float s = 0.0f;
    int n = a->rows * a->cols;
    for (int i = 0; i < n; i++) s += a->data[i];
    return s;
}

float mat_mean(const Matrix *a) {
    return mat_sum(a) / (float)(a->rows * a->cols);
}

/* ------------------------------------------------------------------ */
/*  Debug                                                               */
/* ------------------------------------------------------------------ */

void mat_print(const Matrix *a, const char *name) {
    if (name) printf("%s  [%d x %d]:\n", name, a->rows, a->cols);
    for (int i = 0; i < a->rows; i++) {
        printf("  [");
        for (int j = 0; j < a->cols; j++) {
            printf(" %8.4f", AT(a, i, j));
        }
        printf(" ]\n");
    }
    printf("\n");
}

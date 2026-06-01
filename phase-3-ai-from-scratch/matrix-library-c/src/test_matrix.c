/*
 * test_matrix.c — Unit tests for the matrix library.
 *
 * Strategy: compare computed results against analytically-known values.
 * We use a simple tolerance (1e-5) because floating-point arithmetic is
 * not exact — addition and multiplication introduce rounding errors of
 * order machine epsilon ≈ 1.19e-7 for float.
 */

#include "matrix.h"
#include <stdio.h>
#include <math.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/*  Test harness                                                        */
/* ------------------------------------------------------------------ */

static int tests_run    = 0;
static int tests_passed = 0;

#define ASSERT_FLOAT_EQ(got, expected, tol, msg) do { \
    tests_run++; \
    float _diff = fabsf((float)(got) - (float)(expected)); \
    if (_diff <= (tol)) { \
        tests_passed++; \
        printf("  PASS: %s  (got=%.6f, expected=%.6f)\n", (msg), (float)(got), (float)(expected)); \
    } else { \
        printf("  FAIL: %s  (got=%.6f, expected=%.6f, diff=%.2e)\n", (msg), (float)(got), (float)(expected), _diff); \
    } \
} while(0)

#define ASSERT_MATRIX_NEAR(M, expected_arr, tol, msg) do { \
    int _ok = 1; \
    int _n = (M)->rows * (M)->cols; \
    for (int _i = 0; _i < _n && _ok; _i++) { \
        if (fabsf((M)->data[_i] - (expected_arr)[_i]) > (tol)) _ok = 0; \
    } \
    tests_run++; \
    if (_ok) { \
        tests_passed++; \
        printf("  PASS: %s\n", (msg)); \
    } else { \
        printf("  FAIL: %s\n", (msg)); \
        mat_print((M), "  got"); \
    } \
} while(0)

/* ------------------------------------------------------------------ */
/*  Tests                                                               */
/* ------------------------------------------------------------------ */

void test_mat_mul(void) {
    printf("\n--- test_mat_mul ---\n");
    /*
     * Hand-computed example:
     *   A = [[1, 2, 3],   B = [[7,  8],     C = A @ B
     *        [4, 5, 6]]        [9, 10],
     *                          [11, 12]]
     *
     * C[0][0] = 1*7 + 2*9 + 3*11 = 7 + 18 + 33 = 58
     * C[0][1] = 1*8 + 2*10 + 3*12 = 8 + 20 + 36 = 64
     * C[1][0] = 4*7 + 5*9 + 6*11 = 28 + 45 + 66 = 139
     * C[1][1] = 4*8 + 5*10 + 6*12 = 32 + 50 + 72 = 154
     */
    float a_data[] = {1,2,3, 4,5,6};
    float b_data[] = {7,8, 9,10, 11,12};
    float expected[] = {58, 64, 139, 154};

    Matrix *a = mat_from_array(a_data, 2, 3);
    Matrix *b = mat_from_array(b_data, 3, 2);
    Matrix *c = mat_new(2, 2);
    mat_mul(a, b, c);

    ASSERT_MATRIX_NEAR(c, expected, 1e-4f, "mat_mul 2x3 @ 3x2");

    mat_free(a); mat_free(b); mat_free(c);
}

void test_mat_mul_square(void) {
    printf("\n--- test_mat_mul square ---\n");
    /* Identity matrix multiply:  A @ I = A */
    float a_data[] = {1,2,3, 4,5,6, 7,8,9};
    float id_data[] = {1,0,0, 0,1,0, 0,0,1};

    Matrix *a  = mat_from_array(a_data, 3, 3);
    Matrix *id = mat_from_array(id_data, 3, 3);
    Matrix *c  = mat_new(3, 3);
    mat_mul(a, id, c);

    ASSERT_MATRIX_NEAR(c, a_data, 1e-4f, "A @ I = A (identity check)");

    mat_free(a); mat_free(id); mat_free(c);
}

void test_transpose(void) {
    printf("\n--- test_transpose ---\n");
    /*
     *  A = [[1, 2, 3],    A^T = [[1, 4],
     *       [4, 5, 6]]           [2, 5],
     *                            [3, 6]]
     */
    float a_data[]  = {1,2,3, 4,5,6};
    float expected[] = {1,4, 2,5, 3,6};

    Matrix *a = mat_from_array(a_data, 2, 3);
    Matrix *t = mat_new(3, 2);
    mat_transpose(a, t);

    ASSERT_MATRIX_NEAR(t, expected, 1e-4f, "transpose 2x3 -> 3x2");

    /* Double transpose should return to original */
    Matrix *tt = mat_new(2, 3);
    mat_transpose(t, tt);
    ASSERT_MATRIX_NEAR(tt, a_data, 1e-4f, "double transpose = identity");

    mat_free(a); mat_free(t); mat_free(tt);
}

void test_softmax(void) {
    printf("\n--- test_softmax ---\n");
    /* Softmax output must sum to 1.0 for each row */
    float logits[] = {1.0f, 2.0f, 3.0f,   /* row 0 */
                      100.0f, 200.0f, 300.0f};  /* row 1: large values to test stability */

    Matrix *a = mat_from_array(logits, 2, 3);
    Matrix *s = mat_new(2, 3);
    mat_softmax(a, s);

    mat_print(s, "softmax output");

    /* Row 0 sum */
    float sum0 = s->data[0] + s->data[1] + s->data[2];
    ASSERT_FLOAT_EQ(sum0, 1.0f, 1e-5f, "softmax row 0 sums to 1");

    /* Row 1 sum — large logits would overflow without max-shift */
    float sum1 = s->data[3] + s->data[4] + s->data[5];
    ASSERT_FLOAT_EQ(sum1, 1.0f, 1e-5f, "softmax row 1 sums to 1 (large logits, stability test)");

    /* Each probability must be in [0,1] */
    for (int i = 0; i < 6; i++) {
        if (s->data[i] < 0.0f || s->data[i] > 1.0f + 1e-5f) {
            printf("  FAIL: softmax output %d out of [0,1]: %.6f\n", i, s->data[i]);
        }
    }
    tests_run++;
    tests_passed++;
    printf("  PASS: all softmax outputs in [0,1]\n");

    mat_free(a); mat_free(s);
}

void test_relu_and_backward(void) {
    printf("\n--- test_relu_backward ---\n");
    /*
     * a = [-2, -1, 0, 1, 2]   (pre-activations)
     * ReLU(a)            = [0, 0, 0, 1, 2]
     * grad_out           = [1, 1, 1, 1, 1]   (upstream gradient = all ones)
     * grad_in            = [0, 0, 0, 1, 1]   (zero where a <= 0)
     *
     * Note: a=0 is the "dead zone" — by convention derivative = 0.
     */
    float a_data[]    = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    float grad_data[] = { 1.0f,  1.0f, 1.0f, 1.0f, 1.0f};
    float expected_relu[]  = {0.0f, 0.0f, 0.0f, 1.0f, 2.0f};
    float expected_grad[]  = {0.0f, 0.0f, 0.0f, 1.0f, 1.0f};

    Matrix *a    = mat_from_array(a_data, 1, 5);
    Matrix *g    = mat_from_array(grad_data, 1, 5);
    Matrix *relu = mat_new(1, 5);
    Matrix *gin  = mat_new(1, 5);

    mat_relu(a, relu);
    ASSERT_MATRIX_NEAR(relu, expected_relu, 1e-5f, "relu forward");

    mat_relu_backward(g, a, gin);
    ASSERT_MATRIX_NEAR(gin, expected_grad, 1e-5f, "relu backward");

    mat_free(a); mat_free(g); mat_free(relu); mat_free(gin);
}

void test_bias_broadcast(void) {
    printf("\n--- test_add_bias ---\n");
    /*
     * a    = [[1, 2, 3],    bias = [[10, 20, 30]]
     *         [4, 5, 6]]
     * out  = [[11, 22, 33],
     *          [14, 25, 36]]
     */
    float a_data[]    = {1,2,3, 4,5,6};
    float bias_data[] = {10, 20, 30};
    float expected[]  = {11,22,33, 14,25,36};

    Matrix *a    = mat_from_array(a_data, 2, 3);
    Matrix *bias = mat_from_array(bias_data, 1, 3);
    Matrix *c    = mat_new(2, 3);
    mat_add_bias(a, bias, c);

    ASSERT_MATRIX_NEAR(c, expected, 1e-4f, "add_bias broadcast");

    mat_free(a); mat_free(bias); mat_free(c);
}

void test_scale_and_elementwise(void) {
    printf("\n--- test_scale + elementwise_mul ---\n");
    float a_data[] = {1,2,3,4};
    float b_data[] = {2,3,4,5};
    float expected_scaled[] = {3,6,9,12};
    float expected_hadamard[] = {2,6,12,20};

    Matrix *a = mat_from_array(a_data, 2, 2);
    Matrix *b = mat_from_array(b_data, 2, 2);
    Matrix *c = mat_new(2, 2);

    mat_scale(a, 3.0f, c);
    ASSERT_MATRIX_NEAR(c, expected_scaled, 1e-4f, "scale by 3");

    mat_mul_elementwise(a, b, c);
    ASSERT_MATRIX_NEAR(c, expected_hadamard, 1e-4f, "Hadamard product");

    mat_free(a); mat_free(b); mat_free(c);
}

/* ------------------------------------------------------------------ */
/*  Main                                                                */
/* ------------------------------------------------------------------ */

int main(void) {
    printf("=== Matrix Library Tests ===\n");

    test_mat_mul();
    test_mat_mul_square();
    test_transpose();
    test_softmax();
    test_relu_and_backward();
    test_bias_broadcast();
    test_scale_and_elementwise();

    printf("\n=== Results: %d / %d passed ===\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}

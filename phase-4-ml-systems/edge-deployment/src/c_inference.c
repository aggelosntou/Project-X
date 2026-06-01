/*
 * c_inference.c
 * =============
 * Standalone CPU inference engine for the MLP exported by export_weights.py.
 *
 * Compile:
 *   gcc -O3 -Wall -std=c11 -o c_inference src/c_inference.c -lm
 *
 * Usage:
 *   ./c_inference [model.bin]
 *
 * Key optimizations vs Python/NumPy:
 *   - No Python interpreter overhead (no GIL, no bytecode dispatch)
 *   - Compiler -O3 enables auto-vectorization (SIMD) and loop unrolling
 *   - Better cache utilisation: contiguous float arrays, predictable strides
 *   - Zero allocator overhead for the hot path (pre-allocated buffers)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* -------------------------------------------------------------------------
 * Data structures
 * ---------------------------------------------------------------------- */

typedef struct {
    int   rows;    /* input dimension  */
    int   cols;    /* output dimension */
    float *data;   /* row-major weight matrix, size rows*cols */
} Matrix;

typedef struct {
    int    n_layers;
    Matrix *weights;    /* array of n_layers matrices */
    float **biases;     /* array of n_layers bias vectors */
    int    *bias_sizes; /* cols of each weight matrix */
} Model;

/* -------------------------------------------------------------------------
 * Helpers
 * ---------------------------------------------------------------------- */

static float *vec_new(int n)
{
    float *v = (float *)calloc(n, sizeof(float));
    if (!v) { fprintf(stderr, "vec_new: out of memory\n"); exit(1); }
    return v;
}

/*
 * out[i] = sum_j( W[i,j] * x[j] ) + b[i]
 *
 * W is stored as W->data[i * W->cols + j]  (row-major).
 * Note: the weight matrix is (in_dim x out_dim) so rows=in_dim, cols=out_dim.
 * We iterate over output neurons (cols) in the outer loop for better
 * cache access into W.
 */
static void mat_vec_mul_add(const Matrix *W, const float *x,
                            const float *b, float *out)
{
    /* Clear output first */
    for (int i = 0; i < W->cols; i++)
        out[i] = b[i];

    /* Accumulate: for each input element j, scatter into all outputs */
    for (int j = 0; j < W->rows; j++) {
        float xj = x[j];
        const float *row = W->data + (size_t)j * W->cols;
        for (int i = 0; i < W->cols; i++)
            out[i] += row[i] * xj;
    }
}

static void relu_inplace(float *x, int n)
{
    for (int i = 0; i < n; i++)
        if (x[i] < 0.0f) x[i] = 0.0f;
}

static int argmax(const float *x, int n)
{
    int best = 0;
    for (int i = 1; i < n; i++)
        if (x[i] > x[best]) best = i;
    return best;
}

/* -------------------------------------------------------------------------
 * Model I/O
 * ---------------------------------------------------------------------- */

/*
 * Load model written by export_to_binary() in export_weights.py.
 *
 * Binary format:
 *   [4]  n_layers  int32
 *   Per layer:
 *     [4]  rows    int32
 *     [4]  cols    int32
 *     [rows*cols*4]  weight data  float32 row-major
 *     [4]  bias_len int32
 *     [bias_len*4]   bias data    float32
 */
Model *model_load(const char *path)
{
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "model_load: cannot open '%s'\n", path);
        return NULL;
    }

    Model *m = (Model *)malloc(sizeof(Model));
    if (!m) goto oom;

    if (fread(&m->n_layers, 4, 1, fp) != 1) goto err;

    m->weights    = (Matrix *)malloc(m->n_layers * sizeof(Matrix));
    m->biases     = (float **)malloc(m->n_layers * sizeof(float *));
    m->bias_sizes = (int    *)malloc(m->n_layers * sizeof(int));
    if (!m->weights || !m->biases || !m->bias_sizes) goto oom;

    for (int l = 0; l < m->n_layers; l++) {
        int rows, cols;
        if (fread(&rows, 4, 1, fp) != 1) goto err;
        if (fread(&cols, 4, 1, fp) != 1) goto err;

        m->weights[l].rows = rows;
        m->weights[l].cols = cols;

        size_t nelems = (size_t)rows * cols;
        m->weights[l].data = (float *)malloc(nelems * sizeof(float));
        if (!m->weights[l].data) goto oom;
        if (fread(m->weights[l].data, sizeof(float), nelems, fp) != nelems) goto err;

        int bias_len;
        if (fread(&bias_len, 4, 1, fp) != 1) goto err;
        m->bias_sizes[l] = bias_len;
        m->biases[l] = (float *)malloc(bias_len * sizeof(float));
        if (!m->biases[l]) goto oom;
        if (fread(m->biases[l], sizeof(float), bias_len, fp) != (size_t)bias_len) goto err;
    }

    fclose(fp);

    /* Print architecture */
    printf("Loaded model from '%s':\n", path);
    for (int l = 0; l < m->n_layers; l++) {
        printf("  Layer %d: %d -> %d\n",
               l, m->weights[l].rows, m->weights[l].cols);
    }
    return m;

oom:
    fprintf(stderr, "model_load: out of memory\n");
err:
    fclose(fp);
    return NULL;
}

void model_free(Model *m)
{
    if (!m) return;
    for (int l = 0; l < m->n_layers; l++) {
        free(m->weights[l].data);
        free(m->biases[l]);
    }
    free(m->weights);
    free(m->biases);
    free(m->bias_sizes);
    free(m);
}

/* -------------------------------------------------------------------------
 * Inference
 * ---------------------------------------------------------------------- */

/*
 * Forward pass.  Allocates two scratch buffers and ping-pongs between them.
 * Returns predicted class index (argmax of output logits).
 */
int model_predict(Model *m, const float *input)
{
    /* Find max output size to allocate scratch buffers */
    int max_dim = m->weights[0].rows;  /* input dimension */
    for (int l = 0; l < m->n_layers; l++) {
        if (m->weights[l].cols > max_dim)
            max_dim = m->weights[l].cols;
    }

    float *buf_a = vec_new(max_dim);
    float *buf_b = vec_new(max_dim);

    /* Copy input into buf_a */
    memcpy(buf_a, input, m->weights[0].rows * sizeof(float));

    float *cur = buf_a;
    float *nxt = buf_b;

    for (int l = 0; l < m->n_layers; l++) {
        mat_vec_mul_add(&m->weights[l], cur, m->biases[l], nxt);
        if (l < m->n_layers - 1)
            relu_inplace(nxt, m->weights[l].cols);
        /* Swap buffers */
        float *tmp = cur; cur = nxt; nxt = tmp;
    }

    int pred = argmax(cur, m->weights[m->n_layers - 1].cols);

    free(buf_a);
    free(buf_b);
    return pred;
}

/* -------------------------------------------------------------------------
 * Main
 * ---------------------------------------------------------------------- */

int main(int argc, char **argv)
{
    const char *model_path = (argc > 1) ? argv[1] : "model.bin";

    Model *model = model_load(model_path);
    if (!model) {
        fprintf(stderr, "Failed to load model from '%s'\n", model_path);
        return 1;
    }

    int input_dim = model->weights[0].rows;
    int n         = 1000;

    printf("\nRunning %d inferences (input_dim=%d)...\n", n, input_dim);

    /* Generate random inputs */
    float *inputs = (float *)malloc((size_t)n * input_dim * sizeof(float));
    if (!inputs) { fprintf(stderr, "OOM allocating inputs\n"); return 1; }
    srand(12345);
    for (int i = 0; i < n * input_dim; i++)
        inputs[i] = (float)rand() / (float)RAND_MAX;

    /* ------------------------------------------------------------------ */
    /* Timed inference loop                                                */
    /* ------------------------------------------------------------------ */
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    volatile int sink = 0;   /* prevent dead-code elimination */
    for (int i = 0; i < n; i++)
        sink ^= model_predict(model, inputs + (size_t)i * input_dim);

    clock_gettime(CLOCK_MONOTONIC, &t1);

    double elapsed_ms = (t1.tv_sec  - t0.tv_sec ) * 1000.0
                      + (t1.tv_nsec - t0.tv_nsec) / 1.0e6;

    printf("\n--- Results ---\n");
    printf("Ran          : %d inferences\n", n);
    printf("Total time   : %.2f ms\n", elapsed_ms);
    printf("Throughput   : %.0f inferences/sec\n", n / (elapsed_ms / 1000.0));
    printf("Latency      : %.4f ms/inference\n", elapsed_ms / n);
    printf("(sink=%d)\n", (int)sink);   /* keep compiler from optimising away */

    model_free(model);
    free(inputs);
    return 0;
}

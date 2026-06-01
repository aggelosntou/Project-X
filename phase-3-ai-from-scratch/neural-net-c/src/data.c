/*
 * data.c — Data utility implementations.
 */

#include "data.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <time.h>

void dataset_free(Dataset *ds) {
    if (!ds) return;
    mat_free(ds->X);
    mat_free(ds->y);
}

/* ------------------------------------------------------------------ */
/*  Shuffle                                                             */
/* ------------------------------------------------------------------ */

void dataset_shuffle(Dataset *ds) {
    /* Fisher-Yates shuffle on rows */
    for (int i = ds->n_samples - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        /* Swap row i and row j in X */
        for (int k = 0; k < ds->n_features; k++) {
            float tmp = ds->X->data[i * ds->n_features + k];
            ds->X->data[i * ds->n_features + k] = ds->X->data[j * ds->n_features + k];
            ds->X->data[j * ds->n_features + k] = tmp;
        }
        /* Swap row i and row j in y */
        for (int k = 0; k < ds->n_outputs; k++) {
            float tmp = ds->y->data[i * ds->n_outputs + k];
            ds->y->data[i * ds->n_outputs + k] = ds->y->data[j * ds->n_outputs + k];
            ds->y->data[j * ds->n_outputs + k] = tmp;
        }
    }
}

/* ------------------------------------------------------------------ */
/*  Train/test split                                                    */
/* ------------------------------------------------------------------ */

void train_test_split(const Dataset *ds, float test_fraction, int shuffle,
                      Dataset *train_out, Dataset *test_out) {
    int n = ds->n_samples;
    int n_test  = (int)(n * test_fraction);
    int n_train = n - n_test;

    /* Create shuffled index array */
    int *idx = (int *)malloc((size_t)n * sizeof(int));
    for (int i = 0; i < n; i++) idx[i] = i;
    if (shuffle) {
        for (int i = n - 1; i > 0; i--) {
            int j = rand() % (i + 1);
            int tmp = idx[i]; idx[i] = idx[j]; idx[j] = tmp;
        }
    }

    /* Allocate train/test datasets */
    train_out->n_samples  = n_train;
    train_out->n_features = ds->n_features;
    train_out->n_outputs  = ds->n_outputs;
    train_out->X = mat_new(n_train, ds->n_features);
    train_out->y = mat_new(n_train, ds->n_outputs);

    test_out->n_samples  = n_test;
    test_out->n_features = ds->n_features;
    test_out->n_outputs  = ds->n_outputs;
    test_out->X = mat_new(n_test, ds->n_features);
    test_out->y = mat_new(n_test, ds->n_outputs);

    for (int i = 0; i < n_train; i++) {
        int src = idx[i];
        memcpy(&train_out->X->data[i * ds->n_features],
               &ds->X->data[src * ds->n_features],
               (size_t)ds->n_features * sizeof(float));
        memcpy(&train_out->y->data[i * ds->n_outputs],
               &ds->y->data[src * ds->n_outputs],
               (size_t)ds->n_outputs * sizeof(float));
    }
    for (int i = 0; i < n_test; i++) {
        int src = idx[n_train + i];
        memcpy(&test_out->X->data[i * ds->n_features],
               &ds->X->data[src * ds->n_features],
               (size_t)ds->n_features * sizeof(float));
        memcpy(&test_out->y->data[i * ds->n_outputs],
               &ds->y->data[src * ds->n_outputs],
               (size_t)ds->n_outputs * sizeof(float));
    }

    free(idx);
}

/* ------------------------------------------------------------------ */
/*  Normalization                                                       */
/* ------------------------------------------------------------------ */

void normalize_zscore(Matrix *X, float *mean_out, float *std_out) {
    int N = X->rows, F = X->cols;

    /* Compute per-feature mean */
    for (int j = 0; j < F; j++) {
        float s = 0.0f;
        for (int i = 0; i < N; i++) s += X->data[i * F + j];
        mean_out[j] = s / (float)N;
    }

    /* Compute per-feature std */
    for (int j = 0; j < F; j++) {
        float var = 0.0f;
        for (int i = 0; i < N; i++) {
            float d = X->data[i * F + j] - mean_out[j];
            var += d * d;
        }
        std_out[j] = sqrtf(var / (float)N + 1e-8f);
    }

    /* Normalize in-place */
    normalize_apply(X, mean_out, std_out);
}

void normalize_apply(Matrix *X, const float *mean, const float *std) {
    int N = X->rows, F = X->cols;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < F; j++)
            X->data[i * F + j] = (X->data[i * F + j] - mean[j]) / std[j];
}

/* ------------------------------------------------------------------ */
/*  Mini-batch                                                          */
/* ------------------------------------------------------------------ */

void get_batch(const Dataset *ds, int start_idx, int batch_size,
               Matrix *batch_X, Matrix *batch_y) {
    for (int i = 0; i < batch_size; i++) {
        int src = (start_idx + i) % ds->n_samples;
        memcpy(&batch_X->data[i * ds->n_features],
               &ds->X->data[src * ds->n_features],
               (size_t)ds->n_features * sizeof(float));
        memcpy(&batch_y->data[i * ds->n_outputs],
               &ds->y->data[src * ds->n_outputs],
               (size_t)ds->n_outputs * sizeof(float));
    }
}

/* ------------------------------------------------------------------ */
/*  Synthetic generators                                                */
/* ------------------------------------------------------------------ */

Dataset *make_xor_dataset(void) {
    Dataset *ds = (Dataset *)malloc(sizeof(Dataset));
    ds->n_samples  = 4;
    ds->n_features = 2;
    ds->n_outputs  = 1;

    float X_data[] = {0,0, 0,1, 1,0, 1,1};
    float y_data[] = {0, 1, 1, 0};

    ds->X = mat_from_array(X_data, 4, 2);
    ds->y = mat_from_array(y_data, 4, 1);
    return ds;
}

Dataset *make_blobs(int n_samples, int n_features, float noise) {
    Dataset *ds = (Dataset *)malloc(sizeof(Dataset));
    ds->n_samples  = n_samples;
    ds->n_features = n_features;
    ds->n_outputs  = 1;
    ds->X = mat_new(n_samples, n_features);
    ds->y = mat_new(n_samples, 1);

    /* Class 0: centered at (-1, -1, ...) */
    /* Class 1: centered at (+1, +1, ...) */
    for (int i = 0; i < n_samples; i++) {
        int cls = i % 2;
        float center = cls ? 1.0f : -1.0f;
        for (int j = 0; j < n_features; j++) {
            float r = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * noise;
            ds->X->data[i * n_features + j] = center + r;
        }
        ds->y->data[i] = (float)cls;
    }
    return ds;
}

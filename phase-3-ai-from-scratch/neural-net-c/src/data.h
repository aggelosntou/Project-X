/*
 * data.h — Data utilities: loading, splitting, normalizing, shuffling.
 */

#ifndef DATA_H
#define DATA_H

#include "matrix.h"

/* ------------------------------------------------------------------ */
/*  Dataset struct                                                      */
/* ------------------------------------------------------------------ */

typedef struct {
    Matrix *X;    /* Features: (n_samples × n_features) */
    Matrix *y;    /* Labels:   (n_samples × n_outputs)  */
    int     n_samples;
    int     n_features;
    int     n_outputs;
} Dataset;

/* Free dataset (frees X and y matrices) */
void dataset_free(Dataset *ds);

/* ------------------------------------------------------------------ */
/*  Train/test split                                                    */
/* ------------------------------------------------------------------ */

/* Split ds into train/test.
 * test_fraction: e.g. 0.2 for 80/20 split.
 * Shuffles before splitting if shuffle=1. */
void train_test_split(const Dataset *ds, float test_fraction, int shuffle,
                      Dataset *train_out, Dataset *test_out);

/* ------------------------------------------------------------------ */
/*  Normalization                                                       */
/* ------------------------------------------------------------------ */

/* Z-score normalization: x' = (x - mean) / std
 * Stores mean and std arrays for later use (e.g. normalizing test data).
 * mean and std must be pre-allocated arrays of length n_features. */
void normalize_zscore(Matrix *X, float *mean_out, float *std_out);

/* Apply stored normalization to a new matrix */
void normalize_apply(Matrix *X, const float *mean, const float *std);

/* ------------------------------------------------------------------ */
/*  Mini-batch sampling                                                 */
/* ------------------------------------------------------------------ */

/* Fill batch_X (batch_size × n_features) and batch_y from dataset,
 * starting at index start_idx (wraps around). */
void get_batch(const Dataset *ds, int start_idx, int batch_size,
               Matrix *batch_X, Matrix *batch_y);

/* Shuffle the dataset rows (in-place) */
void dataset_shuffle(Dataset *ds);

/* ------------------------------------------------------------------ */
/*  Synthetic data generators                                           */
/* ------------------------------------------------------------------ */

/* Generate XOR dataset: 4 samples (or n_per_class*4) */
Dataset *make_xor_dataset(void);

/* Generate 2-class blobs for binary classification */
Dataset *make_blobs(int n_samples, int n_features, float noise);

#endif /* DATA_H */

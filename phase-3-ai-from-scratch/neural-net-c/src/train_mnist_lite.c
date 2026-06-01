/*
 * train_mnist_lite.c — Train on a synthetic classification task mimicking MNIST structure.
 *
 * The full MNIST dataset requires binary file parsing (IDX format).
 * This file instead generates a synthetic 10-class classification problem
 * with 784-dimensional binary inputs (like MNIST pixels), demonstrating
 * that the same network architecture and training procedure applies.
 *
 * Architecture: 784 → 128 → 64 → 10 (one-hot targets, argmax for prediction)
 *
 * For real MNIST:
 *   - Download: http://yann.lecun.com/exdb/mnist/
 *   - Parse the IDX binary format (see data.h comments for reference)
 *   - Replace synthetic data with loaded arrays
 */

#include "matrix.h"
#include "nn.h"
#include "data.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define N_TRAIN    1000
#define N_TEST     200
#define N_CLASSES  10
#define N_FEATURES 784
#define BATCH_SIZE 32
#define N_EPOCHS   50

/* Generate synthetic data: class k has feature k*78 set to 1.0 + noise */
static void make_synthetic_mnist(Matrix *X, Matrix *y, int n_samples) {
    for (int i = 0; i < n_samples; i++) {
        int cls = i % N_CLASSES;

        /* Set class-specific features */
        for (int f = 0; f < N_FEATURES; f++) {
            float noise = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
            X->data[i * N_FEATURES + f] = noise;
        }
        /* Strong signal for this class */
        int feat_start = cls * (N_FEATURES / N_CLASSES);
        for (int f = feat_start; f < feat_start + 10 && f < N_FEATURES; f++) {
            X->data[i * N_FEATURES + f] = 1.0f;
        }

        /* One-hot target */
        for (int c = 0; c < N_CLASSES; c++)
            y->data[i * N_CLASSES + c] = (c == cls) ? 1.0f : 0.0f;
    }
}

static int argmax(const float *v, int n) {
    int best = 0;
    for (int i = 1; i < n; i++)
        if (v[i] > v[best]) best = i;
    return best;
}

int main(void) {
    srand((unsigned int)time(NULL));
    printf("=== Synthetic MNIST-like Classification (784→128→64→10) ===\n\n");

    /* Allocate full dataset */
    Matrix *X_train = mat_new(N_TRAIN, N_FEATURES);
    Matrix *y_train = mat_new(N_TRAIN, N_CLASSES);
    Matrix *X_test  = mat_new(N_TEST,  N_FEATURES);
    Matrix *y_test  = mat_new(N_TEST,  N_CLASSES);

    make_synthetic_mnist(X_train, y_train, N_TRAIN);
    make_synthetic_mnist(X_test,  y_test,  N_TEST);

    printf("Generated %d train, %d test samples (%d features, %d classes)\n\n",
           N_TRAIN, N_TEST, N_FEATURES, N_CLASSES);

    /* Network: 784 → 128 (relu) → 64 (relu) → 10 (linear, then argmax) */
    int sizes[] = {N_FEATURES, 128, 64, N_CLASSES};
    Network *net = network_new(sizes, 4, 0.01f, ACT_RELU, BATCH_SIZE);
    network_print_summary(net);

    /* Mini-batch SGD */
    Matrix *batch_X = mat_new(BATCH_SIZE, N_FEATURES);
    Matrix *batch_y = mat_new(BATCH_SIZE, N_CLASSES);

    printf("\nTraining (%d epochs, batch_size=%d)...\n", N_EPOCHS, BATCH_SIZE);

    for (int epoch = 0; epoch < N_EPOCHS; epoch++) {
        float epoch_loss = 0.0f;
        int n_batches = N_TRAIN / BATCH_SIZE;

        for (int b = 0; b < n_batches; b++) {
            get_batch(&(Dataset){X_train, y_train, N_TRAIN, N_FEATURES, N_CLASSES},
                      b * BATCH_SIZE, BATCH_SIZE, batch_X, batch_y);
            network_forward(net, batch_X);
            epoch_loss += network_loss(net, batch_y);
            network_backward_with_input(net, batch_X, batch_y);
            network_update(net);
        }

        if (epoch % 10 == 0 || epoch == N_EPOCHS - 1) {
            /* Compute test accuracy */
            int correct = 0;
            for (int i = 0; i < N_TEST; i++) {
                /* Single-sample forward */
                Matrix *xi = mat_from_array(&X_test->data[i * N_FEATURES], 1, N_FEATURES);
                network_forward(net, xi);
                float *out = net->layers[net->n_layers-1]->activation->data;
                int pred = argmax(out, N_CLASSES);
                int true_cls = argmax(&y_test->data[i * N_CLASSES], N_CLASSES);
                if (pred == true_cls) correct++;
                mat_free(xi);
            }
            printf("  Epoch %3d: train_loss=%.4f, test_acc=%.1f%%\n",
                   epoch, epoch_loss / n_batches, 100.0f * correct / N_TEST);
        }
    }

    mat_free(X_train); mat_free(y_train);
    mat_free(X_test);  mat_free(y_test);
    mat_free(batch_X); mat_free(batch_y);
    network_free(net);
    return 0;
}

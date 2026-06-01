/*
 * train_xor.c — Train a 2-4-1 network to learn XOR via backpropagation.
 *
 * XOR is the canonical non-linear benchmark:
 *   Input (0,0) → 0,  (0,1) → 1,  (1,0) → 1,  (1,1) → 0
 *
 * A single linear layer cannot solve XOR (not linearly separable).
 * A 2-layer network with hidden nonlinearity can:
 *   The hidden layer learns to map the 2D input to a space where XOR is
 *   linearly separable, then the output layer applies a linear boundary.
 *
 * Expected convergence: loss < 0.01 within ~5000 epochs.
 */

#include "matrix.h"
#include "nn.h"
#include "data.h"
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    printf("=== XOR Training (C Neural Network) ===\n\n");

    /* XOR data — 4 samples, 2 features, 1 output */
    float X_data[] = {0,0, 0,1, 1,0, 1,1};
    float y_data[] = {0, 1, 1, 0};

    /* Network: 2 inputs → 4 hidden (tanh) → 1 output (linear)
     * Why 4 hidden units? XOR needs at least 2, but 4 trains more reliably. */
    int sizes[] = {2, 4, 1};
    Network *net = network_new(sizes, 3, 0.1f, ACT_TANH, 4);
    network_print_summary(net);

    /* Create input/target matrices */
    Matrix *X = mat_from_array(X_data, 4, 2);
    Matrix *y = mat_from_array(y_data, 4, 1);

    printf("\nTraining...\n");

    int n_epochs = 10000;
    for (int epoch = 0; epoch < n_epochs; epoch++) {
        /* Forward pass */
        network_forward(net, X);

        /* Compute loss */
        float loss = network_loss(net, y);

        /* Backward pass + update */
        network_backward_with_input(net, X, y);
        network_update(net);

        if (epoch % 1000 == 0 || epoch == n_epochs - 1) {
            printf("  Epoch %5d: loss = %.6f\n", epoch, loss);
        }

        /* Early stopping */
        if (loss < 0.001f) {
            printf("  Converged at epoch %d (loss=%.6f)\n", epoch, loss);
            break;
        }
    }

    /* Evaluate */
    printf("\nFinal predictions:\n");
    network_forward(net, X);
    float *names_x[] = {"(0,0)","(0,1)","(1,0)","(1,1)"};
    float expected[] = {0,1,1,0};
    int correct = 0;
    for (int i = 0; i < 4; i++) {
        float pred = net->layers[net->n_layers-1]->activation->data[i];
        int pred_cls = pred > 0.5f ? 1 : 0;
        int ok = (pred_cls == (int)expected[i]);
        correct += ok;
        printf("  XOR%s = %.0f | pred = %.4f (%s)\n",
               names_x[i], expected[i], pred, ok ? "correct" : "WRONG");
    }
    printf("\nAccuracy: %d/4\n", correct);

    float final_loss = network_loss(net, y);
    printf("Final loss: %.6f\n", final_loss);
    if (final_loss < 0.01f)
        printf("SUCCESS: loss < 0.01\n");
    else
        printf("Note: did not fully converge. Try more epochs or different lr.\n");

    mat_free(X);
    mat_free(y);
    network_free(net);
    return 0;
}

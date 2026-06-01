/*
 * nn.c — Neural network implementation.
 *
 * Backpropagation derivation (matrix form):
 *
 * For layer L (output layer), MSE loss L = (1/N) ||a - target||^2:
 *   delta_L = (2/N)(a_L - target)   [dL/dz_L = dL/da_L * da_L/dz_L]
 *   For linear output: da/dz = I, so delta_L = dL/da_L directly.
 *
 * For hidden layer l:
 *   delta_l = (delta_{l+1} @ W_{l+1}^T) ⊙ f'(z_l)
 *   Interpretation:
 *     delta_{l+1} @ W_{l+1}^T  — propagate error signal backward through W
 *     ⊙ f'(z_l)               — gate by local activation derivative
 *
 * Weight gradients:
 *   dL/dW_l = a_{l-1}^T @ delta_l   (outer product summed over batch)
 *   dL/db_l = sum(delta_l, axis=0)  (average over batch dimension)
 */

#include "nn.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>

/* ------------------------------------------------------------------ */
/*  Utility: Xavier initialization                                      */
/* ------------------------------------------------------------------ */

/*
 * Xavier (Glorot) initialization — Glorot & Bengio, 2010.
 *
 * Goal: keep the variance of activations and gradients roughly the same
 * across all layers.
 *
 * For a layer with n_in inputs and n_out outputs:
 *   Var(W) should satisfy: Var(output) = Var(input)
 *   This requires: n_in * Var(W) = 1 → Var(W) = 1/n_in
 *
 * For both forward and backward pass:
 *   Var(W) = 2 / (n_in + n_out)
 *
 * Uniform Xavier: W ~ U(-limit, limit) where limit = sqrt(6/(n_in+n_out))
 * (The factor 6 comes from Var(U(-a,a)) = a^2/3, so a = sqrt(6/(n_in+n_out)))
 */
static float xavier_uniform(int n_in, int n_out) {
    float limit = sqrtf(6.0f / (float)(n_in + n_out));
    return ((float)rand() / (float)RAND_MAX) * 2.0f * limit - limit;
}

/* ------------------------------------------------------------------ */
/*  Layer                                                               */
/* ------------------------------------------------------------------ */

Layer *layer_new(int n_in, int n_out, ActivationType act, int batch_size) {
    Layer *l = (Layer *)calloc(1, sizeof(Layer));
    l->n_in    = n_in;
    l->n_out   = n_out;
    l->act_type = act;

    /* Allocate and initialize weight matrix */
    l->weights = mat_new(n_in, n_out);
    for (int i = 0; i < n_in * n_out; i++)
        l->weights->data[i] = xavier_uniform(n_in, n_out);

    l->biases      = mat_new(1, n_out);    /* zeros (calloc) */
    l->pre_activation = mat_new(batch_size, n_out);
    l->activation     = mat_new(batch_size, n_out);
    l->delta          = mat_new(batch_size, n_out);
    l->grad_weights   = mat_new(n_in, n_out);
    l->grad_biases    = mat_new(1, n_out);

    return l;
}

void layer_free(Layer *l) {
    if (!l) return;
    mat_free(l->weights);
    mat_free(l->biases);
    mat_free(l->pre_activation);
    mat_free(l->activation);
    mat_free(l->delta);
    mat_free(l->grad_weights);
    mat_free(l->grad_biases);
    free(l);
}

/* Apply the activation function */
static void apply_activation(ActivationType act, const Matrix *z, Matrix *a) {
    int n = z->rows * z->cols;
    switch (act) {
        case ACT_RELU:
            mat_relu(z, a);
            break;
        case ACT_SIGMOID:
            mat_sigmoid(z, a);
            break;
        case ACT_TANH:
            for (int i = 0; i < n; i++)
                a->data[i] = tanhf(z->data[i]);
            break;
        case ACT_LINEAR:
        default:
            memcpy(a->data, z->data, (size_t)n * sizeof(float));
            break;
    }
}

/* Compute element-wise activation derivative f'(z) into grad */
static void activation_derivative(ActivationType act,
                                   const Matrix *z,
                                   const Matrix *a,
                                   Matrix *grad) {
    int n = z->rows * z->cols;
    switch (act) {
        case ACT_RELU:
            for (int i = 0; i < n; i++)
                grad->data[i] = z->data[i] > 0.0f ? 1.0f : 0.0f;
            break;
        case ACT_SIGMOID:
            /* sigma'(z) = sigma(z) * (1 - sigma(z)) = a * (1-a) */
            for (int i = 0; i < n; i++)
                grad->data[i] = a->data[i] * (1.0f - a->data[i]);
            break;
        case ACT_TANH:
            /* tanh'(z) = 1 - tanh^2(z) = 1 - a^2 */
            for (int i = 0; i < n; i++)
                grad->data[i] = 1.0f - a->data[i] * a->data[i];
            break;
        case ACT_LINEAR:
        default:
            for (int i = 0; i < n; i++)
                grad->data[i] = 1.0f;
            break;
    }
}

/* ------------------------------------------------------------------ */
/*  Network                                                             */
/* ------------------------------------------------------------------ */

Network *network_new(const int *layer_sizes, int n_layers, float lr,
                     ActivationType hidden_act, int batch_size) {
    srand((unsigned int)time(NULL));

    Network *net   = (Network *)malloc(sizeof(Network));
    net->n_layers  = n_layers - 1;  /* number of weight layers = n_layer_sizes - 1 */
    net->lr        = lr;
    net->layers    = (Layer **)malloc((size_t)net->n_layers * sizeof(Layer *));

    for (int i = 0; i < net->n_layers; i++) {
        ActivationType act = (i == net->n_layers - 1) ? ACT_LINEAR : hidden_act;
        net->layers[i] = layer_new(layer_sizes[i], layer_sizes[i+1], act, batch_size);
    }
    return net;
}

void network_free(Network *net) {
    if (!net) return;
    for (int i = 0; i < net->n_layers; i++)
        layer_free(net->layers[i]);
    free(net->layers);
    free(net);
}

/* ------------------------------------------------------------------ */
/*  Forward pass                                                        */
/* ------------------------------------------------------------------ */

Matrix *network_forward(Network *net, const Matrix *x) {
    /*
     * For each layer l:
     *   z_l = a_{l-1} @ W_l + b_l     (matrix multiply + broadcast bias)
     *   a_l = f(z_l)                   (element-wise activation)
     *
     * We store z_l in l->pre_activation and a_l in l->activation.
     * These are needed during backprop.
     */
    const Matrix *input = x;

    for (int i = 0; i < net->n_layers; i++) {
        Layer *l = net->layers[i];

        /* Need temporary for the multiply result (pre_activation) */
        /* z = input @ W: input is (batch × n_in), W is (n_in × n_out) */
        mat_mul(input, l->weights, l->pre_activation);
        mat_add_bias(l->pre_activation, l->biases, l->pre_activation);

        /* a = f(z) */
        apply_activation(l->act_type, l->pre_activation, l->activation);

        input = l->activation;  /* Output of this layer is input to next */
    }

    return net->layers[net->n_layers - 1]->activation;
}

/* ------------------------------------------------------------------ */
/*  Backward pass                                                       */
/* ------------------------------------------------------------------ */

void network_backward(Network *net, const Matrix *target) {
    int batch_size = target->rows;
    int L = net->n_layers;

    /*
     * Compute delta for output layer (using MSE loss).
     *
     * MSE: loss = (1/N) * ||a_L - target||^2
     * dL/da_L = (2/N) * (a_L - target)
     *
     * For LINEAR output activation: dL/dz_L = dL/da_L
     * For other activations: dL/dz_L = dL/da_L ⊙ f'(z_L)
     */
    Layer *out_layer = net->layers[L-1];
    float scale = 2.0f / (float)batch_size;

    for (int i = 0; i < batch_size * out_layer->n_out; i++) {
        out_layer->delta->data[i] = scale *
            (out_layer->activation->data[i] - target->data[i]);
    }

    /* Apply activation derivative for output layer if non-linear */
    if (out_layer->act_type != ACT_LINEAR) {
        Matrix *act_deriv = mat_new(batch_size, out_layer->n_out);
        activation_derivative(out_layer->act_type, out_layer->pre_activation,
                              out_layer->activation, act_deriv);
        mat_mul_elementwise(out_layer->delta, act_deriv, out_layer->delta);
        mat_free(act_deriv);
    }

    /* Propagate delta backward through hidden layers */
    for (int i = L - 2; i >= 0; i--) {
        Layer *curr = net->layers[i];
        Layer *next = net->layers[i+1];

        /*
         * delta_i = (delta_{i+1} @ W_{i+1}^T) ⊙ f'(z_i)
         *
         * delta_{i+1} @ W_{i+1}^T:
         *   delta_{i+1}: (batch × n_out_{i+1})
         *   W_{i+1}^T:   (n_out_{i+1} × n_in_{i+1}) = (n_out_{i+1} × n_out_i)
         *   result:      (batch × n_out_i)  ✓
         *
         * Then multiply element-wise by f'(z_i) — this gates the gradient.
         */
        Matrix *wT = mat_new(next->weights->cols, next->weights->rows);
        mat_transpose(next->weights, wT);

        mat_mul(next->delta, wT, curr->delta);
        mat_free(wT);

        /* Gate by activation derivative */
        Matrix *act_deriv = mat_new(batch_size, curr->n_out);
        activation_derivative(curr->act_type, curr->pre_activation,
                              curr->activation, act_deriv);
        mat_mul_elementwise(curr->delta, act_deriv, curr->delta);
        mat_free(act_deriv);
    }

    /* Compute weight and bias gradients for all layers */
    const Matrix *prev_activation = NULL;
    for (int i = 0; i < L; i++) {
        Layer *l = net->layers[i];

        /*
         * dL/dW_l = a_{l-1}^T @ delta_l
         *   a_{l-1}: (batch × n_in), transposed → (n_in × batch)
         *   delta_l: (batch × n_out)
         *   result:  (n_in × n_out)  ✓
         *
         * This is the outer product summed over the batch.
         * Each sample contributes one rank-1 outer product to dL/dW.
         */
        if (i == 0) {
            /* For the first layer, prev_activation is the input x.
             * We store it in layer 0's pre_activation before sub in nn.
             * Actually let's handle via a passed parameter approach — see train code. */
            /* Handled by setting prev_act pointer at call site */
            prev_activation = NULL; /* set externally in train code */
        }

        /* We need the previous layer's activation as input to this layer */
        /* For layer 0 we use the network's stored input (set in train_xor.c) */
    }
    (void)prev_activation; /* suppress unused warning for now */
}

/* Recompute backward pass properly, taking the input explicitly */
void network_backward_with_input(Network *net, const Matrix *x, const Matrix *target) {
    int batch_size = target->rows;
    int L = net->n_layers;

    /* Output layer delta */
    Layer *out_layer = net->layers[L-1];
    float scale = 2.0f / (float)batch_size;
    for (int i = 0; i < batch_size * out_layer->n_out; i++) {
        out_layer->delta->data[i] = scale *
            (out_layer->activation->data[i] - target->data[i]);
    }
    if (out_layer->act_type != ACT_LINEAR) {
        Matrix *ad = mat_new(batch_size, out_layer->n_out);
        activation_derivative(out_layer->act_type, out_layer->pre_activation,
                              out_layer->activation, ad);
        mat_mul_elementwise(out_layer->delta, ad, out_layer->delta);
        mat_free(ad);
    }

    /* Hidden layer deltas (backward) */
    for (int i = L - 2; i >= 0; i--) {
        Layer *curr = net->layers[i];
        Layer *next = net->layers[i+1];
        Matrix *wT  = mat_new(next->weights->cols, next->weights->rows);
        mat_transpose(next->weights, wT);
        mat_mul(next->delta, wT, curr->delta);
        mat_free(wT);

        Matrix *ad = mat_new(batch_size, curr->n_out);
        activation_derivative(curr->act_type, curr->pre_activation,
                              curr->activation, ad);
        mat_mul_elementwise(curr->delta, ad, curr->delta);
        mat_free(ad);
    }

    /* Compute gradients */
    for (int i = 0; i < L; i++) {
        Layer *l = net->layers[i];
        const Matrix *a_prev = (i == 0) ? x : net->layers[i-1]->activation;

        /* grad_W = a_prev^T @ delta */
        Matrix *aT = mat_new(a_prev->cols, a_prev->rows);
        mat_transpose(a_prev, aT);
        mat_mul(aT, l->delta, l->grad_weights);
        mat_free(aT);

        /* grad_b = mean(delta, axis=0) */
        memset(l->grad_biases->data, 0, (size_t)l->n_out * sizeof(float));
        for (int b = 0; b < batch_size; b++)
            for (int j = 0; j < l->n_out; j++)
                l->grad_biases->data[j] += l->delta->data[b * l->n_out + j];
        float inv_batch = 1.0f / (float)batch_size;
        for (int j = 0; j < l->n_out; j++)
            l->grad_biases->data[j] *= inv_batch;
    }
}

/* ------------------------------------------------------------------ */
/*  SGD update                                                          */
/* ------------------------------------------------------------------ */

void network_update(Network *net) {
    /* theta = theta - lr * grad */
    for (int i = 0; i < net->n_layers; i++) {
        Layer *l = net->layers[i];
        int nw = l->n_in * l->n_out;
        for (int j = 0; j < nw; j++)
            l->weights->data[j] -= net->lr * l->grad_weights->data[j];
        for (int j = 0; j < l->n_out; j++)
            l->biases->data[j] -= net->lr * l->grad_biases->data[j];
    }
}

/* ------------------------------------------------------------------ */
/*  Loss                                                                */
/* ------------------------------------------------------------------ */

float network_loss(Network *net, const Matrix *target) {
    Matrix *out = net->layers[net->n_layers - 1]->activation;
    int n = out->rows * out->cols;
    float loss = 0.0f;
    for (int i = 0; i < n; i++) {
        float diff = out->data[i] - target->data[i];
        loss += diff * diff;
    }
    return loss / (float)n;
}

/* ------------------------------------------------------------------ */
/*  Debug                                                               */
/* ------------------------------------------------------------------ */

void network_print_summary(const Network *net) {
    printf("Network (%d weight layers, lr=%.4f):\n", net->n_layers, net->lr);
    for (int i = 0; i < net->n_layers; i++) {
        Layer *l = net->layers[i];
        const char *act_names[] = {"linear", "relu", "sigmoid", "tanh"};
        printf("  Layer %d: %d -> %d  (%s)\n",
               i, l->n_in, l->n_out, act_names[l->act_type]);
    }
}

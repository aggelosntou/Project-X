/*
 * nn.h — Complete neural network in C: forward pass, backprop, SGD update.
 *
 * Architecture: fully-connected (dense) network with configurable depth and
 * width.  Supports ReLU and sigmoid activations.
 *
 * Mathematical foundation:
 *   Forward:  z_l = a_{l-1} W_l + b_l,  a_l = f(z_l)
 *   Backward: delta_L = dL/dz_L
 *             delta_l = (delta_{l+1} W_{l+1}^T) ⊙ f'(z_l)
 *             dL/dW_l = a_{l-1}^T delta_l
 *             dL/db_l = sum(delta_l, rows)
 */

#ifndef NN_H
#define NN_H

#include "matrix.h"

/* ------------------------------------------------------------------ */
/*  Activation function enum                                            */
/* ------------------------------------------------------------------ */

typedef enum {
    ACT_LINEAR  = 0,
    ACT_RELU    = 1,
    ACT_SIGMOID = 2,
    ACT_TANH    = 3
} ActivationType;

/* ------------------------------------------------------------------ */
/*  Layer struct                                                        */
/* ------------------------------------------------------------------ */

/*
 * A Layer stores everything needed for forward AND backward passes.
 * We must cache pre-activations (z) and post-activations (a) during
 * forward so that backward can compute gradients correctly.
 *
 * Why store pre-activations?
 *   The backprop formula for delta_l requires f'(z_l), not f'(a_l).
 *   For ReLU: f'(z) = 1 if z > 0, else 0 — we need to know which z were positive.
 *   For sigmoid: f'(z) = σ(z)(1-σ(z)) — we could use a_l here, but
 *   it's cleaner to store z explicitly.
 */
typedef struct {
    Matrix         *weights;        /* W: (n_in × n_out)  */
    Matrix         *biases;         /* b: (1 × n_out)     */
    Matrix         *pre_activation; /* z = xW + b: (batch × n_out) */
    Matrix         *activation;     /* a = f(z): (batch × n_out)   */
    Matrix         *delta;          /* δ = dL/dz: (batch × n_out)  */
    Matrix         *grad_weights;   /* dL/dW: (n_in × n_out) */
    Matrix         *grad_biases;    /* dL/db: (1 × n_out)    */
    int             n_in;
    int             n_out;
    ActivationType  act_type;
} Layer;

/* ------------------------------------------------------------------ */
/*  Network struct                                                      */
/* ------------------------------------------------------------------ */

typedef struct {
    Layer **layers;     /* Array of pointers to layers */
    int     n_layers;
    float   lr;         /* Learning rate */
} Network;

/* ------------------------------------------------------------------ */
/*  Layer operations                                                    */
/* ------------------------------------------------------------------ */

/* Allocate a layer with Xavier-initialized weights.
 * Xavier init: W ~ Uniform(-sqrt(6/(n_in+n_out)), sqrt(6/(n_in+n_out)))
 * This preserves variance of activations across layers.            */
Layer *layer_new(int n_in, int n_out, ActivationType act, int batch_size);
void   layer_free(Layer *l);

/* ------------------------------------------------------------------ */
/*  Network operations                                                  */
/* ------------------------------------------------------------------ */

/* Create network from layer size array.
 * layer_sizes[0] = input dimension
 * layer_sizes[1..n_layers] = hidden/output sizes
 * Last layer uses linear activation; hidden layers use 'hidden_act'. */
Network *network_new(const int *layer_sizes, int n_layers, float lr,
                     ActivationType hidden_act, int batch_size);

void network_free(Network *net);

/* Forward pass: compute activations for all layers given input x.
 * x shape: (batch_size × n_in).
 * Returns pointer to the final layer's activation (owned by net). */
Matrix *network_forward(Network *net, const Matrix *x);

/* Backward pass: given the target output, compute gradients.
 * Uses MSE loss: L = (1/N) * ||a_L - target||^2_F */
void network_backward(Network *net, const Matrix *target);

/* SGD update: W_l = W_l - lr * dL/dW_l  for all layers */
void network_update(Network *net);

/* MSE loss between final activation and target */
float network_loss(Network *net, const Matrix *target);

/* Save/print summary */
void network_print_summary(const Network *net);

#endif /* NN_H */

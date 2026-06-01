/*
 * attention_naive.cu
 * ==================
 * Naive multi-head attention in CUDA.
 *
 * Shapes:  Q, K, V, Out — (batch, heads, seq_len, d_head)  stored contiguously.
 *
 * Three separate kernel launches:
 *   1. qkt_kernel      — computes scores[b,h,i,j] = dot(Q[b,h,i], K[b,h,j]) / sqrt(d)
 *   2. softmax_kernel  — row-wise softmax of scores[b,h,i,:]
 *   3. av_kernel       — out[b,h,i,k] = sum_j scores[b,h,i,j] * V[b,h,j,k]
 *
 * Memory cost: O(batch * heads * seq_len^2) for the scores tensor.
 * At seq_len=1024, heads=8, batch=1, float32: 8 * 1024^2 * 4 = 32 MB.
 *
 * Compile:
 *   nvcc -O2 -arch=sm_70 -std=c++14 -o naive_test src/attention_naive.cu
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* -------------------------------------------------------------------------
 * Error-checking macro
 * ---------------------------------------------------------------------- */
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t _e = (call);                                            \
        if (_e != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s:%d — %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(_e));             \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

/* =========================================================================
 * Kernel 1: Q @ K^T / sqrt(d_head)
 *
 * Grid:  (batch, heads, ceil(seq_len / BLOCK) * ceil(seq_len / BLOCK))
 *        — use a flat blockIdx.x for (i,j) pairs to allow large seq_len.
 * Block: (BLOCK * BLOCK,)  — 1-D block of BLOCK*BLOCK threads.
 *
 * Each thread computes one score[b, h, i, j].
 * ====================================================================== */
#define BLOCK 16

__global__ void qkt_kernel(
    const float * __restrict__ Q,
    const float * __restrict__ K,
    float       * __restrict__ scores,
    int batch, int heads, int seq_len, int d_head,
    float scale)
{
    int bh    = blockIdx.z;                      /* combined batch*heads index */
    int b     = bh / heads;
    int h     = bh % heads;
    int i     = blockIdx.y * BLOCK + threadIdx.y; /* query position */
    int j     = blockIdx.x * BLOCK + threadIdx.x; /* key   position */

    if (i >= seq_len || j >= seq_len) return;

    /* Pointer offsets:  [b, h, pos, :] = base + (b*heads + h) * seq_len * d_head + pos * d_head */
    const float *qi = Q + ((b * heads + h) * seq_len + i) * d_head;
    const float *kj = K + ((b * heads + h) * seq_len + j) * d_head;

    float dot = 0.0f;
    for (int d = 0; d < d_head; d++)
        dot += qi[d] * kj[d];

    int out_idx = ((b * heads + h) * seq_len + i) * seq_len + j;
    scores[out_idx] = dot * scale;
}

/* =========================================================================
 * Kernel 2: row-wise softmax
 *
 * One thread per row [b, h, i, :].
 * Uses two passes: (1) find max, (2) exp + sum, (3) divide.
 * For large seq_len a proper parallel reduction is needed; here we do it
 * sequentially within the thread for clarity.
 * ====================================================================== */
__global__ void softmax_kernel(
    float * __restrict__ scores,
    int batch, int heads, int seq_len)
{
    int bh = blockIdx.y;
    int b  = bh / heads;
    int h  = bh % heads;
    int i  = blockIdx.x * blockDim.x + threadIdx.x;  /* query row */

    if (i >= seq_len) return;

    float *row = scores + ((b * heads + h) * seq_len + i) * seq_len;

    /* Pass 1: max */
    float mx = row[0];
    for (int j = 1; j < seq_len; j++)
        if (row[j] > mx) mx = row[j];

    /* Pass 2: exp(x - max) and sum */
    float sum = 0.0f;
    for (int j = 0; j < seq_len; j++) {
        row[j] = expf(row[j] - mx);
        sum   += row[j];
    }

    /* Pass 3: normalise */
    float inv_sum = 1.0f / sum;
    for (int j = 0; j < seq_len; j++)
        row[j] *= inv_sum;
}

/* =========================================================================
 * Kernel 3: softmax(scores) @ V
 *
 * out[b,h,i,k] = sum_j scores[b,h,i,j] * V[b,h,j,k]
 * One thread per (i, k) pair.
 * ====================================================================== */
__global__ void av_kernel(
    const float * __restrict__ scores,
    const float * __restrict__ V,
    float       * __restrict__ out,
    int batch, int heads, int seq_len, int d_head)
{
    int bh = blockIdx.z;
    int b  = bh / heads;
    int h  = bh % heads;
    int i  = blockIdx.y * BLOCK + threadIdx.y;
    int k  = blockIdx.x * BLOCK + threadIdx.x;

    if (i >= seq_len || k >= d_head) return;

    const float *s_row = scores + ((b * heads + h) * seq_len + i) * seq_len;
    const float *V_bh  = V      +  (b * heads + h) * seq_len * d_head;

    float acc = 0.0f;
    for (int j = 0; j < seq_len; j++)
        acc += s_row[j] * V_bh[j * d_head + k];

    out[((b * heads + h) * seq_len + i) * d_head + k] = acc;
}

/* =========================================================================
 * Host-side wrapper
 * ====================================================================== */

/*
 * naive_attention — single-head or multi-head attention.
 *
 * All pointers must be device pointers already allocated.
 * scores_d must have size batch * heads * seq_len * seq_len * sizeof(float).
 *
 * Returns total kernel time in milliseconds (measured with CUDA events).
 */
float naive_attention(
    const float *Q_d, const float *K_d, const float *V_d,
    float *scores_d, float *out_d,
    int batch, int heads, int seq_len, int d_head)
{
    float scale = 1.0f / sqrtf((float)d_head);

    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));
    CUDA_CHECK(cudaEventRecord(ev_start));

    /* --- Kernel 1: Q @ K^T --- */
    {
        dim3 block(BLOCK, BLOCK, 1);
        dim3 grid(
            (seq_len + BLOCK - 1) / BLOCK,  /* j tiles */
            (seq_len + BLOCK - 1) / BLOCK,  /* i tiles */
            batch * heads);                  /* batch × heads */
        qkt_kernel<<<grid, block>>>(Q_d, K_d, scores_d,
                                    batch, heads, seq_len, d_head, scale);
    }

    /* --- Kernel 2: softmax --- */
    {
        dim3 block(32, 1, 1);
        dim3 grid(
            (seq_len + 31) / 32,
            batch * heads,
            1);
        softmax_kernel<<<grid, block>>>(scores_d, batch, heads, seq_len);
    }

    /* --- Kernel 3: scores @ V --- */
    {
        dim3 block(BLOCK, BLOCK, 1);
        dim3 grid(
            (d_head  + BLOCK - 1) / BLOCK,
            (seq_len + BLOCK - 1) / BLOCK,
            batch * heads);
        av_kernel<<<grid, block>>>(scores_d, V_d, out_d,
                                   batch, heads, seq_len, d_head);
    }

    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
    return ms;
}

/* =========================================================================
 * Stand-alone test (only compiled when this file is the main translation unit)
 * ====================================================================== */
#ifndef BENCHMARK_MAIN

int main(void)
{
    int batch = 1, heads = 4, seq_len = 64, d_head = 32;
    size_t qkv_elems  = (size_t)batch * heads * seq_len * d_head;
    size_t scr_elems  = (size_t)batch * heads * seq_len * seq_len;

    printf("Naive attention: batch=%d heads=%d seq=%d d=%d\n",
           batch, heads, seq_len, d_head);
    printf("Score matrix: %.2f MB\n",
           scr_elems * sizeof(float) / 1024.0f / 1024.0f);

    float *Q_d, *K_d, *V_d, *S_d, *O_d;
    CUDA_CHECK(cudaMalloc(&Q_d, qkv_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&K_d, qkv_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&V_d, qkv_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&S_d, scr_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&O_d, qkv_elems * sizeof(float)));

    /* Fill with small random values from host */
    {
        float *h = (float *)malloc(qkv_elems * sizeof(float));
        for (size_t i = 0; i < qkv_elems; i++) h[i] = (float)rand() / RAND_MAX * 0.1f;
        CUDA_CHECK(cudaMemcpy(Q_d, h, qkv_elems * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(K_d, h, qkv_elems * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(V_d, h, qkv_elems * sizeof(float), cudaMemcpyHostToDevice));
        free(h);
    }

    /* Warm-up */
    naive_attention(Q_d, K_d, V_d, S_d, O_d, batch, heads, seq_len, d_head);
    CUDA_CHECK(cudaDeviceSynchronize());

    float ms = naive_attention(Q_d, K_d, V_d, S_d, O_d, batch, heads, seq_len, d_head);
    printf("Kernel time: %.4f ms\n", ms);

    cudaFree(Q_d); cudaFree(K_d); cudaFree(V_d); cudaFree(S_d); cudaFree(O_d);
    return 0;
}

#endif /* BENCHMARK_MAIN */

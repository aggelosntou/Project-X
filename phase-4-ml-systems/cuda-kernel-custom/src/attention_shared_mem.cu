/*
 * attention_shared_mem.cu
 * =======================
 * Tiled attention using shared memory.
 *
 * Optimization over naive:
 *   - Load TILE × d_head blocks of Q and K into shared memory once.
 *   - Each thread block computes a TILE × TILE sub-block of the score matrix.
 *   - Reduces global memory reads from O(N^2 * d) down to
 *     O(N^2 / TILE) loads per element (amortised across the tile).
 *
 * Still materialises the full N×N score matrix (like naive).
 * See flash_attention.cu for the version that avoids this.
 *
 * Compile:
 *   nvcc -O2 -arch=sm_70 -std=c++14 -o sharedmem_test src/attention_shared_mem.cu
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t _e = (call);                                            \
        if (_e != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s:%d — %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(_e));             \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

/* Tile dimension along sequence length */
#define TILE 16

/* =========================================================================
 * Kernel 1: Tiled Q @ K^T / sqrt(d_head)
 *
 * Each thread block (TILE × TILE threads) is responsible for a
 * (TILE × TILE) block of scores[b,h,:,:].
 *
 * Tiling along d_head:
 *   Load Qs[TILE][TILE] from Q and Ks[TILE][TILE] from K.
 *   Accumulate partial dot-products across D_HEAD/TILE tiles.
 * ====================================================================== */
__global__ void tiled_qkt_kernel(
    const float * __restrict__ Q,
    const float * __restrict__ K,
    float       * __restrict__ scores,
    int seq_len, int d_head,
    float scale)
{
    /* Each block handles one (batch, head) pair encoded in blockIdx.z */
    int bh = blockIdx.z;

    int row = blockIdx.y * TILE + threadIdx.y;  /* query row  (i) */
    int col = blockIdx.x * TILE + threadIdx.x;  /* key   col  (j) */

    /* Shared memory tiles for a chunk of Q and K along d_head */
    __shared__ float Qs[TILE][TILE];
    __shared__ float Ks[TILE][TILE];

    float acc = 0.0f;

    /* Sweep through d_head in steps of TILE */
    int n_tiles = (d_head + TILE - 1) / TILE;
    for (int t = 0; t < n_tiles; t++) {
        int d_q = t * TILE + threadIdx.x;  /* d index loaded by this thread for Q */
        int d_k = t * TILE + threadIdx.y;  /* d index loaded by this thread for K */

        /* Load Q[bh, row, d_q] → Qs[threadIdx.y][threadIdx.x] */
        Qs[threadIdx.y][threadIdx.x] =
            (row < seq_len && d_q < d_head)
                ? Q[bh * seq_len * d_head + row * d_head + d_q]
                : 0.0f;

        /* Load K[bh, col, d_k] → Ks[threadIdx.y][threadIdx.x]
         * Note: K is transposed conceptually — we load K[col, d] not K[row, d]. */
        Ks[threadIdx.y][threadIdx.x] =
            (col < seq_len && d_k < d_head)
                ? K[bh * seq_len * d_head + col * d_head + d_k]
                : 0.0f;

        __syncthreads();

        /* Accumulate partial dot product for this tile */
        for (int k = 0; k < TILE; k++)
            acc += Qs[threadIdx.y][k] * Ks[k][threadIdx.x];

        __syncthreads();
    }

    if (row < seq_len && col < seq_len)
        scores[bh * seq_len * seq_len + row * seq_len + col] = acc * scale;
}

/* =========================================================================
 * Kernel 2: row-wise softmax (same as naive — softmax is memory-bound
 * regardless, the bottleneck is Q@K^T)
 * ====================================================================== */
__global__ void softmax_kernel_shared(
    float * __restrict__ scores,
    int seq_len)
{
    int bh = blockIdx.y;
    int i  = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= seq_len) return;

    float *row = scores + (bh * seq_len + i) * seq_len;

    float mx = row[0];
    for (int j = 1; j < seq_len; j++)
        if (row[j] > mx) mx = row[j];

    float sum = 0.0f;
    for (int j = 0; j < seq_len; j++) {
        row[j] = expf(row[j] - mx);
        sum   += row[j];
    }
    float inv = 1.0f / sum;
    for (int j = 0; j < seq_len; j++)
        row[j] *= inv;
}

/* =========================================================================
 * Kernel 3: Tiled scores @ V
 *
 * Tile the j (key) dimension: load TILE×d_head blocks of V and
 * TILE×TILE blocks of scores into shared memory.
 * ====================================================================== */
__global__ void tiled_av_kernel(
    const float * __restrict__ scores,
    const float * __restrict__ V,
    float       * __restrict__ out,
    int seq_len, int d_head)
{
    int bh  = blockIdx.z;
    int row = blockIdx.y * TILE + threadIdx.y;  /* query index i */
    int col = blockIdx.x * TILE + threadIdx.x;  /* output head dim k */

    __shared__ float Ss[TILE][TILE];  /* scores[i, j] tile */
    __shared__ float Vs[TILE][TILE];  /* V[j, k] tile */

    float acc = 0.0f;

    int n_tiles = (seq_len + TILE - 1) / TILE;
    for (int t = 0; t < n_tiles; t++) {
        int j_s = t * TILE + threadIdx.x;  /* j index for scores tile */
        int j_v = t * TILE + threadIdx.y;  /* j index for V tile */

        Ss[threadIdx.y][threadIdx.x] =
            (row < seq_len && j_s < seq_len)
                ? scores[bh * seq_len * seq_len + row * seq_len + j_s]
                : 0.0f;

        Vs[threadIdx.y][threadIdx.x] =
            (j_v < seq_len && col < d_head)
                ? V[bh * seq_len * d_head + j_v * d_head + col]
                : 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE; k++)
            acc += Ss[threadIdx.y][k] * Vs[k][threadIdx.x];

        __syncthreads();
    }

    if (row < seq_len && col < d_head)
        out[bh * seq_len * d_head + row * d_head + col] = acc;
}

/* =========================================================================
 * Host-side wrapper
 * ====================================================================== */

/*
 * shared_mem_attention
 *
 * All device pointers must be pre-allocated.
 * Q, K, V, out: (batch * heads) × seq_len × d_head  (batch and heads flattened).
 * scores:       (batch * heads) × seq_len × seq_len.
 * Returns kernel time in ms.
 */
float shared_mem_attention(
    const float *Q_d, const float *K_d, const float *V_d,
    float *scores_d, float *out_d,
    int batch, int heads, int seq_len, int d_head)
{
    float scale = 1.0f / sqrtf((float)d_head);
    int   bh    = batch * heads;

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);

    /* Kernel 1: tiled Q @ K^T */
    {
        dim3 block(TILE, TILE, 1);
        dim3 grid(
            (seq_len + TILE - 1) / TILE,
            (seq_len + TILE - 1) / TILE,
            bh);
        tiled_qkt_kernel<<<grid, block>>>(Q_d, K_d, scores_d,
                                          seq_len, d_head, scale);
    }

    /* Kernel 2: softmax */
    {
        dim3 block(32, 1, 1);
        dim3 grid((seq_len + 31) / 32, bh, 1);
        softmax_kernel_shared<<<grid, block>>>(scores_d, seq_len);
    }

    /* Kernel 3: tiled scores @ V */
    {
        dim3 block(TILE, TILE, 1);
        dim3 grid(
            (d_head  + TILE - 1) / TILE,
            (seq_len + TILE - 1) / TILE,
            bh);
        tiled_av_kernel<<<grid, block>>>(scores_d, V_d, out_d,
                                         seq_len, d_head);
    }

    cudaEventRecord(t1);
    cudaEventSynchronize(t1);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, t0, t1);
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    return ms;
}

/* =========================================================================
 * Stand-alone test
 * ====================================================================== */
#ifndef BENCHMARK_MAIN

int main(void)
{
    int batch = 1, heads = 4, seq_len = 64, d_head = 32;
    int bh = batch * heads;
    size_t qkv_sz  = (size_t)bh * seq_len * d_head * sizeof(float);
    size_t scr_sz  = (size_t)bh * seq_len * seq_len * sizeof(float);

    printf("Shared-mem attention: batch=%d heads=%d seq=%d d=%d\n",
           batch, heads, seq_len, d_head);

    float *Q_d, *K_d, *V_d, *S_d, *O_d;
    cudaMalloc(&Q_d, qkv_sz);
    cudaMalloc(&K_d, qkv_sz);
    cudaMalloc(&V_d, qkv_sz);
    cudaMalloc(&S_d, scr_sz);
    cudaMalloc(&O_d, qkv_sz);

    /* Warm-up */
    shared_mem_attention(Q_d, K_d, V_d, S_d, O_d, batch, heads, seq_len, d_head);
    cudaDeviceSynchronize();

    float ms = shared_mem_attention(Q_d, K_d, V_d, S_d, O_d, batch, heads, seq_len, d_head);
    printf("Kernel time: %.4f ms\n", ms);

    cudaFree(Q_d); cudaFree(K_d); cudaFree(V_d); cudaFree(S_d); cudaFree(O_d);
    return 0;
}

#endif /* BENCHMARK_MAIN */

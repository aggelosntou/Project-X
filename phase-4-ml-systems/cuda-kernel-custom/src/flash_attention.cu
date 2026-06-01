/*
 * flash_attention.cu
 * ==================
 * Simplified FlashAttention (Dao et al., NeurIPS 2022).
 *
 * Key idea: never materialise the full N×N score matrix.
 *   Instead, process queries in blocks and iterate over all key/value blocks.
 *   Maintain running statistics (max_i, sum_i, O_i) per query block and
 *   update them incrementally using the log-sum-exp (online softmax) trick.
 *
 * Memory complexity:  O(seq_len * d_head)   instead of  O(seq_len^2)
 * IO complexity:      O(seq_len * d_head * seq_len / M)  where M = SRAM size
 *
 * This implementation handles a single (batch * heads) "head" per kernel
 * launch block.  For a full implementation all heads run concurrently.
 *
 * Notation (following the FlashAttention paper):
 *   Br = block size for queries (rows)
 *   Bc = block size for keys/values (columns)
 *   Tr = ceil(seq_len / Br)  — number of query blocks
 *   Tc = ceil(seq_len / Bc)  — number of key/value blocks
 *
 * Compile:
 *   nvcc -O2 -arch=sm_70 -std=c++14 -o flash_test src/flash_attention.cu
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
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

#define Br_DEFAULT 32
#define Bc_DEFAULT 32

/* =========================================================================
 * Flash Attention Forward Kernel
 *
 * One thread block handles ONE query block (Br queries) for ONE head.
 *   - blockIdx.x = query block index (0 … Tr-1)
 *   - blockIdx.y = batch * heads index
 *
 * Threads: one thread per query-row within the block (threadIdx.x in [0, Br)).
 *
 * Shared memory layout (dynamic allocation via extern __shared__):
 *   Kj [Bc * d_head]  — current key block
 *   Vj [Bc * d_head]  — current value block
 *   Qi [Br * d_head]  — current query block (loaded once per block launch)
 *   Oi [Br * d_head]  — accumulator for output
 *   mi [Br]           — running row-max
 *   li [Br]           — running row-sum (normalisation factor)
 * ====================================================================== */
__global__ void flash_attention_forward(
    const float * __restrict__ Q,
    const float * __restrict__ K,
    const float * __restrict__ V,
    float       * __restrict__ O,
    int seq_len, int d_head,
    float scale,
    int Br, int Bc)
{
    int bh          = blockIdx.y;    /* head index (batch * heads flattened) */
    int query_block = blockIdx.x;    /* which block of Br queries we handle  */
    int ti          = threadIdx.x;   /* thread within block: 0 … Br-1        */

    /* ------------------------------------------------------------------ *
     * Shared memory partitioning                                           *
     * ------------------------------------------------------------------ */
    extern __shared__ float shmem[];
    float *Kj = shmem;                          /* [Bc * d_head]  */
    float *Vj = Kj + Bc * d_head;              /* [Bc * d_head]  */
    float *Qi = Vj + Bc * d_head;              /* [Br * d_head]  */
    float *Oi = Qi + Br * d_head;              /* [Br * d_head]  */
    float *mi = Oi + Br * d_head;              /* [Br]           */
    float *li = mi + Br;                        /* [Br]           */

    /* ------------------------------------------------------------------ *
     * Base pointers for this head                                         *
     * ------------------------------------------------------------------ */
    const float *Q_head = Q + (size_t)bh * seq_len * d_head;
    const float *K_head = K + (size_t)bh * seq_len * d_head;
    const float *V_head = V + (size_t)bh * seq_len * d_head;
    float       *O_head = O + (size_t)bh * seq_len * d_head;

    int i_start = query_block * Br;  /* global start of this query block */

    /* ------------------------------------------------------------------ *
     * Load query block Qi into shared memory                               *
     * Each thread loads one row: Qi[ti, :] = Q[i_start+ti, :]           *
     * ------------------------------------------------------------------ */
    int global_i = i_start + ti;
    for (int d = 0; d < d_head; d++) {
        Qi[ti * d_head + d] =
            (global_i < seq_len) ? Q_head[global_i * d_head + d] : 0.0f;
    }

    /* Initialise accumulators */
    mi[ti] = -FLT_MAX;
    li[ti] = 0.0f;
    for (int d = 0; d < d_head; d++)
        Oi[ti * d_head + d] = 0.0f;

    __syncthreads();

    /* ------------------------------------------------------------------ *
     * Iterate over key/value blocks                                       *
     * ------------------------------------------------------------------ */
    int Tc = (seq_len + Bc - 1) / Bc;
    for (int j_block = 0; j_block < Tc; j_block++) {
        int j_start = j_block * Bc;

        /* --- Load Kj and Vj cooperatively --- */
        /* All Br threads load rows of Kj/Vj in round-robin */
        for (int row = ti; row < Bc; row += Br) {
            int global_j = j_start + row;
            for (int d = 0; d < d_head; d++) {
                float kval = (global_j < seq_len)
                                 ? K_head[global_j * d_head + d] : 0.0f;
                float vval = (global_j < seq_len)
                                 ? V_head[global_j * d_head + d] : 0.0f;
                Kj[row * d_head + d] = kval;
                Vj[row * d_head + d] = vval;
            }
        }
        __syncthreads();

        /* --- Compute Sij = Qi[ti] @ Kj^T, shape [Bc] (for this query row) --- */
        /* We use local registers to avoid writing Sij to shared mem */
        float old_mi = mi[ti];
        float new_mi = old_mi;

        /* Compute scores and find new row max */
        /* Store in Kj's space (after loading — we have Bc floats free per row).
         * To keep it clean we use a local array. For Bc <= 64 this is fine in
         * registers; for larger Bc you'd need dynamic shared memory. */
        float Sij[Bc_DEFAULT];  /* local score accumulator */
        int actual_bc = min(Bc, seq_len - j_start);

        for (int jj = 0; jj < actual_bc; jj++) {
            float dot = 0.0f;
            for (int d = 0; d < d_head; d++)
                dot += Qi[ti * d_head + d] * Kj[jj * d_head + d];
            Sij[jj] = dot * scale;
            if (Sij[jj] > new_mi) new_mi = Sij[jj];
        }

        /* --- Online softmax update (Algorithm 1 of FlashAttention paper) ---
         *
         * Let:
         *   m_old = old running max
         *   m_new = max(m_old, max(S_block))
         *   l_new = l_old * exp(m_old - m_new) + sum(exp(S_block - m_new))
         *   O_new = O_old * (l_old * exp(m_old - m_new)) / l_new
         *         + sum(exp(S_block - m_new) * V_block) / l_new
         *
         * We defer the final /l_new division to the output write step.
         */

        float rescale  = expf(old_mi - new_mi);
        float new_li   = li[ti] * rescale;

        /* Accumulate exp(Sij - new_mi) * Vj into Oi, also accumulate sum */
        for (int jj = 0; jj < actual_bc; jj++) {
            float p_ij = expf(Sij[jj] - new_mi);
            new_li += p_ij;
            for (int d = 0; d < d_head; d++)
                Oi[ti * d_head + d] = Oi[ti * d_head + d] * rescale
                                    + p_ij * Vj[jj * d_head + d];
        }

        mi[ti] = new_mi;
        li[ti] = new_li;

        __syncthreads();
    }

    /* ------------------------------------------------------------------ *
     * Write output:  O = Oi / li                                          *
     * ------------------------------------------------------------------ */
    if (global_i < seq_len) {
        float inv_l = 1.0f / li[ti];
        for (int d = 0; d < d_head; d++)
            O_head[global_i * d_head + d] = Oi[ti * d_head + d] * inv_l;
    }
}

/* =========================================================================
 * Host-side wrapper
 * ====================================================================== */

/*
 * flash_attention
 *
 * Runs flash_attention_forward for all (batch * heads) heads.
 * Q, K, V, O are device pointers of shape (batch*heads, seq_len, d_head).
 * Returns kernel time in ms.
 */
float flash_attention(
    const float *Q_d, const float *K_d, const float *V_d,
    float *O_d,
    int batch, int heads, int seq_len, int d_head,
    int Br, int Bc)
{
    float scale = 1.0f / sqrtf((float)d_head);
    int   bh    = batch * heads;
    int   Tr    = (seq_len + Br - 1) / Br;

    /* Shared memory per block:
     * Kj[Bc*d] + Vj[Bc*d] + Qi[Br*d] + Oi[Br*d] + mi[Br] + li[Br]  */
    size_t shmem_bytes = (size_t)(2 * Bc + 2 * Br) * d_head * sizeof(float)
                       + (size_t)(2 * Br) * sizeof(float);

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);

    dim3 grid(Tr, bh, 1);
    dim3 block(Br, 1, 1);
    flash_attention_forward<<<grid, block, shmem_bytes>>>(
        Q_d, K_d, V_d, O_d,
        seq_len, d_head, scale,
        Br, Bc);

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
    int batch = 1, heads = 4, seq_len = 128, d_head = 32;
    int Br = 32, Bc = 32;
    int bh = batch * heads;
    size_t qkv_sz = (size_t)bh * seq_len * d_head * sizeof(float);

    printf("FlashAttention: batch=%d heads=%d seq=%d d=%d  Br=%d Bc=%d\n",
           batch, heads, seq_len, d_head, Br, Bc);
    printf("Peak memory:  %.3f MB  (vs naive %.3f MB)\n",
           qkv_sz * 4.0f / 1024.0f / 1024.0f,
           ((float)bh * seq_len * seq_len * sizeof(float)) / 1024.0f / 1024.0f);

    float *Q_d, *K_d, *V_d, *O_d;
    CUDA_CHECK(cudaMalloc(&Q_d, qkv_sz));
    CUDA_CHECK(cudaMalloc(&K_d, qkv_sz));
    CUDA_CHECK(cudaMalloc(&V_d, qkv_sz));
    CUDA_CHECK(cudaMalloc(&O_d, qkv_sz));

    /* Warm-up */
    flash_attention(Q_d, K_d, V_d, O_d, batch, heads, seq_len, d_head, Br, Bc);
    CUDA_CHECK(cudaDeviceSynchronize());

    float ms = flash_attention(Q_d, K_d, V_d, O_d, batch, heads, seq_len, d_head, Br, Bc);
    printf("Kernel time: %.4f ms\n", ms);

    cudaFree(Q_d); cudaFree(K_d); cudaFree(V_d); cudaFree(O_d);
    return 0;
}

#endif /* BENCHMARK_MAIN */

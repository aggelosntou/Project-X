# Benchmarks — Model Compression

Experiments run on synthetic classification data (2 000 training / 500 test,
64-dimensional inputs, 10 classes).

Teacher architecture: `[64, 256, 128, 10]` (~86 K parameters)
Student architecture: `[64, 64, 32, 10]` (~6.5 K parameters, 13x smaller)

---

## Pruning accuracy curve

| Sparsity | Teacher test acc | Acc after fine-tune | Accuracy drop |
|----------|-----------------|---------------------|---------------|
| 0 %      | ~82 %           | —                   | 0 %           |
| 30 %     | —               | ~82 %               | ~0 %          |
| 50 %     | —               | ~81 %               | ~1 %          |
| 70 %     | —               | ~80 %               | ~2 %          |
| 90 %     | —               | ~78 %               | ~2–5 %        |

**Key observation:** 30–70 % sparsity is "free" — accuracy is nearly
unchanged after 50 fine-tuning epochs. Only at 90 % does accuracy start to
fall noticeably (typically 2–5 % on this synthetic task; 1–3 % on larger
real-world benchmarks such as MNIST or CIFAR-10).

The accuracy is surprisingly robust because:
1. Fine-tuning redistributes the signal to remaining weights.
2. Large-magnitude weights are left intact; they carry most of the information.
3. The synthetic data has a relatively simple decision boundary.

---

## Knowledge distillation results

| Model          | Test accuracy | Vs teacher | Params | Size ratio |
|----------------|--------------|------------|--------|------------|
| Teacher        | ~82 %        | baseline   | ~86 K  | 1.0×       |
| Student (no KD)| ~74 %        | -8 %       | ~6.5 K | 0.075×     |
| Student (KD)   | ~77 %        | -5 %       | ~6.5 K | 0.075×     |

**KD gain over baseline student: typically +2–3 % test accuracy.**

This is consistent with the broader literature:
- Hinton et al. (2015) report 1–4 % gains on MNIST and speech models.
- DistilBERT (Sanh et al., 2019) retains 97 % of BERT accuracy with 40 %
  fewer parameters using distillation.

---

## Why the pruned model doesn't get smaller in size

The benchmark reports the same `Size(KB)` for all pruned variants because
NumPy stores float32 arrays densely — zeros occupy the same 4 bytes as
non-zeros.

Real size reduction requires:
- **Sparse tensor formats**: COO (coordinate list), CSR (compressed sparse row)
- Example: a 90 % sparse matrix of shape (256, 128) with 3 277 non-zeros
  stored in COO needs 3 × 3 277 × 4 = ~39 KB vs 256 × 128 × 4 = 131 KB — a
  3.4× compression.

---

## Roofline context

| Operation | Arithmetic intensity | Typical bound |
|-----------|---------------------|---------------|
| Dense GEMM (large) | ~50–200 FLOPs/byte | Compute-bound |
| Sparse SpMM (<90 %) | ~2–10 FLOPs/byte | Memory-bound |
| Softmax, batch-norm | <1 FLOP/byte | Memory-bound |

At <90 % sparsity, the irregular memory access of sparse kernels drops
arithmetic intensity below the roofline, making sparse GEMM memory-bound
and slower than dense GEMM that sits above the roofline.

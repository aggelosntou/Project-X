# Learnings — Model Compression

## 1. Why pruning alone does NOT speed up real inference

When you set 70 % of weights to zero and measure wall-clock time, you will
often find **no speedup** compared to the original dense network. This is
counterintuitive but well-documented.

### The hardware reason

Modern CPUs and GPUs are built around **dense matrix-multiply (GEMM)** kernels.
BLAS libraries (OpenBLAS, MKL, cuBLAS) achieve near-peak floating-point
throughput by packing data into cache-aligned tiles and feeding SIMD/tensor
units at full bandwidth. These routines **do not skip zero elements** —
they multiply and add every entry, zeros included.

For a zero entry to save compute, the hardware needs a **sparse kernel** that:

1. Stores only non-zero positions (e.g. COO, CSR, BCSR formats).
2. Iterates only over those positions.

The overhead of the sparse bookkeeping (index lookups, gather/scatter, irregular
memory access patterns) only pays off at **very high sparsity, typically above
90–95 %**. Below that threshold, the sparse kernel is slower than the
equivalent dense multiply because cache utilisation drops.

NVIDIA's cuSPARSE library begins to outperform cuBLAS only above ~70 % sparsity
for large matrices, and structured block-sparse formats (e.g. 2:4 on Ampere GPUs)
require exactly 50 % sparsity with a specific pattern.

### Practical consequence

Unstructured magnitude pruning as implemented here produces **irregular**
zero patterns. Even at 90 % sparsity, the weight matrix has the same shape
and is multiplied with the same dense GEMM call. Structured pruning (removing
entire rows/columns) produces a **smaller dense matrix** that genuinely reduces
FLOPs and memory bandwidth. That is the reason `structured_prune_channels`
exists alongside `magnitude_prune`.

---

## 2. Why magnitude pruning works

### Empirical observation

Large-magnitude weights generally carry more signal. A weight of 0.001 can
be zeroed with minimal change to the output; a weight of 2.5 cannot.
This is not just a heuristic — there is a mathematical basis: the change in
output caused by removing a weight $w_i$ is proportional to $|w_i|$ times
the incoming activation, and activations have bounded variance after
batch-normalisation.

### The Lottery Ticket Hypothesis

Frankle & Carlin (2019) showed that large randomly-initialised networks
contain **sparse subnetworks** ("winning tickets") that, when trained in
isolation from the original initialisation, reach full accuracy in fewer
steps. Magnitude pruning after full training recovers a close approximation
to these subnetworks.

The implication: most of a network's parameters are redundant from the start.
The training process distributes the signal across many weights, but a small
fraction does most of the actual computation.

---

## 3. Why knowledge distillation works

### Hard labels carry almost no information per example

A one-hot label `[0, 0, 1, 0, 0]` tells the student exactly one bit of
useful information: "this is class 2". Everything else is zero.

### Soft labels encode inter-class similarity

A teacher trained to convergence might output:

```
[0.02, 0.03, 0.85, 0.08, 0.02]
```

This tells the student:
- Class 2 is correct (0.85).
- Class 3 is somewhat similar to class 2 (0.08) — the teacher has seen
  examples where 2 and 3 are confusable.
- Classes 0 and 4 are very different from 2 (0.02 each).

The student is receiving a **rich similarity structure** over the label space,
not just a single-bit correct/wrong signal. Hinton et al. called this
"dark knowledge" — information about the problem structure that the teacher
has learned and that lives in the non-argmax entries of its output.

### Gradient signal

With hard labels, the gradient of CE w.r.t. the logit of the correct class is
$p_c - 1$ and of all other classes it is $p_j$. For a well-trained teacher,
the correct class probability is near 1, so the gradient is almost zero.
The network is barely learning.

With soft labels at temperature T, the distribution is spread out, gradients
for all classes are non-trivial, and the student gets stronger learning signal
per example.

---

## 4. Temperature T and the softening mechanism

The softmax with temperature T is:

```
softmax(z, T)[i] = exp(z_i / T) / sum_j exp(z_j / T)
```

**T = 1**: standard softmax, peaked at the argmax.

**T > 1**: distribution becomes more uniform. For T → ∞ all classes receive
equal probability 1/C, regardless of logit values.

**T < 1**: distribution becomes more peaked (approaches one-hot).

### Why T^2 in the loss formula

When you replace `z` with `z/T`, the gradient of the KL term w.r.t. `z` is
scaled by `1/T` compared to the hard loss gradient. To keep the two terms
on the same scale, the soft loss is multiplied by `T^2`:

```
L_total = alpha * L_hard + (1 - alpha) * T^2 * KL(p_teacher || p_student)
```

The net effect: the factor of `1/T` from the softmax derivative cancels with
one `T`, leaving the combined gradient with both terms at approximately the
same magnitude.

### Practical guidance

- T = 1: no distillation effect (just adding a KL term with the same distribution).
- T = 2–4: typical range, produces clear accuracy improvements on standard benchmarks.
- T > 10: distribution is so uniform that class-discriminative information is lost.

The sweet spot (T = 3–6) allows the teacher to communicate its internal
confidence ordering without flattening it to noise.

---

## 5. When to use which technique

| Goal | Technique |
|------|-----------|
| Reduce memory footprint on disk/RAM | Pruning + sparse storage; quantisation |
| Reduce inference latency on CPU | Structured pruning; distillation → smaller model |
| Reduce inference latency on GPU | Structured pruning; 2:4 sparsity on Ampere |
| Train small model to its full potential | Knowledge distillation |
| Compress further after distillation | Distil → quantise → structured prune |

The most effective production pipeline is usually:
**train large teacher → distil to medium student → quantise student to INT8**.

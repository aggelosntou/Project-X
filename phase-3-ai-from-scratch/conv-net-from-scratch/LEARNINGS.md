# CNN From Scratch — Key Learnings

## 1. What Convolution Computes

Mathematical convolution (signal processing) flips the kernel before sliding:
    (f * g)[n] = sum_k f[k] * g[n - k]

Deep learning layers compute cross-correlation (no flip):
    (f ⋆ g)[i, j] = sum_{p=0}^{kH-1} sum_{q=0}^{kW-1} f[i+p, j+q] * g[p, q]

These are equivalent if the kernel is learned — the network learns the "flipped"
version just as easily. Calling it "convolution" is historical convention.

For a multi-channel input with C input channels and F output filters:
    out[n, f, i, j] = sum_{c=0}^{C-1} sum_{p,q} x[n, c, i+p, j+q] * W[f, c, p, q] + b[f]

Each filter W[f] produces one feature map by sliding over all input channels.

## 2. Why Weight Sharing Is the Key Insight

In a fully-connected network applied to a 28×28 image:
    First layer alone: 784 × 256 = 200,704 parameters
    And these weights don't transfer — a "horizontal edge detector" in the top-left
    is completely unrelated to the same detector in the bottom-right.

With convolution, ONE filter of shape (C, kH, kW) slides over the entire image.
The same weights detect the same feature everywhere. Why this is correct:
- Images are statistically translation-equivariant: a cat in the top-left and a
  cat in the bottom-right have the same local pixel patterns.
- Any pattern useful in one region is likely useful everywhere.
- This encodes a strong inductive bias that massively reduces sample complexity.

Result: a 3×3 filter has only 9 weights but detects a pattern at all 26×26 positions
in a 28×28 image. Weight sharing gives translation equivariance "for free."

## 3. Why im2col Makes Convolution = GEMM

Direct convolution with nested Python loops is extremely slow:
    O(N * F * C * kH * kW * H_out * W_out) individual multiply-add operations

im2col reorganizes the computation into a single matrix multiply:

1. Extract all input patches into a 2D matrix:
   cols shape: (N * H_out * W_out,  C * kH * kW)
   Each row is one flattened patch of the input.

2. Reshape filter weights:
   W_flat shape: (F, C * kH * kW)

3. Matrix multiply:
   out_flat = cols @ W_flat.T    shape: (N * H_out * W_out, F)

This is now a single DGEMM (double-precision general matrix multiply) call.
BLAS libraries (MKL, OpenBLAS, cuBLAS) achieve near-peak hardware performance on DGEMM
via cache blocking, SIMD vectorization, and hardware FMA units.

Trade-off: im2col uses more memory (the cols matrix can be kH*kW times larger
than the original input) but is much faster in practice because BLAS can be
5-100x faster than naive loops.

This is why frameworks like PyTorch, TensorFlow, and Caffe all use im2col or
equivalent algorithms internally, and why GPUs are so effective at neural networks.

## 4. What Max Pooling Achieves

Max pooling takes the maximum value in a (kH × kW) window:
    out[n, c, i, j] = max_{p in 0..kH-1, q in 0..kW-1} x[n, c, i*s+p, j*s+q]

This achieves two things:

(a) Spatial invariance (approximately):
    If the strongest activation for "horizontal edge" shifts 1 pixel due to slight
    image misalignment, the max in that 2×2 window is unchanged. This makes the
    network robust to small translations beyond what convolution alone provides.
    (Equivariance + invariance ≈ robust features at each scale)

(b) Downsampling / increasing receptive field:
    Each 2×2 MaxPool reduces spatial dimensions by 2×. After two pools in our
    network, 28×28 becomes 7×7. Neurons in deeper layers now "see" a larger
    region of the original image — larger effective receptive field.

Backward pass: gradient flows only to the max element in each window (the one
that "won" the competition during forward). All other elements get zero gradient.
This is implemented by storing argmax indices and routing gradients accordingly.

## 5. Why CNNs Have Far Fewer Parameters Than MLPs for Images

Comparison for a 28×28 grayscale input:

MLP (2 hidden layers):
    Layer 1: 784 × 512 = 401,408 parameters
    Layer 2: 512 × 256 = 131,072 parameters
    Total before output: ~532,480

CNN (our architecture):
    Conv1: 8 × (1×3×3) + 8  =    80 parameters
    Conv2: 16 × (8×3×3) + 16 = 1,168 parameters
    FC1:  784 × 64 + 64      = 50,240 parameters
    FC2:  64 × 10 + 10       =    650 parameters
    Total: ~52,138

The CNN uses ~10x fewer parameters because:
1. Weight sharing: the same filter detects an edge at all 26×26 positions
   (576 positions for one filter, but only 9 weights)
2. Local connectivity: each neuron only connects to a kH×kW patch, not all inputs

With fewer parameters:
- Less overfitting (lower model complexity)
- Less memory
- Faster training
- Better generalization (inductive bias matches image structure)

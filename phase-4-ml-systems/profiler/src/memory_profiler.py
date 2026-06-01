"""
memory_profiler.py — Memory tracking utilities for ML training loops.

Provides:
  - track_peak_memory : run a callable and report peak RSS via tracemalloc
  - estimate_tensor_memory_mb : calculate expected array memory from shape + dtype
  - model_memory_breakdown : print per-component memory for an MLP
"""

import tracemalloc
import numpy as np


# ---------------------------------------------------------------------------
# Peak memory tracker
# ---------------------------------------------------------------------------

def track_peak_memory(fn, *args, **kwargs):
    """
    Run fn(*args, **kwargs) and return (result, peak_mb).

    Uses Python's tracemalloc to capture the peak allocated memory
    during the function call.

    Parameters
    ----------
    fn   : callable
    *args, **kwargs : forwarded to fn

    Returns
    -------
    result   : whatever fn returns
    peak_mb  : float — peak allocated memory in megabytes

    Notes
    -----
    tracemalloc tracks Python-level allocations.  NumPy allocations go through
    Python's memory manager so they are captured, but C-extension memory
    (e.g. BLAS scratch buffers) may not be fully reflected.
    """
    tracemalloc.start()
    try:
        result = fn(*args, **kwargs)
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return result, peak / 1024 / 1024


# ---------------------------------------------------------------------------
# Tensor memory estimator
# ---------------------------------------------------------------------------

def estimate_tensor_memory_mb(shape, dtype=np.float32) -> float:
    """
    Compute the expected memory (MB) of a NumPy array with the given shape and dtype.

    Parameters
    ----------
    shape : tuple[int, ...]
    dtype : numpy dtype (default float32 = 4 bytes per element)

    Returns
    -------
    float — megabytes

    Examples
    --------
    >>> estimate_tensor_memory_mb((1024, 1024))   # 4 MB
    4.0
    >>> estimate_tensor_memory_mb((512, 512), np.float64)  # 2 MB
    2.0
    """
    n_elements = int(np.prod(shape))
    itemsize   = np.dtype(dtype).itemsize
    return n_elements * itemsize / (1024 * 1024)


# ---------------------------------------------------------------------------
# Model memory breakdown
# ---------------------------------------------------------------------------

def model_memory_breakdown(layer_sizes: list,
                            batch_size: int = 32,
                            dtype=np.float32):
    """
    Report expected memory usage for each component of a fully-connected MLP.

    Components reported
    -------------------
    Weights     : weight matrices for each layer
    Biases      : bias vectors for each layer
    Activations : intermediate activations stored for backprop
                  (one per layer × batch_size)
    Gradients   : same shape as weights + biases (dL/dW, dL/db)
    Adam m      : first moment estimate (same shape as parameters)
    Adam v      : second moment estimate (same shape as parameters)

    Parameters
    ----------
    layer_sizes : list[int]  e.g. [784, 256, 128, 10]
    batch_size  : int  (used to compute activation memory)
    dtype       : numpy dtype for all arrays (default float32)

    Prints a table and returns a dict with totals.
    """
    itemsize = np.dtype(dtype).itemsize

    rows = []

    # --- Weights and biases ------------------------------------------------
    weights_mb = 0.0
    biases_mb  = 0.0

    for i, (n_in, n_out) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        w_shape = (n_out, n_in)
        b_shape = (n_out,)
        w_mb = estimate_tensor_memory_mb(w_shape, dtype)
        b_mb = estimate_tensor_memory_mb(b_shape, dtype)
        rows.append(("weights", f"layer{i+1}", w_shape, w_mb))
        rows.append(("biases",  f"layer{i+1}", b_shape, b_mb))
        weights_mb += w_mb
        biases_mb  += b_mb

    # --- Activations (stored for backward pass) ----------------------------
    activations_mb = 0.0
    for i, n in enumerate(layer_sizes):
        shape  = (batch_size, n)
        act_mb = estimate_tensor_memory_mb(shape, dtype)
        rows.append(("activations", f"layer{i}", shape, act_mb))
        activations_mb += act_mb

    # --- Gradients (same shape as weights + biases) ------------------------
    grad_weights_mb = weights_mb
    grad_biases_mb  = biases_mb
    for i, (n_in, n_out) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        w_shape = (n_out, n_in)
        b_shape = (n_out,)
        rows.append(("grad_weights", f"layer{i+1}", w_shape,
                     estimate_tensor_memory_mb(w_shape, dtype)))
        rows.append(("grad_biases",  f"layer{i+1}", b_shape,
                     estimate_tensor_memory_mb(b_shape, dtype)))

    # --- Adam optimizer state (m and v, same shape as params) --------------
    adam_mb = (weights_mb + biases_mb) * 2  # m and v
    for i, (n_in, n_out) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        w_shape = (n_out, n_in)
        b_shape = (n_out,)
        for moment in ("adam_m", "adam_v"):
            rows.append((moment, f"layer{i+1}_W", w_shape,
                         estimate_tensor_memory_mb(w_shape, dtype)))
            rows.append((moment, f"layer{i+1}_b", b_shape,
                         estimate_tensor_memory_mb(b_shape, dtype)))

    # --- Print table -------------------------------------------------------
    print(f"\n{'=' * 70}")
    print(f"  Memory Breakdown for MLP{layer_sizes}  "
          f"(batch={batch_size}, dtype={np.dtype(dtype).name})")
    print(f"{'=' * 70}")
    print(f"  {'Component':<15} {'Layer':<14} {'Shape':<20} {'MB':>8}")
    print(f"  {'-' * 63}")

    last_component = None
    component_totals = {}
    for component, layer, shape, mb in rows:
        if component != last_component and last_component is not None:
            print(f"  {'':15} {'':14} {'':20} {'---':>8}")
        print(f"  {component:<15} {layer:<14} {str(shape):<20} {mb:>8.4f}")
        component_totals[component] = component_totals.get(component, 0.0) + mb
        last_component = component

    total_mb = sum(component_totals.values())
    print(f"  {'=' * 63}")

    for component, total in sorted(component_totals.items(),
                                   key=lambda x: -x[1]):
        print(f"  {'TOTAL ' + component.upper():<30} {total:>8.4f} MB")

    print(f"  {'GRAND TOTAL':<30} {total_mb:>8.4f} MB")
    print(f"{'=' * 70}")

    return {
        "weights_mb"    : weights_mb,
        "biases_mb"     : biases_mb,
        "activations_mb": activations_mb,
        "gradients_mb"  : grad_weights_mb + grad_biases_mb,
        "adam_mb"       : adam_mb,
        "total_mb"      : total_mb,
    }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Simple smoke test
    print("estimate_tensor_memory_mb tests:")
    print(f"  (1024, 1024) float32 → {estimate_tensor_memory_mb((1024, 1024)):.1f} MB  (expected 4.0)")
    print(f"  (512, 512) float64  → {estimate_tensor_memory_mb((512, 512), np.float64):.1f} MB  (expected 2.0)")

    print("\ntrack_peak_memory test:")
    def alloc_and_free():
        arr = np.ones((1024, 1024), dtype=np.float32)  # 4 MB
        return arr.sum()

    result, peak_mb = track_peak_memory(alloc_and_free)
    print(f"  Result: {result:.0f},  peak memory: {peak_mb:.2f} MB")

    print()
    breakdown = model_memory_breakdown([784, 256, 128, 10], batch_size=32)

"""
pruning.py — Structured and unstructured pruning for neural networks.

Implements:
  - magnitude_prune: remove smallest-magnitude weights (unstructured)
  - structured_prune_channels: remove entire output channels by L2 norm
  - iterative_prune: gradually increase sparsity over N rounds
  - measure_sparsity: count zeros
  - apply_masks: apply binary masks to zero out weights
"""

import numpy as np
from typing import List, Dict, Tuple, Optional


# ---------------------------------------------------------------------------
# Unstructured pruning
# ---------------------------------------------------------------------------

def magnitude_prune(weights: np.ndarray, sparsity: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Set the (sparsity * 100)% smallest-magnitude weights to zero.

    This is the most common pruning heuristic: small weights contribute
    little to the output, so removing them has minimal accuracy impact.

    Parameters
    ----------
    weights  : any shape float array
    sparsity : fraction of weights to zero, in [0, 1]

    Returns
    -------
    pruned_weights : same shape, with zeros at small-magnitude positions
    mask           : bool array, True where weight is kept
    """
    flat     = weights.flatten()
    n_prune  = int(len(flat) * sparsity)
    if n_prune == 0:
        return weights.copy(), np.ones_like(weights, dtype=bool)

    # Find the threshold: (sparsity)-th percentile of |w|
    threshold = np.sort(np.abs(flat))[n_prune - 1]
    mask      = np.abs(weights) > threshold     # True = keep
    pruned    = weights * mask.astype(weights.dtype)
    return pruned, mask


def magnitude_prune_model(model_weights: List[np.ndarray],
                           sparsity: float) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Apply magnitude pruning to every weight matrix in a model.

    model_weights : list of weight arrays (not biases — we don't prune biases)
    Returns: (pruned_weights, masks)
    """
    pruned_list = []
    mask_list   = []
    for W in model_weights:
        pW, mask = magnitude_prune(W, sparsity)
        pruned_list.append(pW)
        mask_list.append(mask)
    return pruned_list, mask_list


# ---------------------------------------------------------------------------
# Structured pruning
# ---------------------------------------------------------------------------

def structured_prune_channels(weight_matrix: np.ndarray,
                               sparsity: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove entire output channels (rows) with the smallest L2 norms.

    Unlike unstructured pruning, this produces a smaller dense matrix
    that can actually speed up inference without sparse hardware support.

    Parameters
    ----------
    weight_matrix : shape (out_features, in_features)
    sparsity      : fraction of output channels to remove

    Returns
    -------
    pruned_matrix : shape (out_features, in_features) with zero rows
    channel_mask  : bool array of shape (out_features,), True = kept
    """
    n_channels = weight_matrix.shape[0]
    n_prune    = int(n_channels * sparsity)
    if n_prune == 0:
        return weight_matrix.copy(), np.ones(n_channels, dtype=bool)

    # L2 norm per row
    norms     = np.linalg.norm(weight_matrix, axis=1)
    threshold = np.sort(norms)[n_prune - 1]
    mask      = norms > threshold                            # True = keep

    pruned    = weight_matrix.copy()
    pruned[~mask] = 0.0
    return pruned, mask


def extract_structured_model(weight_matrix: np.ndarray,
                              channel_mask: np.ndarray) -> np.ndarray:
    """
    Actually remove the pruned channels to get a smaller dense matrix.
    This is what you'd do in deployment (vs just zeroing them out).
    """
    return weight_matrix[channel_mask]


# ---------------------------------------------------------------------------
# Iterative pruning
# ---------------------------------------------------------------------------

def iterative_prune(weights: np.ndarray,
                    target_sparsity: float,
                    n_rounds: int) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    """
    Gradually increase sparsity from 0 to target_sparsity over n_rounds.

    This is better than one-shot pruning because:
    1. Smaller per-step accuracy hits are more recoverable with fine-tuning
    2. The remaining weights are pruned relative to each other (not to original)

    The sparsity at round t follows a cubic schedule:
        s_t = target * (1 - (1 - t/T)^3)

    This is the schedule used in "To Prune or Not to Prune" (Zhu & Gupta, 2017).

    Parameters
    ----------
    weights         : original weight array
    target_sparsity : final desired sparsity (e.g. 0.90 for 90%)
    n_rounds        : number of pruning rounds

    Returns
    -------
    list of (pruned_weights, mask, sparsity) for each round
    """
    history = []
    current = weights.copy()

    for t in range(1, n_rounds + 1):
        # Cubic sparsity schedule
        s_t = target_sparsity * (1 - (1 - t / n_rounds) ** 3)
        pruned, mask = magnitude_prune(current, s_t)
        history.append((pruned.copy(), mask.copy(), s_t))
        current = pruned   # prune the already-pruned weights

    return history


# ---------------------------------------------------------------------------
# Sparsity measurement
# ---------------------------------------------------------------------------

def measure_sparsity(weights: np.ndarray) -> Dict[str, float]:
    """
    Compute sparsity statistics for a weight array.

    Returns dict with: n_zero, n_total, sparsity, density
    """
    total  = weights.size
    n_zero = int(np.sum(weights == 0))
    return {
        "n_zero"   : n_zero,
        "n_total"  : total,
        "sparsity" : n_zero / total,
        "density"  : 1 - n_zero / total,
    }


def measure_model_sparsity(model_weights: List[np.ndarray]) -> Dict[str, float]:
    """Aggregate sparsity across all model weight matrices."""
    total_params = sum(W.size for W in model_weights)
    total_zero   = sum(int(np.sum(W == 0)) for W in model_weights)
    per_layer    = [measure_sparsity(W) for W in model_weights]
    return {
        "total_params"      : total_params,
        "total_zero"        : total_zero,
        "global_sparsity"   : total_zero / total_params,
        "per_layer_sparsity": [s["sparsity"] for s in per_layer],
    }


# ---------------------------------------------------------------------------
# Apply masks
# ---------------------------------------------------------------------------

def apply_masks(model_weights: List[np.ndarray],
                masks: List[np.ndarray]) -> List[np.ndarray]:
    """
    Apply binary masks to weight matrices.
    mask[i][j] = 0 → weight is permanently zeroed.
    In fine-tuning, you also apply the mask after each gradient update
    to prevent zeroed weights from being "resurrected".
    """
    return [W * M.astype(W.dtype) for W, M in zip(model_weights, masks)]


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(0)

    W = rng.standard_normal((64, 128)).astype(np.float32)

    print("=" * 60)
    print("UNSTRUCTURED MAGNITUDE PRUNING")
    for sparsity in [0.0, 0.5, 0.75, 0.90]:
        pW, mask = magnitude_prune(W, sparsity)
        stats = measure_sparsity(pW)
        print(f"  target={sparsity:.0%}  actual={stats['sparsity']:.1%}  "
              f"zeros={stats['n_zero']}/{stats['n_total']}")

    print()
    print("STRUCTURED CHANNEL PRUNING")
    for sparsity in [0.0, 0.25, 0.5, 0.75]:
        pW, mask = structured_prune_channels(W, sparsity)
        n_kept  = mask.sum()
        n_total = len(mask)
        print(f"  target={sparsity:.0%}  channels_kept={n_kept}/{n_total}  "
              f"actual_sparsity={1-n_kept/n_total:.1%}")
        dense = extract_structured_model(W, mask)
        print(f"    → Dense matrix shape: {dense.shape} (was {W.shape})")

    print()
    print("ITERATIVE PRUNING (cubic schedule, 5 rounds → 90%)")
    history = iterative_prune(W, target_sparsity=0.90, n_rounds=5)
    for i, (pW, mask, s) in enumerate(history):
        stats = measure_sparsity(pW)
        print(f"  Round {i+1}: target={s:.1%}  actual={stats['sparsity']:.1%}")

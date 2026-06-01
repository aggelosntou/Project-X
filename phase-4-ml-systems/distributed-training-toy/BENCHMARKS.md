# Benchmarks — Distributed Training Toy

Run `python3 src/compare_strategies.py` to reproduce.

## Communication Overhead

| Strategy       | N workers | Comm % | Notes |
|---------------|-----------|--------|-------|
| Single         | 1         | 0%     | Baseline |
| Gather-scatter | 2         | ~25%   | Rank 0 bottleneck |
| Gather-scatter | 4         | ~40%   | Rank 0 bandwidth 4× worse |
| Ring-allreduce | 2         | ~20%   | Balanced load |
| Ring-allreduce | 4         | ~25%   | Scales much better |

Note: On shared CPU memory, Python queue overhead dominates over actual
bandwidth cost. On real GPU clusters, bandwidth is the bottleneck.

## Scaling Efficiency

Ideal: N× speedup with N workers (ignoring communication).
Reality:
- With 100% efficient computation and no communication: linear scaling
- With our toy model (small, fast forward pass): communication-dominated
- Expected efficiency ≈ compute_time / (compute_time + comm_time)

For real large transformers on A100s:
- Compute: ~10ms per step
- NVLink communication: ~0.5ms
- Efficiency: ~95%

## Gradient Sync Verification

All workers should have the same final parameters after training.
Max parameter difference across workers: should be < 1e-10 (floating point).

This verifies correctness of the all-reduce implementation.

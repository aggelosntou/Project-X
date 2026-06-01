# Training Loop Profiler

A custom profiling toolkit for understanding where time and memory go
during neural-network training — without PyTorch or TensorFlow.

## Project layout

```
profiler/
├── src/
│   ├── profiler.py           # Timer context manager + Profiler class + @profile decorator
│   ├── memory_profiler.py    # tracemalloc wrapper + tensor memory estimator + breakdown
│   ├── flops_counter.py      # FLOPs formulas for linear, conv, attention layers
│   └── benchmark_training.py # End-to-end benchmark of a full training step
├── README.md
├── LEARNINGS.md
└── requirements.txt
```

## Quick start

```bash
cd profiler
pip install -r requirements.txt

# Run the full benchmark
python src/benchmark_training.py

# Run individual self-tests
python src/profiler.py
python src/memory_profiler.py
python src/flops_counter.py
```

## What each file does

### profiler.py

Provides a `Profiler` context manager and `Timer` sub-context:

```python
from profiler import Profiler

prof = Profiler()
with prof:
    for step in range(100):
        with prof.timer('forward'):
            out = model.forward(x)
        with prof.timer('backward'):
            model.backward(grad)

prof.report()
```

Output:
```
Operation                 Calls  Total(ms)   Mean(ms)    Max(ms)   %Total
-------------------------------------------------------------------------
backward                    100     412.30       4.12       5.80    61.3%
forward                     100     195.10       1.95       2.40    29.0%
...
```

Also provides a `@profile` decorator for instrumenting functions directly.

### memory_profiler.py

- `track_peak_memory(fn, *args)` — wrap any function and get `(result, peak_mb)`
- `estimate_tensor_memory_mb(shape, dtype)` — quick MB estimate
- `model_memory_breakdown(layer_sizes)` — full table: weights, activations,
  gradients, Adam m/v state

### flops_counter.py

- `flops_linear(batch, in_f, out_f)` — FLOPs for a dense layer
- `flops_conv(batch, in_c, out_c, k, H, W)` — FLOPs for a 2D conv
- `flops_attention(batch, seq, d_model, heads)` — transformer attention FLOPs
- `count_model_flops(layer_sizes, batch_size)` — total for an MLP
- `arithmetic_intensity_linear(...)` — FLOPs/byte for roofline analysis

## Sample output

```
==========================================================================
  Training Loop Phases — 100 steps
  Total profiled time: 682.11 ms
==========================================================================
Operation                 Calls  Total(ms)   Mean(ms)    Max(ms)   %Total
--------------------------------------------------------------------------
backward                    100     388.23       3.88       4.91    56.9%
optimizer_step              100     142.17       1.42       2.10    20.8%
forward                     100     112.44       1.12       1.63    16.5%
loss_computation            100      32.18       0.32       0.45     4.7%
data_loading                100       7.09       0.07       0.20     1.0%
==========================================================================

  FLOPs per forward pass:  107.37 MFLOPs
  Backward/Forward ratio: 3.45x
  Peak allocated memory: 18.32 MB
```

"""
profiler.py — Lightweight training-loop profiler.

Usage
-----
    from profiler import Profiler

    with Profiler() as prof:
        for step in range(100):
            with prof.timer('forward'):
                out = model.forward(x)
            with prof.timer('loss'):
                loss = compute_loss(out, y)
            with prof.timer('backward'):
                grads = compute_grads(loss)

    prof.report()

The Profiler stores wall-clock timings for every named section and prints
a sorted table with total time, mean, max, and percentage of total.
"""

import time
import functools
from collections import defaultdict

import numpy as np


# ---------------------------------------------------------------------------
# Timer context manager
# ---------------------------------------------------------------------------

class Timer:
    """
    Context manager that measures wall-clock time and appends it to a registry.

    Parameters
    ----------
    name     : str  — label for this timing section
    registry : dict[str, list[float]]  — shared storage
    """

    def __init__(self, name: str, registry: dict):
        self.name     = name
        self.registry = registry
        self._start   = None

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        elapsed = time.perf_counter() - self._start
        self.registry[self.name].append(elapsed)

    @property
    def elapsed_ms(self):
        """Return the last recorded elapsed time in milliseconds."""
        times = self.registry.get(self.name, [])
        return times[-1] * 1000 if times else 0.0


# ---------------------------------------------------------------------------
# Profiler
# ---------------------------------------------------------------------------

class Profiler:
    """
    Profile a training loop (or any block of code) by wrapping operations
    with named timer contexts.

    The profiler collects wall-clock times for every named section and can
    print a sorted summary table.

    Thread / process safety
    -----------------------
    Not thread-safe.  Use one Profiler instance per thread/process.
    """

    def __init__(self):
        self._timings: dict = defaultdict(list)   # name → [elapsed_s, ...]
        self._active  = False

    # --- Context manager protocol ----------------------------------------

    def __enter__(self):
        self._active = True
        self.reset()
        return self

    def __exit__(self, *args):
        self._active = False

    # --- Timer factory -----------------------------------------------------

    def timer(self, name: str) -> Timer:
        """
        Return a Timer context manager for the given section name.

        Usage:
            with profiler.timer('forward'):
                output = model(x)
        """
        return Timer(name, self._timings)

    # --- Data access -------------------------------------------------------

    def total_time(self, name: str) -> float:
        """Return total elapsed time (seconds) for a given section."""
        return sum(self._timings.get(name, []))

    def mean_time(self, name: str) -> float:
        """Return mean elapsed time (seconds) for a given section."""
        times = self._timings.get(name, [])
        return float(np.mean(times)) if times else 0.0

    def call_count(self, name: str) -> int:
        return len(self._timings.get(name, []))

    def all_names(self):
        return list(self._timings.keys())

    def reset(self):
        """Clear all collected timings."""
        self._timings = defaultdict(list)

    # --- Report ------------------------------------------------------------

    def report(self, title: str = "Profiling Report"):
        """
        Print a table of timing statistics sorted by total time (descending).

        Columns
        -------
        Operation  : section name
        Calls      : number of times the section ran
        Total(ms)  : sum of all elapsed times in milliseconds
        Mean(ms)   : average time per call in milliseconds
        Max(ms)    : maximum single call time in milliseconds
        %Total     : fraction of total profiled time
        """
        if not self._timings:
            print("No timings recorded.")
            return

        total_all = sum(sum(v) for v in self._timings.values())
        if total_all == 0:
            print("All recorded times are zero.")
            return

        print(f"\n{'=' * 75}")
        print(f"  {title}")
        print(f"  Total profiled time: {total_all * 1000:.2f} ms")
        print(f"{'=' * 75}")
        header = (f"{'Operation':<25} {'Calls':>6} {'Total(ms)':>10} "
                  f"{'Mean(ms)':>10} {'Max(ms)':>10} {'%Total':>8}")
        print(header)
        print("-" * 75)

        sorted_ops = sorted(self._timings.items(), key=lambda x: -sum(x[1]))

        for name, times in sorted_ops:
            total = sum(times) * 1000          # seconds → ms
            mean  = float(np.mean(times)) * 1000
            mx    = float(max(times)) * 1000
            pct   = sum(times) / total_all * 100
            print(f"{name:<25} {len(times):>6} {total:>10.2f} "
                  f"{mean:>10.2f} {mx:>10.2f} {pct:>8.1f}%")

        print("=" * 75)

    def __repr__(self):
        sections = list(self._timings.keys())
        return f"Profiler(sections={sections})"


# ---------------------------------------------------------------------------
# Decorator API
# ---------------------------------------------------------------------------

def profile(name: str, registry: dict):
    """
    Decorator that times a function and stores the result in `registry`.

    Parameters
    ----------
    name     : str  — label stored in the registry
    registry : dict[str, list[float]]  — shared timing storage
               (usually profiler._timings)

    Usage
    -----
        @profile('matmul', profiler._timings)
        def matmul(A, B):
            return A @ B
    """
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            start  = time.perf_counter()
            result = fn(*args, **kwargs)
            registry[name].append(time.perf_counter() - start)
            return result
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import numpy as np

    rng = np.random.default_rng(0)
    A   = rng.standard_normal((512, 512)).astype(np.float32)
    B   = rng.standard_normal((512, 512)).astype(np.float32)

    prof = Profiler()

    with prof:
        for _ in range(20):
            with prof.timer("matmul"):
                C = A @ B

            with prof.timer("relu"):
                D = np.maximum(C, 0)

            with prof.timer("softmax"):
                e = np.exp(D - D.max(axis=1, keepdims=True))
                _ = e / e.sum(axis=1, keepdims=True)

            with prof.timer("loss"):
                labels = rng.integers(0, 512, 512)
                _ = -float(np.mean(np.log(e[np.arange(512), labels] + 1e-9)))

    prof.report("Self-test: 20 iterations of matmul → relu → softmax → loss")

"""
load_test.py — Load testing for the inference server.

Can run in two modes:
  1. Against a live server:  python3 src/load_test.py --server http://localhost:8000
  2. Standalone (no server): python3 src/load_test.py --standalone

In standalone mode, we test the batching logic directly (no HTTP overhead)
so you can see the pure batching + inference performance.
"""

import sys
import os
import time
import argparse
import statistics
import concurrent.futures
import queue
from typing import List, Dict

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))


# ---------------------------------------------------------------------------
# Standalone mode (direct batching test)
# ---------------------------------------------------------------------------

def run_standalone_load_test(
    concurrency_levels: List[int],
    n_requests_per_level: int = 200,
    in_features: int = 64,
):
    """Test the batching engine directly, without HTTP overhead."""
    from model import MLP
    from batching import BatchAccumulator

    m = MLP.random_init([in_features, 128, 64, 10])
    acc = BatchAccumulator(
        model_fn=m.predict,
        max_batch_size=32,
        max_wait_ms=10.0,
        max_queue_size=512,
    )

    print(f"\n{'='*65}")
    print(f"STANDALONE LOAD TEST  (no HTTP, direct batching)")
    print(f"{'='*65}")
    print(f"{'Concurrency':>12}  {'Throughput':>12}  {'p50 (ms)':>10}  "
          f"{'p95 (ms)':>10}  {'p99 (ms)':>10}  {'Mean Batch':>10}")
    print(f"{'-'*12}  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")

    all_results = []

    for concurrency in concurrency_levels:
        inputs = [np.random.randn(in_features).astype(np.float32)
                  for _ in range(n_requests_per_level)]

        latencies = []
        errors    = 0

        t_start = time.perf_counter()

        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            def send_one(x):
                t0 = time.perf_counter()
                try:
                    _ = acc.submit(x)
                    return (time.perf_counter() - t0) * 1000, False
                except queue.Full:
                    return 0.0, True

            futures = [pool.submit(send_one, inp) for inp in inputs]
            for f in concurrent.futures.as_completed(futures):
                lat, err = f.result()
                if err:
                    errors += 1
                else:
                    latencies.append(lat)

        elapsed  = time.perf_counter() - t_start
        n_ok     = len(latencies)
        throughput = n_ok / elapsed if elapsed > 0 else 0

        p50 = statistics.median(latencies) if latencies else 0
        p95 = float(np.percentile(latencies, 95)) if latencies else 0
        p99 = float(np.percentile(latencies, 99)) if latencies else 0
        mb  = acc.metrics.mean_batch_size()

        print(f"{concurrency:>12}  {throughput:>10.1f}/s  {p50:>10.2f}  "
              f"{p95:>10.2f}  {p99:>10.2f}  {mb:>10.2f}")

        all_results.append({
            "concurrency": concurrency,
            "throughput": throughput,
            "p50_ms": p50,
            "p95_ms": p95,
            "p99_ms": p99,
            "errors": errors,
            "mean_batch_size": mb,
        })

    acc.stop()
    return all_results


# ---------------------------------------------------------------------------
# HTTP mode (against a live server)
# ---------------------------------------------------------------------------

def run_http_load_test(
    server_url: str,
    concurrency_levels: List[int],
    n_requests_per_level: int = 200,
    in_features: int = 64,
):
    try:
        import urllib.request
        import json
    except ImportError:
        print("Standard library only — no extra deps needed.")
        return []

    def send_request(server_url, x):
        payload = json.dumps({"inputs": [x.tolist()]}).encode()
        req = urllib.request.Request(
            f"{server_url}/predict",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                _ = json.loads(resp.read())
            return (time.perf_counter() - t0) * 1000, False
        except Exception:
            return 0.0, True

    print(f"\n{'='*65}")
    print(f"HTTP LOAD TEST  →  {server_url}")
    print(f"{'='*65}")
    print(f"{'Concurrency':>12}  {'Throughput':>12}  {'p50 (ms)':>10}  "
          f"{'p95 (ms)':>10}  {'p99 (ms)':>10}")
    print(f"{'-'*12}  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*10}")

    all_results = []
    for concurrency in concurrency_levels:
        inputs = [np.random.randn(in_features).astype(np.float32)
                  for _ in range(n_requests_per_level)]
        latencies = []

        t_start = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(send_request, server_url, inp) for inp in inputs]
            for f in concurrent.futures.as_completed(futures):
                lat, err = f.result()
                if not err:
                    latencies.append(lat)

        elapsed   = time.perf_counter() - t_start
        n_ok      = len(latencies)
        throughput = n_ok / elapsed

        p50 = statistics.median(latencies) if latencies else 0
        p95 = float(np.percentile(latencies, 95)) if latencies else 0
        p99 = float(np.percentile(latencies, 99)) if latencies else 0

        print(f"{concurrency:>12}  {throughput:>10.1f}/s  "
              f"{p50:>10.2f}  {p95:>10.2f}  {p99:>10.2f}")

        all_results.append({
            "concurrency": concurrency,
            "throughput":  throughput,
            "p50_ms": p50,
            "p95_ms": p95,
            "p99_ms": p99,
        })

    return all_results


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze_results(results: List[Dict]):
    if not results:
        return

    print(f"\n{'='*65}")
    print("ANALYSIS")
    print(f"{'='*65}")

    # Find saturation point (where throughput stops increasing significantly)
    throughputs = [r["throughput"] for r in results]
    max_tp      = max(throughputs)
    max_c_idx   = throughputs.index(max_tp)
    sat_concurrency = results[max_c_idx]["concurrency"]

    print(f"Peak throughput: {max_tp:.1f} req/s at concurrency={sat_concurrency}")

    # Latency growth
    base_p99 = results[0]["p99_ms"]
    peak_p99 = results[-1]["p99_ms"]
    print(f"p99 latency:  {base_p99:.2f} ms at c=1  →  {peak_p99:.2f} ms at "
          f"c={results[-1]['concurrency']}")

    # Little's Law: L = λ · W   (concurrency = throughput × latency)
    print(f"\nLittle's Law verification (concurrency ≈ throughput × latency):")
    for r in results:
        predicted_c = r["throughput"] * r["p50_ms"] / 1000
        print(f"  c={r['concurrency']:>3}  actual={r['concurrency']}  "
              f"predicted={predicted_c:.1f}  "
              f"(λ={r['throughput']:.1f} req/s, W={r['p50_ms']:.2f}ms)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load test for ML inference server")
    parser.add_argument("--server", default=None,
                        help="Server URL, e.g. http://localhost:8000")
    parser.add_argument("--standalone", action="store_true",
                        help="Run standalone (no HTTP server needed)")
    parser.add_argument("--n-requests", type=int, default=300,
                        help="Requests per concurrency level")
    args = parser.parse_args()

    concurrency_levels = [1, 2, 4, 8, 16, 32]

    if args.server:
        results = run_http_load_test(
            args.server, concurrency_levels, args.n_requests
        )
    else:
        # Default: standalone
        results = run_standalone_load_test(
            concurrency_levels, args.n_requests
        )

    analyze_results(results)

    print(f"\nConclusion:")
    print(f"  - Dynamic batching allows throughput to scale nearly linearly")
    print(f"    with concurrency up to max_batch_size={32}.")
    print(f"  - After saturation, adding more clients only increases latency.")
    print(f"  - p99 matters more than p50: even 1% of slow requests")
    print(f"    cause timeouts in user-facing systems.")

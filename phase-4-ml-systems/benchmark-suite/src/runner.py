"""
runner.py
=========
Run all benchmarks and save results to JSON.

Usage:
    python3 src/runner.py [output_dir]

Output:
    <output_dir>/benchmark_results.json
"""

import json
import datetime
import os
import sys

# Ensure src/ is on the path regardless of how the script is invoked
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from system_info import get_system_info, print_system_info
from benchmarks.matmul_benchmark   import benchmark_matmul
from benchmarks.attention_benchmark import benchmark_attention
from benchmarks.training_benchmark  import benchmark_training
from benchmarks.inference_benchmark import benchmark_inference


def run_all(output_dir: str = '.') -> dict:
    """
    Run all benchmarks in sequence, collect results, and write to JSON.

    Args:
        output_dir: directory where benchmark_results.json is written

    Returns:
        results dict
    """
    print_system_info()

    results = {
        'timestamp': datetime.datetime.now().isoformat(),
        'system':    get_system_info(),
    }

    print("\nRunning benchmarks...")
    print("=" * 50)

    results['matmul']    = benchmark_matmul()    or {}
    results['attention'] = benchmark_attention() or {}
    results['training']  = benchmark_training()  or {}
    results['inference'] = benchmark_inference() or {}

    # Write results to JSON (default= str handles numpy scalars gracefully)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, 'benchmark_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'=' * 50}")
    print(f"All benchmarks complete.")
    print(f"Results saved to {out_path}")
    return results


if __name__ == '__main__':
    output_dir = sys.argv[1] if len(sys.argv) > 1 else '.'
    run_all(output_dir)

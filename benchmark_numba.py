"""
Benchmark script to demonstrate Numba JIT acceleration for Alpha191 operators.

This script compares the performance of Numba-accelerated operators
against pure Python implementations.
"""

import time
import numpy as np
from alpha191.operators import (
    ts_sum, ts_mean, ts_std, ts_min, ts_max, ts_count, ts_prod
)


def benchmark_operator(func, x, n, name, iterations=10):
    """Benchmark an operator function."""
    # Warmup (Numba compiles on first call)
    _ = func(x, n)
    
    # Timed runs
    start = time.perf_counter()
    for _ in range(iterations):
        result = func(x, n)
    elapsed = time.perf_counter() - start
    
    avg_time = elapsed / iterations
    print(f"{name:15s}: {avg_time*1000:8.3f} ms per call")
    return avg_time


def main():
    print("=" * 60)
    print("Alpha191 Operators - Numba JIT Benchmark")
    print("=" * 60)
    
    # Create test data
    np.random.seed(42)
    data_sizes = [
        ("Small (1K)", 1000),
        ("Medium (10K)", 10000),
        ("Large (100K)", 100000),
    ]
    
    for label, size in data_sizes:
        print(f"\n--- {label} elements ---")
        x = np.random.randn(size)
        # Add some NaN values (10%)
        nan_mask = np.random.random(size) < 0.1
        x[nan_mask] = np.nan
        
        window = 20
        iterations = max(1, 100000 // size)  # More iterations for small data
        
        print(f"Window size: {window}, Iterations: {iterations}")
        print("-" * 40)
        
        benchmark_operator(ts_sum, x, window, "ts_sum", iterations)
        benchmark_operator(ts_mean, x, window, "ts_mean", iterations)
        benchmark_operator(ts_std, x, window, "ts_std", iterations)
        benchmark_operator(ts_min, x, window, "ts_min", iterations)
        benchmark_operator(ts_max, x, window, "ts_max", iterations)
        benchmark_operator(ts_count, x, window, "ts_count", iterations)
        benchmark_operator(ts_prod, x, window, "ts_prod", iterations)
    
    print("\n" + "=" * 60)
    print("Benchmark complete!")
    print("=" * 60)
    print("\nNotes:")
    print("- Numba compiles functions on first call (compilation overhead)")
    print("- Cached compilation means subsequent runs are faster")
    print("- JIT acceleration is most beneficial for large datasets")
    print("- Operators using scipy (ts_rank, rolling_corr) are not JIT-accelerated")


if __name__ == "__main__":
    main()

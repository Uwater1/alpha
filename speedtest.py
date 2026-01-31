#!/usr/bin/env python3
"""
Speed test script for Alpha191 factors.

Usage:
    python speedtest.py <alpha_number> [benchmark]

Example:
    python speedtest.py 17 hs300
    python speedtest.py 18
    python speedtest.py 19 zz500

Default benchmark: hs300
"""

import sys
import time
from pathlib import Path
from alpha191 import *


def get_alpha_func(alpha_num: int):
    """Get the alpha function by number (e.g., 17 -> alpha017)."""
    func_name = f"alpha{alpha_num:03d}"
    if hasattr(sys.modules['alpha191'], func_name):
        return getattr(sys.modules['alpha191'], func_name)
    raise ValueError(f"Alpha function '{func_name}' not found in alpha191 module")


def get_stock_codes(benchmark: str) -> list:
    """Get list of stock codes from the benchmark directory."""
    benchmark_dir = Path('bao') / benchmark
    if not benchmark_dir.exists():
        raise FileNotFoundError(f"Benchmark directory not found: {benchmark_dir}")
    
    # Get all CSV files and extract stock codes (without .csv extension)
    csv_files = sorted(benchmark_dir.glob('*.csv'))
    stock_codes = [f.stem for f in csv_files]
    return stock_codes


def main():
    if len(sys.argv) < 2:
        print("Usage: python speedtest.py <alpha_number> [benchmark]")
        print("Example: python speedtest.py 17 hs300")
        print("Example: python speedtest.py 18")
        print("Default benchmark: hs300")
        sys.exit(1)
    
    try:
        alpha_num = int(sys.argv[1])
    except ValueError:
        print(f"Error: Alpha number must be an integer, got '{sys.argv[1]}'")
        sys.exit(1)
    
    # Use hs300 as default benchmark if not provided
    benchmark = sys.argv[2] if len(sys.argv) > 2 else "hs300"
    
    # Get the alpha function
    try:
        alpha_func = get_alpha_func(alpha_num)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Get stock codes
    try:
        stock_codes = get_stock_codes(benchmark)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    print(f"Running alpha_{alpha_num:03d} on {len(stock_codes)} stocks from {benchmark}")
    print("-" * 60)
    
    results = []
    errors = []
    nan_count = 0
    start_time = time.time()
    
    for i, code in enumerate(stock_codes):
        try:
            # Calculate alpha for this stock
            result = alpha_func(code=code, benchmark=benchmark)
            results.append(result)
            # Check if result is NaN
            if result is not None and (isinstance(result, float) and result != result):  # NaN check
                nan_count += 1
        except Exception as e:
            errors.append((code, str(e)))
            results.append(None)
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    # Print results (20 per line)
    print(f"\nResults (alpha_{alpha_num:03d} values):")
    for i, result in enumerate(results):
        if result is not None:
            print(f"{result:12.6f}", end="")
        else:
            print(f"{'N/A':>12}", end="")
        
        # New line every 20 numbers
        if (i + 1) % 20 == 0:
            print()
        else:
            print(" ", end="")
    
    # Ensure final newline if last line wasn't complete
    if len(results) % 20 != 0:
        print()
    
    print("-" * 60)
    print(f"Total stocks processed: {len(stock_codes)}")
    print(f"Successful calculations: {len([r for r in results if r is not None])}")
    print(f"Failed calculations: {len(errors)}")
    print(f"NaN values: {nan_count}")
    print(f"Total running time: {elapsed:.3f} seconds")
    print(f"Average time per stock: {elapsed/len(stock_codes)*1000:.2f} ms")
    
    if errors:
        print(f"\nErrors ({len(errors)}):")
        for code, error in errors[:10]:  # Show first 10 errors
            print(f"  {code}: {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")


if __name__ == "__main__":
    main()

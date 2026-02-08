#!/usr/bin/env python3
"""
Full test script for all Alpha191 factors.

Usage:
    python fulltest.py [benchmark]

Example:
    python fulltest.py hs300
    python fulltest.py zz500
    python fulltest.py

Default benchmark: hs300

Output format: CSV with columns: alpha,total,success,failed,NaN,totalTime,averageTime
Note: Alpha functions 183, 165, 143, 30 do not exist and will show zeros
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


def test_alpha(alpha_num: int, benchmark: str, stock_codes: list) -> dict:
    """Test a single alpha function and return metrics."""
    try:
        alpha_func = get_alpha_func(alpha_num)
    except ValueError:
        return {
            'alpha': alpha_num,
            'total': 0,
            'success': 0,
            'failed': 0,
            'nan': 0,
            'total_time': 0,
            'average_time': 0,
            'error': f"Alpha function not found"
        }
    
    results = []
    errors = []
    nan_count = 0
    start_time = time.time()
    
    for code in stock_codes:
        try:
            result = alpha_func(code=code, benchmark=benchmark)
            results.append(result)
            # Check if result is NaN
            if result is not None and (isinstance(result, float) and result != result):
                nan_count += 1
        except Exception as e:
            errors.append(code)
            results.append(None)
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    return {
        'alpha': alpha_num,
        'total': len(stock_codes),
        'success': len([r for r in results if r is not None]),
        'failed': len(errors),
        'nan': nan_count,
        'total_time': elapsed,
        'average_time': elapsed / len(stock_codes) if len(stock_codes) > 0 else 0,
        'error': None
    }


def main():
    # Use hs300 as default benchmark if not provided
    benchmark = sys.argv[1] if len(sys.argv) > 1 else "hs300"
    
    if len(sys.argv) > 2:
        raise ValueError("Usage: python fulltest.py [benchmark]\nExample: python fulltest.py hs300\nDefault benchmark: hs300")
    
    # Get stock codes
    stock_codes = get_stock_codes(benchmark)
    
    print(f"Testing all alpha functions on {len(stock_codes)} stocks from {benchmark}")
    print("=" * 80)
    
    # CSV header
    print("alpha,total,success,failed,NaN,totalTime,averageTime")
    
    # Test all alpha functions (1-191)
    for alpha_num in range(1, 192):
        result = test_alpha(alpha_num, benchmark, stock_codes)
        
        if result['error']:
            print(f"{result['alpha']},0,0,0,0,0,0")
        else:
            print(f"{result['alpha']},{result['total']},{result['success']},{result['failed']},{result['nan']},{result['total_time']:.6f},{result['average_time']:.6f}")
        
        # Progress indicator
        if alpha_num % 20 == 0:
            print(f"# Progress: {alpha_num}/191", file=sys.stderr)
    
    print(f"# Full test completed for {benchmark}", file=sys.stderr)


if __name__ == "__main__":
    main()

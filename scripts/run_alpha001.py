"""
Validation script for Alpha001 factor.

Loads HS300 stock data and runs alpha_001 to validate implementation.

Usage:
    python scripts/run_alpha001.py

This script will:
1. Load HS300 stock data from bao/hs300/sh_600009.csv
2. Run alpha_001 factor computation
3. Print validation results and last 10 values
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from alpha191.alpha001 import alpha_001


def main():
    """Load data, run alpha_001, and print results."""
    # Load HS300 stock data
    data_path = 'bao/hs300/sh_600009.csv'

    print(f"Loading data from: {data_path}")

    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: File not found: {data_path}")
        print("Please ensure the data file exists.")
        return 1

    print(f"Loaded {len(df)} rows")
    print(f"Columns: {list(df.columns)}")
    print(f"Date range: {df['date'].iloc[0]} to {df['date'].iloc[-1]}")

    # Run alpha_001
    print("\nRunning alpha_001...")
    result = alpha_001(df)

    # Print results
    print("\n" + "="*60)
    print("Alpha001 Results")
    print("="*60)

    print(f"\nOutput shape: {result.shape}")
    print(f"NaN count: {result.isna().sum()}")
    print(f"Non-NaN count: {result.notna().sum()}")

    print("\nLast 10 values:")
    print("-"*40)
    last_10 = result.tail(10)
    for date, value in last_10.items():
        if pd.isna(value):
            print(f"{date.strftime('%Y-%m-%d')}: NaN")
        else:
            print(f"{date.strftime('%Y-%m-%d')}: {value:.6f}")

    # Validation checks
    print("\n" + "="*60)
    print("Validation Checks")
    print("="*60)

    # Check first 5 are NaN
    first_5_nan = np.all(result.isna().values[:5])
    print(f"First 5 values are NaN: {first_5_nan}")
    if not first_5_nan:
        print(f"  WARNING: First 5 values: {result.values[:5]}")

    # Check output length
    correct_length = len(result) == len(df)
    print(f"Output length matches input: {correct_length}")

    # Check index type
    correct_index = isinstance(result.index, pd.DatetimeIndex)
    print(f"Index is DatetimeIndex: {correct_index}")

    # Statistics for non-NaN values
    non_nan_values = result.dropna()
    if len(non_nan_values) > 0:
        print(f"\nNon-NaN values statistics:")
        print(f"  Min: {non_nan_values.min():.6f}")
        print(f"  Max: {non_nan_values.max():.6f}")
        print(f"  Mean: {non_nan_values.mean():.6f}")
        print(f"  Std: {non_nan_values.std():.6f}")

    print("\n" + "="*60)
    print("Validation Complete")
    print("="*60)

    return 0


if __name__ == '__main__':
    sys.exit(main())

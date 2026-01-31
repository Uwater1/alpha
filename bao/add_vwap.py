#!/usr/bin/env python3
"""
Add VWAP (Volume Weighted Average Price) column to CSV files in bao/hs300 and bao/zz500.

VWAP is calculated using the formula:
1. If 'amount' and 'volume' columns exist: vwap = amount / volume
2. The calculated vwap is validated to be between low and high prices
3. If validation fails, use OHLC average: (open + high + low + close) / 4
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path


def calculate_vwap(df: pd.DataFrame) -> pd.Series:
    """
    Calculate VWAP for a DataFrame.
    
    Args:
        df: DataFrame with columns: open, high, low, close, volume, amount
        
    Returns:
        Series with VWAP values
    """
    need_ohlc = True
    if {'amount', 'volume'}.issubset(df.columns):
        vwap_s = df['amount'] / df['volume'].replace(0, np.nan)
        valid = df['amount'].ne(0) & df['volume'].ne(0) & vwap_s.notna() & vwap_s.between(df['low'], df['high'])
        need_ohlc = ~valid.all()
        if not need_ohlc:
            vwap = vwap_s.values

    if need_ohlc:
        ohlc_avg = (df['open'] + df['high'] + df['low'] + df['close']) / 4
        if 'valid' in locals():
            vwap = vwap_s.where(valid, ohlc_avg).values
        else:
            vwap = ohlc_avg.values
    
    return pd.Series(vwap, index=df.index)


def process_csv_file(filepath: Path) -> None:
    """
    Read a CSV file, add VWAP column, and save it back.
    
    Args:
        filepath: Path to the CSV file
    """
    try:
        # Read CSV file
        df = pd.read_csv(filepath)
        
        # Check if required columns exist
        required_cols = {'open', 'high', 'low', 'close', 'volume', 'amount'}
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            print(f"Skipping {filepath}: missing columns {missing}")
            return
        
        # Calculate VWAP and round to 5 decimal places
        df['vwap'] = calculate_vwap(df).round(5)
        
        # Save back to CSV
        df.to_csv(filepath, index=False)
        print(f"Processed: {filepath}")
        
    except Exception as e:
        print(f"Error processing {filepath}: {e}")


def process_directory(directory: Path) -> None:
    """
    Process all CSV files in a directory.
    
    Args:
        directory: Path to the directory containing CSV files
    """
    if not directory.exists():
        print(f"Directory does not exist: {directory}")
        return
    
    csv_files = list(directory.glob("*.csv"))
    print(f"Found {len(csv_files)} CSV files in {directory}")
    
    for csv_file in csv_files:
        process_csv_file(csv_file)


def main():
    """Main function to process all CSV files in bao/hs300 and bao/zz500."""
    base_dir = Path(__file__).parent
    
    # Process hs300 directory
    hs300_dir = base_dir / "hs300"
    print("\n=== Processing hs300 ===")
    process_directory(hs300_dir)
    
    # Process zz500 directory
    zz500_dir = base_dir / "zz500"
    print("\n=== Processing zz500 ===")
    process_directory(zz500_dir)
    
    print("\n=== Done! ===")


if __name__ == "__main__":
    main()

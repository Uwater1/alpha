"""
Utility functions for Alpha191 factors.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from functools import lru_cache

# Cache for benchmark data
_benchmark_cache = {}

# Cache for required columns validation
_validated_files = set()

# Pre-constructed Path objects to avoid repeated construction
_BAO_HS300_PATH = Path('bao/hs300')
_BAO_ZZ500_PATH = Path('bao/zz500')
_BAO_BENCHMARK_PATHS = {
    'hs300': Path('bao/hs300.csv'),
    'zz500': Path('bao/zz500.csv'),
    'zz800': Path('bao/zz800.csv'),
} 


@lru_cache(maxsize=1024)
def _load_stock_csv_cached(code: str, benchmark: str) -> pd.DataFrame:
    """
    Internal cached function to load stock data from CSV file.
    Returns a copy to prevent mutation of cached data.
    """
    if benchmark not in ['hs300', 'zz500', 'zz800']:
        raise ValueError(
            f"Invalid benchmark '{benchmark}'. Must be one of: 'hs300', 'zz500', 'zz800'"
        )

    if benchmark == 'hs300':
        search_paths = [_BAO_HS300_PATH / f'{code}.csv']
    elif benchmark == 'zz500':
        search_paths = [_BAO_ZZ500_PATH / f'{code}.csv']
    else:  # zz800
        search_paths = [
            _BAO_HS300_PATH / f'{code}.csv',
            _BAO_ZZ500_PATH / f'{code}.csv'
        ]

    file_path = None
    for path in search_paths:
        if path.exists():
            file_path = path
            break

    if file_path is None:
        raise FileNotFoundError(f"File '{code}.csv' not found for benchmark {benchmark}")

    # OPTIMIZATION: Use index_col and eliminate inplace operations
    # This reduces memory copies and is faster
    df = pd.read_csv(
        str(file_path),
        parse_dates=['date'],
        index_col='date'
    )
    df = df.sort_index()
    
    # Ensure VWAP column exists
    df = _ensure_vwap(df)
    
    # OPTIMIZATION: Convert to float32 for cache efficiency
    # Reduces memory usage by ~50% with sufficient precision for price data
    float_cols = df.select_dtypes(include=['float64']).columns
    df[float_cols] = df[float_cols].astype(np.float32)
    
    return df


def _ensure_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure VWAP column exists in the DataFrame.
    
    If 'vwap' is missing, calculates it from amount/volume if available,
    otherwise uses the OHLC average. Invalid VWAP values (outside low-high
    range) are replaced with OHLC average.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at minimum OHLC columns
        
    Returns
    -------
    pd.DataFrame
        DataFrame with guaranteed 'vwap' column
    """
    if 'vwap' in df.columns:
        return df
    
    if {'amount', 'volume'}.issubset(df.columns):
        df = df.copy()
        df['vwap'] = df['amount'] / df['volume'].replace(0, np.nan)
        # Replace invalid (NaN or outside low-high) with OHLC average
        ohlc_avg = (df['open'] + df['high'] + df['low'] + df['close']) / 4
        invalid = df['vwap'].isna() | ~df['vwap'].between(df['low'], df['high'], inclusive='both')
        df['vwap'] = df['vwap'].where(~invalid, ohlc_avg)
    else:
        # Must have OHLC
        df = df.copy()
        df['vwap'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    
    return df


def load_stock_csv(code: str, benchmark: str = 'zz800') -> pd.DataFrame:
    """
    Load stock data from CSV file based on benchmark selection.
    Uses LRU caching to avoid repeated disk I/O for the same stock.

    Looks for {code}.csv in the appropriate directory based on benchmark:
    - 'hs300': bao/hs300/ only
    - 'zz500': bao/zz500/ only
    - 'zz800': bao/hs300/ or bao/zz500/ (searches both)

    Parameters
    ----------
    code : str
        Stock code (e.g., 'sh_600016', 'sz_000001')
    benchmark : str, default 'zz800'
        Benchmark selection: 'hs300', 'zz500', or 'zz800'

    Returns
    -------
    pd.DataFrame
        DataFrame with date as index, sorted by date
    """
    # Get cached data and return a copy to prevent mutation
    df = _load_stock_csv_cached(code, benchmark).copy()
    return df


@lru_cache(maxsize=8)
def _load_benchmark_csv_cached(benchmark: str) -> pd.DataFrame:
    """Internal cached loader for benchmark index data."""
    if benchmark not in ['hs300', 'zz500', 'zz800']:
        raise ValueError(
            f"Invalid benchmark '{benchmark}'. Must be one of: 'hs300', 'zz500', 'zz800'"
        )

    benchmark_path = _BAO_BENCHMARK_PATHS[benchmark]
    if not benchmark_path.exists():
        raise FileNotFoundError(f"Benchmark file '{benchmark_path}' not found")

    df = pd.read_csv(
        str(benchmark_path),
        parse_dates=['date'],
        index_col='date'
    )
    df = df.sort_index()
    return df


def load_benchmark_csv(benchmark: str) -> pd.DataFrame:
    """
    Load benchmark index data from CSV file.
    Uses module-level caching to avoid repeated disk I/O.

    Parameters
    ----------
    benchmark : str
        Benchmark selection: 'hs300', 'zz500', or 'zz800'

    Returns
    -------
    pd.DataFrame
        DataFrame with date as index, containing benchmark index data
    """
    df = _load_benchmark_csv_cached(benchmark)
    return df.copy(deep=False)


def run_alpha_factor(
    alpha_func,
    code: str,
    benchmark: str = 'zz800',
    end_date: str = "2026-01-23",
    lookback: int = 350
) -> float:
    """
    Generic runner for Alpha factors.
    Optimized to minimize DataFrame slicing operations.
    """
    df = load_stock_csv(code, benchmark=benchmark)
    
    # OPTIMIZATION: Combined date filtering in one operation
    # Filter by end_date first, then take last lookback rows
    if end_date is not None:
        end_dt = pd.to_datetime(end_date)
        df = df.loc[:end_dt]
    
    if len(df) < lookback:
        raise ValueError("insufficient history")
    
    # Take only the last lookback rows - use iloc for positional indexing
    df = df.iloc[-lookback:]

    # Load and merge benchmark data (cached internally)
    benchmark_df = load_benchmark_csv(benchmark)
    
    # Benchmark alignment optimization: Reindex benchmark series to stock's dates
    # This handles misalignment/suspensions correctly and is faster
    df['benchmark_close'] = benchmark_df['close'].reindex(df.index).values
    df['benchmark_open'] = benchmark_df['open'].reindex(df.index).values
    df['benchmark_index_close'] = df['benchmark_close']
    df['benchmark_index_open'] = df['benchmark_open']

    value = alpha_func(df).iloc[-1]
    return float(value) if not np.isnan(value) else np.nan

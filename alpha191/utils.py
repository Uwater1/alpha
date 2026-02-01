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


@lru_cache(maxsize=512)
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
        search_paths = [Path('bao/hs300') / f'{code}.csv']
    elif benchmark == 'zz500':
        search_paths = [Path('bao/zz500') / f'{code}.csv']
    else:  # zz800
        search_paths = [
            Path('bao/hs300') / f'{code}.csv',
            Path('bao/zz500') / f'{code}.csv'
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
    
    # Add VWAP column if missing, approximating as amount / volume
    if 'vwap' not in df.columns:
        need_ohlc = False
        if {'amount', 'volume'}.issubset(df.columns):
            vwap_calc = df['amount'] / df['volume'].replace(0, np.nan)
            valid = (
                df['amount'].ne(0) & df['volume'].ne(0) & vwap_calc.notna() & vwap_calc.between(df['low'], df['high'])
            )
            need_ohlc = ~valid.all()
            df['vwap'] = vwap_calc
        else:
            need_ohlc = True

        if need_ohlc:
            if not {'open', 'high', 'low', 'close'}.issubset(df.columns):
                raise ValueError(
                    "VWAP column not found and cannot be approximated (missing 'open', 'high', 'low', or 'close' columns)"
                )
            ohlc_avg = (df['open'] + df['high'] + df['low'] + df['close']) / 4
            if 'valid' in locals():
                df['vwap'] = df['vwap'].where(valid, ohlc_avg)
            else:
                df['vwap'] = ohlc_avg
    
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

    benchmark_path = Path(f'bao/{benchmark}.csv')
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
        # Use boolean indexing which is more efficient than .loc for slicing
        mask = df.index <= end_date
        df = df[mask]
    
    if len(df) < lookback:
        raise ValueError("insufficient history")
    
    # Take only the last lookback rows - use iloc for positional indexing
    df = df.iloc[-lookback:]

    # Load and merge benchmark data (cached internally)
    benchmark_df = load_benchmark_csv(benchmark)
    
    # OPTIMIZATION: Filter benchmark data efficiently
    if end_date is not None:
        bench_mask = benchmark_df.index <= end_date
        benchmark_df = benchmark_df[bench_mask]
    
    benchmark_df = benchmark_df.iloc[-lookback:]

    # Add benchmark columns to stock DataFrame
    # OPTIMIZATION: Use direct assignment with reindexed benchmark data
    df = df.copy()  # Ensure we don't modify the cached copy
    df['benchmark_close'] = benchmark_df['close'].values
    df['benchmark_open'] = benchmark_df['open'].values
    df['benchmark_index_close'] = benchmark_df['close'].values
    df['benchmark_index_open'] = benchmark_df['open'].values

    value = alpha_func(df).iloc[-1]
    return float(value) if not np.isnan(value) else np.nan

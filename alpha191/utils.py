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
PROJECT_ROOT = Path(__file__).parent.parent
_BAO_HS300_PATH = PROJECT_ROOT / 'bao/hs300'
_BAO_ZZ500_PATH = PROJECT_ROOT / 'bao/zz500'
_BAO_BENCHMARK_PATHS = {
    'hs300': PROJECT_ROOT / 'bao/hs300.csv',
    'zz500': PROJECT_ROOT / 'bao/zz500.csv',
    'zz800': PROJECT_ROOT / 'bao/zz800.csv',
} 


def _ensure_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure VWAP column exists in the DataFrame.
    """
    if 'vwap' in df.columns:
        return df
    
    if {'amount', 'volume'}.issubset(df.columns):
        df = df.copy()
        df['vwap'] = df['amount'] / df['volume'].replace(0, np.nan)
        ohlc_avg = (df['open'] + df['high'] + df['low'] + df['close']) / 4
        invalid = df['vwap'].isna() | ~df['vwap'].between(df['low'], df['high'], inclusive='both')
        df['vwap'] = df['vwap'].where(~invalid, ohlc_avg)
    else:
        df = df.copy()
        df['vwap'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    
    return df


@lru_cache(maxsize=1024)
def _load_stock_csv_cached(code: str, benchmark: str) -> pd.DataFrame:
    """
    Internal cached function to load stock data from CSV file.
    """
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

    df = pd.read_csv(
        str(file_path),
        parse_dates=['date'],
        index_col='date'
    )
    df = df.sort_index()
    df = _ensure_vwap(df)
    
    float_cols = df.select_dtypes(include=['float64']).columns
    df[float_cols] = df[float_cols].astype(np.float32)
    
    return df


def load_stock_csv(code: str, benchmark: str = 'zz800') -> pd.DataFrame:
    """
    Load stock data from CSV file based on benchmark selection.
    """
    if benchmark not in ['hs300', 'zz500', 'zz800']:
        raise ValueError(f"Invalid benchmark {benchmark}")
        
    return _load_stock_csv_cached(code, benchmark).copy()


@lru_cache(maxsize=8)
def _load_benchmark_csv_cached(benchmark: str) -> pd.DataFrame:
    """Internal cached loader for benchmark index data."""
    if benchmark not in ['hs300', 'zz500', 'zz800']:
        raise ValueError(f"Invalid benchmark {benchmark}")

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
    """ Load benchmark index data from CSV file. """
    return _load_benchmark_csv_cached(benchmark).copy()


def run_alpha_factor(
    alpha_func,
    code: str,
    benchmark: str = 'zz800',
    end_date: str = "2026-01-23",
    lookback: int = 350
) -> float:
    """ Generic runner for Alpha factors. """
    df = load_stock_csv(code, benchmark=benchmark)
    
    if end_date is not None:
        end_dt = pd.to_datetime(end_date)
        df = df.loc[:end_dt]
    
    if len(df) < lookback:
        raise ValueError("insufficient history")
    
    df = df.iloc[-lookback:]
    benchmark_df = load_benchmark_csv(benchmark)
    
    df['benchmark_close'] = benchmark_df['close'].reindex(df.index)
    df['benchmark_open'] = benchmark_df['open'].reindex(df.index)
    df['benchmark_index_close'] = df['benchmark_close']
    df['benchmark_index_open'] = df['benchmark_open']

    value = alpha_func(df).iloc[-1]
    return float(value) if not np.isnan(value) else np.nan

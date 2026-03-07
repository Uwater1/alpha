"""
Utility functions for Alpha191 factors.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from functools import lru_cache
from typing import List, Optional, Union, Any

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

    # Faster CSV loading: skip parse_dates, use engine='c'
    df = pd.read_csv(
        str(file_path),
        engine='c',
        low_memory=False,
        memory_map=True
    )
    # Manual date conversion is often faster than parse_dates
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)
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


def get_benchmark_members(benchmark: str) -> List[str]:
    """Get stock codes for the specified benchmark with caching."""
    # Use a simple cache key
    cache_key = f"members_{benchmark}"
    if cache_key in _benchmark_cache:
        return _benchmark_cache[cache_key]
    
    if benchmark == "hs300":
        df = pd.read_csv(PROJECT_ROOT / "bao/hs300_l.csv")
    elif benchmark == "zz500":
        df = pd.read_csv(PROJECT_ROOT / "bao/zz500-l.csv")
    elif benchmark == "zz800":
        df1 = pd.read_csv(PROJECT_ROOT / "bao/hs300_l.csv")
        df2 = pd.read_csv(PROJECT_ROOT / "bao/zz500-l.csv")
        df = pd.concat([df1, df2])
    else:
        raise ValueError(f"Invalid benchmark: {benchmark}")
    
    # Standardize codes from sh.600000 to sh_600000
    codes = df['code'].str.replace('.', '_', regex=False).tolist()
    
    # Cache the result
    _benchmark_cache[cache_key] = codes
    return codes


def format_alpha_name(alpha_name: str) -> str:
    """Convert input like '1' or '42' to 'alpha001' or 'alpha042' format.
    If input already starts with 'alpha', return it in lowercase format."""
    if alpha_name.lower().startswith("alpha"):
        return alpha_name.lower()
    else:
        # Convert number to zero-padded format
        try:
            num = int(alpha_name)
            return f"alpha{num:03d}"
        except ValueError:
            raise ValueError(f"Invalid alpha name: {alpha_name}. Expected format: '1' or 'alpha001'")


def get_alpha_func(alpha_id: Union[int, str], use_df: bool = False, ignore_errors: bool = False) -> Optional[Any]:
    """
    Get the alpha function by number or name.

    Args:
        alpha_id: Alpha number (e.g., 17) or name (e.g., "alpha017")
        use_df: If True, return the function that takes a DataFrame (alpha_XXX).
               If False, return the function that takes code/benchmark (alphaXXX).
        ignore_errors: If True, return None on failure instead of raising ValueError.

    Returns:
        The alpha function or None if not found and ignore_errors is True.
    """
    import importlib
    try:
        alpha_name = format_alpha_name(str(alpha_id))
        module = importlib.import_module(f"alpha191.{alpha_name}")

        if use_df:
            # Try alpha_XXX first (preferred for DataFrame input)
            func_name = f"alpha_{int(alpha_name[5:]):03d}"
            if hasattr(module, func_name):
                return getattr(module, func_name)
            # Fallback to alphaXXX
            if hasattr(module, alpha_name):
                return getattr(module, alpha_name)
        else:
            # Try alphaXXX first (preferred for code/benchmark input)
            if hasattr(module, alpha_name):
                return getattr(module, alpha_name)
            # Fallback to alpha_XXX
            func_name = f"alpha_{int(alpha_name[5:]):03d}"
            if hasattr(module, func_name):
                return getattr(module, func_name)

    except (ImportError, ModuleNotFoundError, ValueError):
        pass

    if ignore_errors:
        return None
    raise ValueError(f"Alpha function for '{alpha_id}' not found")


def get_stock_codes(benchmark: str) -> List[str]:
    """Get list of stock codes available in the benchmark directory."""
    benchmark_dir = PROJECT_ROOT / 'bao' / benchmark
    if not benchmark_dir.exists():
        raise FileNotFoundError(f"Benchmark directory not found: {benchmark_dir}")

    # Get all CSV files and extract stock codes (without .csv extension)
    csv_files = sorted(benchmark_dir.glob('*.csv'))
    stock_codes = [f.stem for f in csv_files]
    return stock_codes


def _load_single_stock_with_alpha(args):
    """
    Helper function for parallel loading of stock data and alpha computation.
    Returns (code, alpha_series, price_series) or (code, None, None) on error.
    """
    code, alpha_func, benchmark, benchmark_df = args
    try:
        df = load_stock_csv(code, benchmark=benchmark)
        df['benchmark_close'] = benchmark_df['close'].reindex(df.index)
        df['benchmark_open'] = benchmark_df['open'].reindex(df.index)
        
        alpha_series = alpha_func(df).astype(np.float32)
        price_series = df['close'].astype(np.float32)
        
        # Optimize memory by downcasting to float32
        return (code, alpha_series, price_series)
    except Exception:
        return (code, None, None)


def parallel_load_stocks_with_alpha(
    codes: List[str],
    alpha_func,
    benchmark: str,
    n_jobs: int = -1,
    show_progress: bool = True
) -> tuple:
    """
    Load multiple stocks in parallel and compute alpha values.
    
    Parameters:
    -----------
    codes : List[str]
        List of stock codes to load
    alpha_func : callable
        Alpha function to compute on each stock
    benchmark : str
        Benchmark name (hs300, zz500, zz800)
    n_jobs : int
        Number of parallel jobs. -1 uses all CPU cores.
    show_progress : bool
        Whether to show progress information
        
    Returns:
    --------
    tuple : (factor_results, price_results)
        Dictionaries mapping code to alpha series and price series
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from multiprocessing import cpu_count
    
    # Determine number of workers
    if n_jobs == -1:
        n_jobs = cpu_count()
    n_jobs = min(n_jobs, len(codes))
    
    # Load benchmark data once (shared across all workers)
    benchmark_df = load_benchmark_csv(benchmark)
    
    # Prepare arguments for parallel execution
    args_list = [(code, alpha_func, benchmark, benchmark_df) for code in codes]
    
    if show_progress:
        print(f"Loading {len(codes)} stocks using {n_jobs} threads...")
    
    # Threaded execution (usually faster for I/O and low-overhead for data return)
    results = []
    if n_jobs > 1:
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            # Map returns results in order
            results = list(executor.map(_load_single_stock_with_alpha, args_list))
    else:
        # Single-threaded fallback
        results = [_load_single_stock_with_alpha(args) for args in args_list]
    
    # Collect results
    factor_results = {}
    price_results = {}
    failed_count = 0
    
    for code, alpha_series, price_series in results:
        if alpha_series is not None:
            factor_results[code] = alpha_series
            price_results[code] = price_series
        else:
            failed_count += 1
    
    if show_progress:
        success_count = len(factor_results)
        print(f"Successfully loaded {success_count}/{len(codes)} stocks ({failed_count} failed)")
    
    return factor_results, price_results


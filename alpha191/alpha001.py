"""
Alpha001 factor implementation.

Formula:
    alpha_001 = -1 * CORR(
        RANK(DELTA(LOG(VOLUME), 1)),
        RANK((CLOSE-OPEN)/OPEN),
        6
    )
"""

import numpy as np
import pandas as pd
from pathlib import Path
from .operators import ts_rank, rolling_corr


def alpha_001(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha001 factor.

    Parameters
    ----------
    df : pd.DataFrame
        Single-stock DataFrame with columns: date, open, high, low, close, volume, amount
        Data should be date-sorted.

    Returns
    -------
    pd.Series
        Alpha001 values indexed by date

    Usage:
        >>> import pandas as pd
        >>> from alpha191.alpha001 import alpha_001, alpha001
        >>> # Using alpha001 with benchmark parameter
        >>> result = alpha001('sh_600009', benchmark='zz800', end_date="2026-01-23", lookback=350)
        >>> print(result)
        >>> # Or compute alpha_001 directly from DataFrame
        >>> df = pd.read_csv('bao/hs300/sh_600009.csv')
        >>> result = alpha_001(df)
        >>> print(result.tail(10))
    """
    # Ensure we have required columns
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in DataFrame")

    # Get date index
    if 'date' in df.columns:
        index = pd.to_datetime(df['date'])
    else:
        index = df.index

    # Step 1: Compute DELTA(LOG(volume), 1) = log(volume[t]) - log(volume[t-1])
    # Handle zero volume values by replacing with NaN before log
    volume = df['volume'].replace(0, np.nan).values
    log_volume = np.log(volume)
    delta_log_volume = np.diff(log_volume, prepend=np.nan)

    # Step 2: Compute RANK of delta_log_volume over rolling window (default 6)
    rank_delta_volume = ts_rank(delta_log_volume, window=6)

    # Step 3: Compute (close - open) / open
    returns_ratio = (df['close'].values - df['open'].values) / df['open'].values

    # Step 4: Compute RANK of returns_ratio over rolling window (default 6)
    rank_returns = ts_rank(returns_ratio, window=6)

    # Step 5: Compute CORR of the two ranks with window=6
    correlation = rolling_corr(rank_delta_volume, rank_returns, window=6)

    # Step 6: Multiply by -1
    alpha_values = -1 * correlation

    # Step 7: Return as pd.Series indexed by date
    result = pd.Series(alpha_values, index=index, name='alpha_001')

    return result


def load_stock_csv(code: str, benchmark: str = 'zz800') -> pd.DataFrame:
    """
    Load stock data from CSV file based on benchmark selection.

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

    Raises
    ------
    FileNotFoundError
        If file not found in the specified directory/directories
    ValueError
        If benchmark is not one of 'hs300', 'zz500', or 'zz800'
    """
    # Validate benchmark parameter
    if benchmark not in ['hs300', 'zz500', 'zz800']:
        raise ValueError(
            f"Invalid benchmark '{benchmark}'. Must be one of: 'hs300', 'zz500', 'zz800'"
        )

    # Determine search paths based on benchmark
    if benchmark == 'hs300':
        search_paths = [Path('bao/hs300') / f'{code}.csv']
    elif benchmark == 'zz500':
        search_paths = [Path('bao/zz500') / f'{code}.csv']
    else:  # zz800 - search both directories
        search_paths = [
            Path('bao/hs300') / f'{code}.csv',
            Path('bao/zz500') / f'{code}.csv'
        ]

    # Find the file
    file_path = None
    for path in search_paths:
        if path.exists():
            file_path = path
            break

    if file_path is None:
        if benchmark == 'zz800':
            raise FileNotFoundError(
                f"File '{code}.csv' not found in bao/hs300/ or bao/zz500/ directories."
            )
        else:
            dir_name = benchmark
            raise FileNotFoundError(
                f"File '{code}.csv' not found in bao/{dir_name}/ directory."
            )

    df = pd.read_csv(file_path)

    # Parse date column and set as index
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

    # Sort by date
    df.sort_index(inplace=True)

    return df


def alpha001(
    code: str,
    benchmark: str = 'zz800',
    end_date: str = "2026-01-23",
    lookback: int = 350
) -> float:
    """
    Compute Alpha001 factor value for a stock at a specific date.

    Parameters
    ----------
    code : str
        Stock code (e.g., 'sh_600016', 'sz_000001')
    benchmark : str, default 'zz800'
        Benchmark selection: 'hs300', 'zz500', or 'zz800'
        Determines which directory to search for stock data:
        - 'hs300': bao/hs300/ only
        - 'zz500': bao/zz500/ only
        - 'zz800': bao/hs300/ or bao/zz500/ (searches both)
    end_date : str, default "2026-01-23"
        End date for the computation (format: YYYY-MM-DD)
    lookback : int, default 350
        Number of trading days to look back

    Returns
    -------
    float
        Alpha001 factor value at the end date, or np.nan if value is NaN

    Raises
    ------
    ValueError
        If insufficient history (less than lookback rows) or invalid benchmark
    FileNotFoundError
        If stock file not found in the specified directory/directories
    """
    df = load_stock_csv(code, benchmark=benchmark)

    # Filter data to date <= end_date
    df = df.loc[:end_date]

    # Check if sufficient history exists
    if len(df) < lookback:
        raise ValueError("insufficient history")

    # Keep the last lookback rows
    df = df.iloc[-lookback:]

    # Call alpha_001 and get the last value
    value = alpha_001(df).iloc[-1]

    # Return float or np.nan
    return float(value) if not np.isnan(value) else np.nan

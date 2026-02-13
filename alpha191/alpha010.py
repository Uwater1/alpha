"""
Alpha010 factor implementation.

Formula:
    alpha_010 = (RANK(MAX(((RET<0)?STD(RET,20):CLOSE)^2),5))
"""

import numpy as np
import pandas as pd
from .operators import ts_max, ts_std, rank
from .utils import run_alpha_factor


def alpha_010(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha010 factor (inverted).

    Formula (inverted):
        alpha_010 = (RANK(MAX(((RET>=0)?STD(RET,20):CLOSE)^2),5))
    """
    # Ensure we have required columns
    required_cols = ['close']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in DataFrame")

    if 'date' in df.columns:
        index = pd.to_datetime(df['date'])
    else:
        index = df.index

    close = df['close'].values

    # Step 1: Compute RET (daily return)
    # RET = (close - delay(close, 1)) / delay(close, 1)
    delay_close = np.roll(close, 1)
    delay_close[0] = np.nan
    ret = np.full(len(close), np.nan)
    valid_mask = ~np.isnan(delay_close) & (delay_close != 0)
    ret[valid_mask] = (close[valid_mask] - delay_close[valid_mask]) / delay_close[valid_mask]

    # Step 2: Compute STD(RET, 20)
    std_ret = ts_std(ret, 20)

    # Step 3: Compute ((RET<0)?STD(RET,20):CLOSE)
    # If RET < 0, use STD(RET,20), otherwise use CLOSE
    conditional_value = np.where(ret >= 0, std_ret, close)

    # Step 4: Compute squared value
    squared_value = conditional_value ** 2

    # Step 5: Compute MAX with window=5
    max_squared = ts_max(squared_value, 5)

    # Step 6: Compute RANK (cross-sectional)
    # For single stock time series, rank returns 0.5
    # We use the max values directly
    alpha_values = max_squared

    return pd.Series(alpha_values, index=index, name='alpha_010')


def alpha010(
    code: str,
    benchmark: str = 'zz800',
    end_date: str = "2026-01-23",
    lookback: int = 350
) -> float:
    """
    Compute Alpha010 factor value for a stock at a specific date.
    """
    return run_alpha_factor(alpha_010, code, benchmark, end_date, lookback)

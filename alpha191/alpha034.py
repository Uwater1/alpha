"""
Alpha034 factor implementation.

Formula:
    alpha_034 = MEAN(CLOSE,12)/CLOSE
"""

import numpy as np
import pandas as pd
from .operators import ts_mean
from .utils import run_alpha_factor


def alpha_034(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha034 factor.

    Formula:
        alpha_034 = MEAN(CLOSE,12)/CLOSE
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

    # Step 1: Compute MEAN(CLOSE, 12)
    mean_close = ts_mean(close, 12)

    # Step 2: Compute MEAN(CLOSE,12)/CLOSE
    alpha_values = np.full(len(close), np.nan)
    valid_mask = ~np.isnan(close) & (close != 0) & ~np.isnan(mean_close)
    alpha_values[valid_mask] = mean_close[valid_mask] / close[valid_mask]

    return pd.Series(alpha_values, index=index, name='alpha_034')


def alpha034(
    code: str,
    benchmark: str = 'zz800',
    end_date: str = "2026-01-23",
    lookback: int = 350
) -> float:
    """
    Compute Alpha034 factor value for a stock at a specific date.
    """
    return run_alpha_factor(alpha_034, code, benchmark, end_date, lookback)

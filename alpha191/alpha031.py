"""
Alpha031 factor implementation.

Formula:
    alpha_031 = (CLOSE-MEAN(CLOSE,12))/MEAN(CLOSE,12)*100
"""

import numpy as np
import pandas as pd
from .operators import ts_mean
from .utils import run_alpha_factor


def alpha_031(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha031 factor.

    Formula:
        alpha_031 = (CLOSE-MEAN(CLOSE,12))/MEAN(CLOSE,12)*100
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

    # Step 2: Compute (CLOSE-MEAN(CLOSE,12))/MEAN(CLOSE,12)*100
    alpha_values = np.full(len(close), np.nan)
    valid_mask = ~np.isnan(mean_close) & (mean_close != 0)
    alpha_values[valid_mask] = (close[valid_mask] - mean_close[valid_mask]) / mean_close[valid_mask] * 100

    return pd.Series(alpha_values, index=index, name='alpha_031')


def alpha031(
    code: str,
    benchmark: str = 'zz800',
    end_date: str = "2026-01-23",
    lookback: int = 350
) -> float:
    """
    Compute Alpha031 factor value for a stock at a specific date.
    """
    return run_alpha_factor(alpha_031, code, benchmark, end_date, lookback)

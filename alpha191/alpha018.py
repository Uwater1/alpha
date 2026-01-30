"""
Alpha018 factor implementation.

Formula:
    alpha_018 = CLOSE/DELAY(CLOSE,5)
"""

import numpy as np
import pandas as pd
from .operators import delay
from .utils import run_alpha_factor


def alpha_018(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha018 factor.

    Formula:
        alpha_018 = CLOSE/DELAY(CLOSE,5)
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

    # Step 1: Compute DELAY(CLOSE, 5)
    delay_close = delay(close, 5)

    # Step 2: Compute CLOSE/DELAY(CLOSE,5)
    # Handle division by zero
    ratio = np.full(len(close), np.nan)
    valid_mask = ~np.isnan(delay_close) & (delay_close != 0)
    ratio[valid_mask] = close[valid_mask] / delay_close[valid_mask]

    alpha_values = ratio

    return pd.Series(alpha_values, index=index, name='alpha_018')


def alpha018(
    code: str,
    benchmark: str = 'zz800',
    end_date: str = "2026-01-23",
    lookback: int = 350
) -> float:
    """
    Compute Alpha018 factor value for a stock at a specific date.
    """
    return run_alpha_factor(alpha_018, code, benchmark, end_date, lookback)

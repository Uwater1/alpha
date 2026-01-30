"""
Alpha020 factor implementation.

Formula:
    alpha_020 = (CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*100
"""

import numpy as np
import pandas as pd
from .operators import delay
from .utils import run_alpha_factor


def alpha_020(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha020 factor.

    Formula:
        alpha_020 = (CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*100
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

    # Step 1: Compute DELAY(CLOSE, 6)
    delay_close = delay(close, 6)

    # Step 2: Compute (CLOSE-DELAY(CLOSE,6))
    diff = close - delay_close

    # Step 3: Compute (CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*100
    # Handle division by zero
    ratio = np.full(len(close), np.nan)
    valid_mask = ~np.isnan(delay_close) & (delay_close != 0)
    ratio[valid_mask] = diff[valid_mask] / delay_close[valid_mask]

    alpha_values = ratio * 100

    return pd.Series(alpha_values, index=index, name='alpha_020')


def alpha020(
    code: str,
    benchmark: str = 'zz800',
    end_date: str = "2026-01-23",
    lookback: int = 350
) -> float:
    """
    Compute Alpha020 factor value for a stock at a specific date.
    """
    return run_alpha_factor(alpha_020, code, benchmark, end_date, lookback)

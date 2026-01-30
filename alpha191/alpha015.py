"""
Alpha015 factor implementation.

Formula:
    alpha_015 = OPEN/DELAY(CLOSE,1)-1
"""

import numpy as np
import pandas as pd
from .operators import delay
from .utils import run_alpha_factor


def alpha_015(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha015 factor.

    Formula:
        alpha_015 = OPEN/DELAY(CLOSE,1)-1
    """
    # Ensure we have required columns
    required_cols = ['open', 'close']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in DataFrame")

    if 'date' in df.columns:
        index = pd.to_datetime(df['date'])
    else:
        index = df.index

    open_price = df['open'].values
    close = df['close'].values

    # Step 1: Compute DELAY(CLOSE, 1)
    delay_close = delay(close, 1)

    # Step 2: Compute OPEN/DELAY(CLOSE,1)-1
    # Handle division by zero
    ratio = np.full(len(open_price), np.nan)
    valid_mask = ~np.isnan(delay_close) & (delay_close != 0)
    ratio[valid_mask] = open_price[valid_mask] / delay_close[valid_mask]

    alpha_values = ratio - 1

    return pd.Series(alpha_values, index=index, name='alpha_015')


def alpha015(
    code: str,
    benchmark: str = 'zz800',
    end_date: str = "2026-01-23",
    lookback: int = 350
) -> float:
    """
    Compute Alpha015 factor value for a stock at a specific date.
    """
    return run_alpha_factor(alpha_015, code, benchmark, end_date, lookback)

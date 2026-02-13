"""
Alpha024 factor implementation.

Formula:
    alpha_024 = SMA(CLOSE-DELAY(CLOSE,5),5,1)
"""

import numpy as np
import pandas as pd
from .operators import delay, sma
from .utils import run_alpha_factor


def alpha_024(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha024 factor (inverted).

    Formula (inverted):
        alpha_024 = SMA(DELAY(CLOSE,5)-CLOSE,5,1)
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

    # Step 2: Compute DELAY(CLOSE, 5) - CLOSE
    diff = delay_close - close

    # Step 3: Compute SMA(..., 5, 1)
    alpha_values = sma(diff, 5, 1)

    return pd.Series(alpha_values, index=index, name='alpha_024')


def alpha024(
    code: str,
    benchmark: str = 'zz800',
    end_date: str = "2026-01-23",
    lookback: int = 350
) -> float:
    """
    Compute Alpha024 factor value for a stock at a specific date.
    """
    return run_alpha_factor(alpha_024, code, benchmark, end_date, lookback)

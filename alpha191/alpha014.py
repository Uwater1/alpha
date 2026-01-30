"""
Alpha014 factor implementation.

Formula:
    alpha_014 = CLOSE-DELAY(CLOSE,5)
"""

import numpy as np
import pandas as pd
from .operators import delay
from .utils import run_alpha_factor


def alpha_014(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha014 factor.

    Formula:
        alpha_014 = CLOSE-DELAY(CLOSE,5)
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

    # Step 2: Compute CLOSE-DELAY(CLOSE,5)
    alpha_values = close - delay_close

    return pd.Series(alpha_values, index=index, name='alpha_014')


def alpha014(
    code: str,
    benchmark: str = 'zz800',
    end_date: str = "2026-01-23",
    lookback: int = 350
) -> float:
    """
    Compute Alpha014 factor value for a stock at a specific date.
    """
    return run_alpha_factor(alpha_014, code, benchmark, end_date, lookback)

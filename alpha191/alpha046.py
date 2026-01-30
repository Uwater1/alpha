"""
Alpha046 factor implementation.

Formula:
    alpha_046 = (MEAN(CLOSE,3)+MEAN(CLOSE,6)+MEAN(CLOSE,12)+MEAN(CLOSE,24))/(4*CLOSE)
"""

import numpy as np
import pandas as pd
from .operators import ts_mean
from .utils import run_alpha_factor


def alpha_046(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha046 factor.

    Formula:
        alpha_046 = (MEAN(CLOSE,3)+MEAN(CLOSE,6)+MEAN(CLOSE,12)+MEAN(CLOSE,24))/(4*CLOSE)
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

    # Step 1: Compute MEAN(CLOSE, 3)
    mean_close_3 = ts_mean(close, 3)

    # Step 2: Compute MEAN(CLOSE, 6)
    mean_close_6 = ts_mean(close, 6)

    # Step 3: Compute MEAN(CLOSE, 12)
    mean_close_12 = ts_mean(close, 12)

    # Step 4: Compute MEAN(CLOSE, 24)
    mean_close_24 = ts_mean(close, 24)

    # Step 5: Compute the numerator
    numerator = mean_close_3 + mean_close_6 + mean_close_12 + mean_close_24

    # Step 6: Compute the denominator
    denominator = 4 * close

    # Step 7: Compute the final result
    alpha_values = numerator / denominator

    return pd.Series(alpha_values, index=index, name='alpha_046')


def alpha046(
    code: str,
    benchmark: str = 'zz800',
    end_date: str = "2026-01-23",
    lookback: int = 350
) -> float:
    """
    Compute Alpha046 factor value for a stock at a specific date.
    """
    return run_alpha_factor(alpha_046, code, benchmark, end_date, lookback)
"""
Alpha022 factor implementation.

Formula:
    alpha_022 = SMEAN(((CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6)-DELAY((CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6),3)),12,1)
"""

import numpy as np
import pandas as pd
from .operators import ts_mean, delay, sma
from .utils import run_alpha_factor


def alpha_022(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha022 factor.

    Formula:
        alpha_022 = SMEAN(((CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6)-DELAY((CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6),3)),12,1)
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

    # Step 1: Compute MEAN(CLOSE, 6)
    mean_close = ts_mean(close, 6)

    # Step 2: Compute (CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6)
    # Handle division by zero
    ratio = np.full(len(close), np.nan)
    valid_mask = ~np.isnan(mean_close) & (mean_close != 0)
    ratio[valid_mask] = (close[valid_mask] - mean_close[valid_mask]) / mean_close[valid_mask]

    # Step 3: Compute DELAY((CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6), 3)
    delay_ratio = delay(ratio, 3)

    # Step 4: Compute ((CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6)-DELAY(...,3))
    diff = ratio - delay_ratio

    # Step 5: Compute SMEAN(..., 12, 1)
    alpha_values = sma(diff, 12, 1)

    return pd.Series(alpha_values, index=index, name='alpha_022')


def alpha022(
    code: str,
    benchmark: str = 'zz800',
    end_date: str = "2026-01-23",
    lookback: int = 350
) -> float:
    """
    Compute Alpha022 factor value for a stock at a specific date.
    """
    return run_alpha_factor(alpha_022, code, benchmark, end_date, lookback)

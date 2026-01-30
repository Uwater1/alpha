"""
Alpha005 factor implementation.

Formula:
    alpha_005 = (-1*TSMAX(CORR(TSRANK(VOLUME,5),TSRANK(HIGH,5),5),3))
"""

import numpy as np
import pandas as pd
from .operators import ts_rank, rolling_corr, ts_max
from .utils import run_alpha_factor


def alpha_005(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha005 factor.

    Formula:
        alpha_005 = (-1*TSMAX(CORR(TSRANK(VOLUME,5),TSRANK(HIGH,5),5),3))
    """
    # Ensure we have required columns
    required_cols = ['volume', 'high']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in DataFrame")

    if 'date' in df.columns:
        index = pd.to_datetime(df['date'])
    else:
        index = df.index

    volume = df['volume'].values
    high = df['high'].values

    # Step 1: Compute TSRANK(VOLUME, 5)
    rank_volume = ts_rank(volume, window=5)

    # Step 2: Compute TSRANK(HIGH, 5)
    rank_high = ts_rank(high, window=5)

    # Step 3: Compute CORR with window=5
    correlation = rolling_corr(rank_volume, rank_high, window=5)

    # Step 4: Compute TSMAX with window=3
    max_corr = ts_max(correlation, 3)

    # Step 5: Final result
    alpha_values = -1 * max_corr

    return pd.Series(alpha_values, index=index, name='alpha_005')


def alpha005(
    code: str,
    benchmark: str = 'zz800',
    end_date: str = "2026-01-23",
    lookback: int = 350
) -> float:
    """
    Compute Alpha005 factor value for a stock at a specific date.
    """
    return run_alpha_factor(alpha_005, code, benchmark, end_date, lookback)

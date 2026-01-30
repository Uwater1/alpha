"""
Alpha032 factor implementation.

Formula:
    alpha_032 = (-1*SUM(RANK(CORR(RANK(HIGH),RANK(VOLUME),3)),3))
"""

import numpy as np
import pandas as pd
from .operators import rank, rolling_corr, ts_sum, ts_rank
from .utils import run_alpha_factor


def alpha_032(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha032 factor.

    Formula:
        alpha_032 = (-1*SUM(RANK(CORR(RANK(HIGH),RANK(VOLUME),3)),3))
    """
    # Ensure we have required columns
    required_cols = ['high', 'volume']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in DataFrame")

    if 'date' in df.columns:
        index = pd.to_datetime(df['date'])
    else:
        index = df.index

    high = df['high'].values
    volume = df['volume'].values

    # Step 1: Compute RANK(HIGH) inside CORR(..., 3)
    rank_high = ts_rank(high, 3)

    # Step 2: Compute RANK(VOLUME) inside CORR(..., 3)
    rank_volume = ts_rank(volume, 3)

    # Step 3: Compute CORR(RANK(HIGH), RANK(VOLUME), 3)
    corr_val = rolling_corr(rank_high, rank_volume, 3)

    # Step 4: Compute RANK(CORR(...)) inside SUM(..., 3)
    rank_corr = ts_rank(corr_val, 3)

    # Step 5: Compute SUM(RANK(CORR(...)), 3)
    sum_rank = ts_sum(rank_corr, 3)

    # Step 6: Compute -1 * SUM(...)
    alpha_values = -1 * sum_rank

    return pd.Series(alpha_values, index=index, name='alpha_032')


def alpha032(
    code: str,
    benchmark: str = 'zz800',
    end_date: str = "2026-01-23",
    lookback: int = 350
) -> float:
    """
    Compute Alpha032 factor value for a stock at a specific date.
    """
    return run_alpha_factor(alpha_032, code, benchmark, end_date, lookback)

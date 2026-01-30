"""
Alpha042 factor implementation.

Formula:
    alpha_042 = (-1 * RANK(STD(HIGH,10))) * CORR(HIGH,VOLUME,10)
"""

import numpy as np
import pandas as pd
from .operators import ts_std, rank, rolling_corr
from .utils import run_alpha_factor


def alpha_042(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha042 factor.

    Formula:
        alpha_042 = (-1 * RANK(STD(HIGH,10))) * CORR(HIGH,VOLUME,10)
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

    # Step 1: Compute STD(HIGH, 10)
    std_high = ts_std(high, 10)

    # Step 2: Compute RANK(STD(HIGH, 10))
    ranked_std = rank(std_high)

    # Step 3: Compute -1 * RANK(STD(HIGH, 10))
    neg_ranked_std = -1 * ranked_std

    # Step 4: Compute CORR(HIGH, VOLUME, 10)
    corr_high_volume = rolling_corr(high, volume, 10)

    # Step 5: Multiply the two components
    alpha_values = neg_ranked_std * corr_high_volume

    return pd.Series(alpha_values, index=index, name='alpha_042')


def alpha042(
    code: str,
    benchmark: str = 'zz800',
    end_date: str = "2026-01-23",
    lookback: int = 350
) -> float:
    """
    Compute Alpha042 factor value for a stock at a specific date.
    """
    return run_alpha_factor(alpha_042, code, benchmark, end_date, lookback)
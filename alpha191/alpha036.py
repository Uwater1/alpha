"""
Alpha036 factor implementation.

Formula:
    alpha_036 = RANK(SUM(CORR(RANK(VOLUME),RANK(VWAP)),6),2)
"""

import numpy as np
import pandas as pd
from .operators import rank, rolling_corr, ts_sum, ts_rank
from .utils import run_alpha_factor


def alpha_036(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha036 factor.

    Formula:
        alpha_036 = RANK(SUM(CORR(RANK(VOLUME),RANK(VWAP)),6),2)
    """
    # Ensure we have required columns
    required_cols = ['volume']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in DataFrame")

    if 'date' in df.columns:
        index = pd.to_datetime(df['date'])
    else:
        index = df.index

    volume = df['volume'].values
    close = df['close'].values

    # Handle VWAP - use typical price if not available
    if 'vwap' in df.columns:
        vwap = df['vwap'].values
    elif 'amount' in df.columns and 'volume' in df.columns:
        vwap = df['amount'].values / df['volume'].values
    else:
        # Approximate VWAP using (open + high + low + close) / 4
        vwap = (df['open'].values + df['high'].values + df['low'].values + close) / 4

    # Step 1: Compute RANK(VOLUME) inside CORR(..., 6)? 
    # Enclosing op is CORR, let's assume window 6 from the SUM/CORR context
    rank_volume = ts_rank(volume, 6)

    # Step 2: Compute RANK(VWAP)
    rank_vwap = ts_rank(vwap, 6)

    # Step 3: Compute CORR(RANK(VOLUME), RANK(VWAP), 6)
    corr_val = rolling_corr(rank_volume, rank_vwap, 6)

    # Step 4: Compute SUM(CORR(...), 6)
    sum_corr = ts_sum(corr_val, 6)

    # Step 5: Compute RANK(SUM(...), 2) - using ts_rank for time-series rank
    alpha_values = ts_rank(sum_corr, 2)

    return pd.Series(alpha_values, index=index, name='alpha_036')


def alpha036(
    code: str,
    benchmark: str = 'zz800',
    end_date: str = "2026-01-23",
    lookback: int = 350
) -> float:
    """
    Compute Alpha036 factor value for a stock at a specific date.
    """
    return run_alpha_factor(alpha_036, code, benchmark, end_date, lookback)

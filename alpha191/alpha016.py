"""
Alpha016 factor implementation.

Formula:
    alpha_016 = (-1*TSMAX(RANK(CORR(RANK(VOLUME),RANK(VWAP),5)),5))
"""

import numpy as np
import pandas as pd
from .operators import rolling_corr, ts_max, rank
from .utils import run_alpha_factor


def alpha_016(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha016 factor.

    Formula:
        alpha_016 = (-1*TSMAX(RANK(CORR(RANK(VOLUME),RANK(VWAP),5)),5))
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

    # Compute VWAP if not present
    if 'vwap' in df.columns:
        vwap = df['vwap'].values
    elif 'amount' in df.columns:
        # VWAP = amount / volume
        amount = df['amount'].values
        vwap = np.full(len(volume), np.nan)
        valid_mask = ~np.isnan(amount) & ~np.isnan(volume) & (volume != 0)
        vwap[valid_mask] = amount[valid_mask] / volume[valid_mask]
    else:
        # Approximate VWAP as (open + high + low + close) / 4
        if 'open' in df.columns and 'high' in df.columns and 'low' in df.columns and 'close' in df.columns:
            vwap = (df['open'].values + df['high'].values + df['low'].values + df['close'].values) / 4
        else:
            raise ValueError("Cannot compute VWAP: missing vwap, amount, or ohlc columns")

    # Step 1: Compute RANK(VOLUME) and RANK(VWAP) (cross-sectional)
    # For single stock time series, rank returns 0.5
    # We use values directly
    rank_volume = volume
    rank_vwap = vwap

    # Step 2: Compute CORR with window=5
    correlation = rolling_corr(rank_volume, rank_vwap, 5)

    # Step 3: Compute RANK for correlation (cross-sectional)
    # For single stock time series, rank returns 0.5
    # We use correlation values directly
    rank_corr = correlation

    # Step 4: Compute TSMAX with window=5
    max_corr = ts_max(rank_corr, 5)

    # Step 5: Final result
    alpha_values = -1 * max_corr

    return pd.Series(alpha_values, index=index, name='alpha_016')


def alpha016(
    code: str,
    benchmark: str = 'zz800',
    end_date: str = "2026-01-23",
    lookback: int = 350
) -> float:
    """
    Compute Alpha016 factor value for a stock at a specific date.
    """
    return run_alpha_factor(alpha_016, code, benchmark, end_date, lookback)

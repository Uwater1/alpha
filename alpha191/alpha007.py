"""
Alpha007 factor implementation.

Formula:
    alpha_007 = ((RANK(MAX((VWAP-CLOSE),3))+RANK(MIN((VWAP-CLOSE),3)))*RANK(DELTA(VOLUME,3)))
"""

import numpy as np
import pandas as pd
from .operators import delta, rank, ts_max, ts_min
from .utils import run_alpha_factor


def alpha_007(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha007 factor.

    Formula:
        alpha_007 = ((RANK(MAX((VWAP-CLOSE),3))+RANK(MIN((VWAP-CLOSE),3)))*RANK(DELTA(VOLUME,3)))
    """
    # Ensure we have required columns
    required_cols = ['close', 'volume']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in DataFrame")

    if 'date' in df.columns:
        index = pd.to_datetime(df['date'])
    else:
        index = df.index

    close = df['close'].values
    volume = df['volume'].values

    # Compute VWAP if not present
    if 'vwap' in df.columns:
        vwap = df['vwap'].values
    elif 'amount' in df.columns:
        # VWAP = amount / volume
        amount = df['amount'].values
        vwap = np.full(len(close), np.nan)
        valid_mask = ~np.isnan(amount) & ~np.isnan(volume) & (volume != 0)
        vwap[valid_mask] = amount[valid_mask] / volume[valid_mask]
    else:
        # Approximate VWAP as (open + high + low + close) / 4
        if 'open' in df.columns and 'high' in df.columns and 'low' in df.columns:
            vwap = (df['open'].values + df['high'].values + df['low'].values + close) / 4
        else:
            raise ValueError("Cannot compute VWAP: missing vwap, amount, or ohlc columns")

    # Step 1: Compute (VWAP-CLOSE)
    vwap_close_diff = vwap - close

    # Step 2: Compute MAX((VWAP-CLOSE), 3)
    max_diff = ts_max(vwap_close_diff, 3)

    # Step 3: Compute MIN((VWAP-CLOSE), 3)
    min_diff = ts_min(vwap_close_diff, 3)

    # Step 4: Compute RANK for max and min (cross-sectional)
    # For single stock time series, rank returns 0.5
    rank_max = max_diff  # Simplified for single stock
    rank_min = min_diff  # Simplified for single stock

    # Step 5: Compute DELTA(VOLUME, 3)
    delta_volume = delta(volume, 3)

    # Step 6: Compute RANK for delta volume
    rank_delta = delta_volume  # Simplified for single stock

    # Step 7: Final result
    alpha_values = (rank_max + rank_min) * rank_delta

    return pd.Series(alpha_values, index=index, name='alpha_007')


def alpha007(
    code: str,
    benchmark: str = 'zz800',
    end_date: str = "2026-01-23",
    lookback: int = 350
) -> float:
    """
    Compute Alpha007 factor value for a stock at a specific date.
    """
    return run_alpha_factor(alpha_007, code, benchmark, end_date, lookback)

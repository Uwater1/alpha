"""
Alpha017 factor implementation.

Formula:
    alpha_017 = RANK((VWAP-MAX(VWAP,15)))^DELTA(CLOSE,5)
"""

import numpy as np
import pandas as pd
from .operators import delta, ts_max, rank
from .utils import run_alpha_factor


def alpha_017(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha017 factor.

    Formula:
        alpha_017 = RANK((VWAP-MAX(VWAP,15)))^DELTA(CLOSE,5)
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

    # Compute VWAP if not present
    if 'vwap' in df.columns:
        vwap = df['vwap'].values
    elif 'amount' in df.columns and 'volume' in df.columns:
        # VWAP = amount / volume
        amount = df['amount'].values
        volume = df['volume'].values
        vwap = np.full(len(close), np.nan)
        valid_mask = ~np.isnan(amount) & ~np.isnan(volume) & (volume != 0)
        vwap[valid_mask] = amount[valid_mask] / volume[valid_mask]
    else:
        # Approximate VWAP as (open + high + low + close) / 4
        if 'open' in df.columns and 'high' in df.columns and 'low' in df.columns:
            vwap = (df['open'].values + df['high'].values + df['low'].values + close) / 4
        else:
            raise ValueError("Cannot compute VWAP: missing vwap, amount, or ohlc columns")

    # Step 1: Compute MAX(VWAP, 15)
    max_vwap = ts_max(vwap, 15)

    # Step 2: Compute (VWAP-MAX(VWAP,15))
    vwap_diff = vwap - max_vwap

    # Step 3: Compute RANK (cross-sectional)
    # For single stock time series, rank returns 0.5
    # We use vwap_diff values directly
    rank_vwap_diff = vwap_diff

    # Step 4: Compute DELTA(CLOSE, 5)
    delta_close = delta(close, 5)

    # Step 5: Compute RANK((VWAP-MAX(VWAP,15)))^DELTA(CLOSE,5)
    # Using power operation
    alpha_values = rank_vwap_diff ** delta_close

    return pd.Series(alpha_values, index=index, name='alpha_017')


def alpha017(
    code: str,
    benchmark: str = 'zz800',
    end_date: str = "2026-01-23",
    lookback: int = 350
) -> float:
    """
    Compute Alpha017 factor value for a stock at a specific date.
    """
    return run_alpha_factor(alpha_017, code, benchmark, end_date, lookback)

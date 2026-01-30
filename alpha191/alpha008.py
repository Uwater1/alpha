"""
Alpha008 factor implementation.

Formula:
    alpha_008 = RANK(DELTA(((((HIGH+LOW)/2)*0.2)+(VWAP*0.8)),4))*-1
"""

import numpy as np
import pandas as pd
from .operators import delta, rank
from .utils import run_alpha_factor


def alpha_008(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha008 factor.

    Formula:
        alpha_008 = RANK(DELTA(((((HIGH+LOW)/2)*0.2)+(VWAP*0.8)),4))*-1
    """
    # Ensure we have required columns
    required_cols = ['high', 'low']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in DataFrame")

    if 'date' in df.columns:
        index = pd.to_datetime(df['date'])
    else:
        index = df.index

    high = df['high'].values
    low = df['low'].values

    # Compute VWAP if not present
    if 'vwap' in df.columns:
        vwap = df['vwap'].values
    elif 'amount' in df.columns and 'volume' in df.columns:
        # VWAP = amount / volume
        amount = df['amount'].values
        volume = df['volume'].values
        vwap = np.full(len(high), np.nan)
        valid_mask = ~np.isnan(amount) & ~np.isnan(volume) & (volume != 0)
        vwap[valid_mask] = amount[valid_mask] / volume[valid_mask]
    else:
        # Approximate VWAP as (open + high + low + close) / 4
        if 'open' in df.columns and 'close' in df.columns:
            vwap = (df['open'].values + high + low + df['close'].values) / 4
        else:
            raise ValueError("Cannot compute VWAP: missing vwap, amount, or ohlc columns")

    # Step 1: Compute ((HIGH+LOW)/2)*0.2
    hl_avg = (high + low) / 2
    hl_weighted = hl_avg * 0.2

    # Step 2: Compute VWAP*0.8
    vwap_weighted = vwap * 0.8

    # Step 3: Compute combined value
    combined = hl_weighted + vwap_weighted

    # Step 4: Compute DELTA with n=4
    delta_combined = delta(combined, 4)

    # Step 5: Compute RANK (cross-sectional)
    # For single stock time series, rank returns 0.5
    # We use the delta values directly
    alpha_values = delta_combined * -1

    return pd.Series(alpha_values, index=index, name='alpha_008')


def alpha008(
    code: str,
    benchmark: str = 'zz800',
    end_date: str = "2026-01-23",
    lookback: int = 350
) -> float:
    """
    Compute Alpha008 factor value for a stock at a specific date.
    """
    return run_alpha_factor(alpha_008, code, benchmark, end_date, lookback)

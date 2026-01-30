"""
Alpha013 factor implementation.

Formula:
    alpha_013 = (((HIGH*LOW)^0.5)-VWAP)
"""

import numpy as np
import pandas as pd
from .utils import run_alpha_factor


def alpha_013(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha013 factor.

    Formula:
        alpha_013 = (((HIGH*LOW)^0.5)-VWAP)
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

    # Step 1: Compute (HIGH*LOW)^0.5 (geometric mean)
    hl_product = high * low
    hl_sqrt = np.sqrt(hl_product)

    # Step 2: Compute (((HIGH*LOW)^0.5)-VWAP)
    alpha_values = hl_sqrt - vwap

    return pd.Series(alpha_values, index=index, name='alpha_013')


def alpha013(
    code: str,
    benchmark: str = 'zz800',
    end_date: str = "2026-01-23",
    lookback: int = 350
) -> float:
    """
    Compute Alpha013 factor value for a stock at a specific date.
    """
    return run_alpha_factor(alpha_013, code, benchmark, end_date, lookback)

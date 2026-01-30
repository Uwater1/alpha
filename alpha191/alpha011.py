"""
Alpha011 factor implementation.

Formula:
    alpha_011 = (SUM(((CLOSE-LOW)-(HIGH-CLOSE))./(HIGH-LOW).*VOLUME,6)
"""

import numpy as np
import pandas as pd
from .operators import ts_sum
from .utils import run_alpha_factor


def alpha_011(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha011 factor.

    Formula:
        alpha_011 = (SUM(((CLOSE-LOW)-(HIGH-CLOSE))./(HIGH-LOW).*VOLUME,6)
    """
    # Ensure we have required columns
    required_cols = ['close', 'high', 'low', 'volume']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in DataFrame")

    if 'date' in df.columns:
        index = pd.to_datetime(df['date'])
    else:
        index = df.index

    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    volume = df['volume'].values

    # Step 1: Compute (CLOSE-LOW)
    close_low = close - low

    # Step 2: Compute (HIGH-CLOSE)
    high_close = high - close

    # Step 3: Compute ((CLOSE-LOW)-(HIGH-CLOSE))
    diff = close_low - high_close

    # Step 4: Compute (HIGH-LOW)
    hl_diff = high - low

    # Step 5: Compute ((CLOSE-LOW)-(HIGH-CLOSE))./(HIGH-LOW)
    # Handle division by zero
    ratio = np.full(len(close), np.nan)
    valid_mask = ~np.isnan(diff) & ~np.isnan(hl_diff) & (hl_diff != 0)
    ratio[valid_mask] = diff[valid_mask] / hl_diff[valid_mask]

    # Step 6: Compute ((CLOSE-LOW)-(HIGH-CLOSE))./(HIGH-LOW).*VOLUME
    weighted = ratio * volume

    # Step 7: Compute SUM with window=6
    alpha_values = ts_sum(weighted, 6)

    return pd.Series(alpha_values, index=index, name='alpha_011')


def alpha011(
    code: str,
    benchmark: str = 'zz800',
    end_date: str = "2026-01-23",
    lookback: int = 350
) -> float:
    """
    Compute Alpha011 factor value for a stock at a specific date.
    """
    return run_alpha_factor(alpha_011, code, benchmark, end_date, lookback)

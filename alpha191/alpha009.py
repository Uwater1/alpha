"""
Alpha009 factor implementation.

Formula:
    alpha_009 = SMA(((HIGH+LOW)/2-(DELAY(HIGH,1)+DELAY(LOW,1))/2)*(HIGH-LOW)/VOLUME,7,2)
"""

import numpy as np
import pandas as pd
from .operators import delay, sma
from .utils import run_alpha_factor


def alpha_009(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha009 factor.

    Formula:
        alpha_009 = SMA(((HIGH+LOW)/2-(DELAY(HIGH,1)+DELAY(LOW,1))/2)*(HIGH-LOW)/VOLUME,7,2)
    """
    # Ensure we have required columns
    required_cols = ['high', 'low', 'volume']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in DataFrame")

    if 'date' in df.columns:
        index = pd.to_datetime(df['date'])
    else:
        index = df.index

    high = df['high'].values
    low = df['low'].values
    volume = df['volume'].values

    # Step 1: Compute (HIGH+LOW)/2
    hl_avg = (high + low) / 2

    # Step 2: Compute DELAY(HIGH,1) and DELAY(LOW,1)
    delay_high = delay(high, 1)
    delay_low = delay(low, 1)

    # Step 3: Compute (DELAY(HIGH,1)+DELAY(LOW,1))/2
    delay_hl_avg = (delay_high + delay_low) / 2

    # Step 4: Compute (HIGH-LOW)
    hl_diff = high - low

    # Step 5: Compute ((HIGH+LOW)/2-(DELAY(HIGH,1)+DELAY(LOW,1))/2)*(HIGH-LOW)/VOLUME
    # Handle division by zero for volume
    volume_safe = volume.astype(float).copy()
    volume_safe[volume_safe == 0] = np.nan

    numerator = (hl_avg - delay_hl_avg) * hl_diff
    ratio = np.full(len(volume), np.nan)
    valid_mask = ~np.isnan(numerator) & ~np.isnan(volume_safe) & (volume_safe != 0)
    ratio[valid_mask] = numerator[valid_mask] / volume_safe[valid_mask]

    # Step 6: Compute SMA with n=7, m=2
    alpha_values = sma(ratio, 7, 2)

    return pd.Series(alpha_values, index=index, name='alpha_009')


def alpha009(
    code: str,
    benchmark: str = 'zz800',
    end_date: str = "2026-01-23",
    lookback: int = 350
) -> float:
    """
    Compute Alpha009 factor value for a stock at a specific date.
    """
    return run_alpha_factor(alpha_009, code, benchmark, end_date, lookback)

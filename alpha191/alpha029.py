"""
Alpha029 factor implementation.

Formula:
    alpha_029 = (CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*VOLUME
"""

import numpy as np
import pandas as pd
from .operators import delay
from .utils import run_alpha_factor


def alpha_029(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha029 factor.

    Formula:
        alpha_029 = (CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*VOLUME
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

    # Step 1: Compute DELAY(CLOSE, 6)
    delay_close = delay(close, 6)

    # Step 2: Compute (CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)
    ratio = np.full(len(close), np.nan)
    valid_mask = ~np.isnan(delay_close) & (delay_close != 0)
    ratio[valid_mask] = (close[valid_mask] - delay_close[valid_mask]) / delay_close[valid_mask]

    # Step 3: Multiply by VOLUME
    alpha_values = ratio * volume

    return pd.Series(alpha_values, index=index, name='alpha_029')


def alpha029(
    code: str,
    benchmark: str = 'zz800',
    end_date: str = "2026-01-23",
    lookback: int = 350
) -> float:
    """
    Compute Alpha029 factor value for a stock at a specific date.
    """
    return run_alpha_factor(alpha_029, code, benchmark, end_date, lookback)

"""
Alpha019 factor implementation.

Formula:
    alpha_019 = (CLOSE<DELAY(CLOSE,5)?(CLOSE-DELAY(CLOSE,5))/DELAY(CLOSE,5):(CLOSE=DELAY(CLOSE,5)?0:(CLOSE-DELAY(CLOSE,5))/CLOSE))
"""

import numpy as np
import pandas as pd
from .operators import delay
from .utils import run_alpha_factor


def alpha_019(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha019 factor.

    Formula:
        alpha_019 = (CLOSE<DELAY(CLOSE,5)?(CLOSE-DELAY(CLOSE,5))/DELAY(CLOSE,5):(CLOSE=DELAY(CLOSE,5)?0:(CLOSE-DELAY(CLOSE,5))/CLOSE))
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

    # Step 1: Compute DELAY(CLOSE, 5)
    delay_close = delay(close, 5)

    # Step 2: Compute (CLOSE-DELAY(CLOSE,5))
    diff = close - delay_close

    # Step 3: Apply conditional logic
    # If CLOSE < DELAY(CLOSE,5): (CLOSE-DELAY(CLOSE,5))/DELAY(CLOSE,5)
    # If CLOSE = DELAY(CLOSE,5): 0
    # Otherwise: (CLOSE-DELAY(CLOSE,5))/CLOSE

    # First condition: CLOSE < DELAY(CLOSE,5)
    cond1 = close < delay_close

    # Second condition: CLOSE = DELAY(CLOSE,5)
    cond2 = close == delay_close

    # Compute result for each condition
    # For cond1: diff / delay_close
    result1 = np.full(len(close), np.nan)
    valid_mask1 = ~np.isnan(delay_close) & (delay_close != 0)
    result1[valid_mask1] = diff[valid_mask1] / delay_close[valid_mask1]

    # For cond2: 0
    result2 = np.zeros(len(close))

    # For else: diff / close
    result3 = np.full(len(close), np.nan)
    valid_mask3 = ~np.isnan(close) & (close != 0)
    result3[valid_mask3] = diff[valid_mask3] / close[valid_mask3]

    # Apply nested conditional: A?B:C
    # First level: if cond1 then result1 else check cond2
    alpha_values = np.where(
        cond1,
        result1,
        np.where(
            cond2,
            result2,
            result3
        )
    )

    return pd.Series(alpha_values, index=index, name='alpha_019')


def alpha019(
    code: str,
    benchmark: str = 'zz800',
    end_date: str = "2026-01-23",
    lookback: int = 350
) -> float:
    """
    Compute Alpha019 factor value for a stock at a specific date.
    """
    return run_alpha_factor(alpha_019, code, benchmark, end_date, lookback)

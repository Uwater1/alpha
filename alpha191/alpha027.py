"""
Alpha027 factor implementation.

Formula:
    alpha_027 = WMA((CLOSE-DELAY(CLOSE,3))/DELAY(CLOSE,3)*100+(CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*100,12)
"""

import numpy as np
import pandas as pd
from .operators import delay, wma
from .utils import run_alpha_factor


def alpha_027(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha027 factor.

    Formula:
        alpha_027 = WMA((CLOSE-DELAY(CLOSE,3))/DELAY(CLOSE,3)*100+(CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*100,12)
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

    # Step 1: Compute DELAY(CLOSE, 3)
    delay_close_3 = delay(close, 3)

    # Step 2: Compute DELAY(CLOSE, 6)
    delay_close_6 = delay(close, 6)

    # Step 3: Compute (CLOSE-DELAY(CLOSE,3))/DELAY(CLOSE,3)*100
    ratio_3 = np.full(len(close), np.nan)
    valid_mask_3 = ~np.isnan(delay_close_3) & (delay_close_3 != 0)
    ratio_3[valid_mask_3] = (close[valid_mask_3] - delay_close_3[valid_mask_3]) / delay_close_3[valid_mask_3] * 100

    # Step 4: Compute (CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*100
    ratio_6 = np.full(len(close), np.nan)
    valid_mask_6 = ~np.isnan(delay_close_6) & (delay_close_6 != 0)
    ratio_6[valid_mask_6] = (close[valid_mask_6] - delay_close_6[valid_mask_6]) / delay_close_6[valid_mask_6] * 100

    # Step 5: Compute sum of the two ratios
    combined = ratio_3 + ratio_6

    # Step 6: Compute WMA(..., 12)
    alpha_values = wma(combined, 12)

    return pd.Series(alpha_values, index=index, name='alpha_027')


def alpha027(
    code: str,
    benchmark: str = 'zz800',
    end_date: str = "2026-01-23",
    lookback: int = 350
) -> float:
    """
    Compute Alpha027 factor value for a stock at a specific date.
    """
    return run_alpha_factor(alpha_027, code, benchmark, end_date, lookback)

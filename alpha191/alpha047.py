"""
Alpha047 factor implementation.

Formula:
    alpha_047 = SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))*100,9,1)
"""

import numpy as np
import pandas as pd
from .operators import ts_max, ts_min, sma
from .utils import run_alpha_factor


def alpha_047(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha047 factor.

    Formula:
        alpha_047 = SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))*100,9,1)
    """
    # Ensure we have required columns
    required_cols = ['high', 'close', 'low']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in DataFrame")

    if 'date' in df.columns:
        index = pd.to_datetime(df['date'])
    else:
        index = df.index

    high = df['high'].values
    close = df['close'].values
    low = df['low'].values

    # Step 1: Compute TSMAX(HIGH, 6)
    max_high_6 = ts_max(high, 6)

    # Step 2: Compute TSMIN(LOW, 6)
    min_low_6 = ts_min(low, 6)

    # Step 3: Compute (TSMAX(HIGH,6) - CLOSE)
    numerator = max_high_6 - close

    # Step 4: Compute (TSMAX(HIGH,6) - TSMIN(LOW,6))
    denominator = max_high_6 - min_low_6

    # Step 5: Compute (TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))
    # Protect against division by zero
    denom = denominator.copy()
    denom[denom == 0] = np.nan
    ratio = numerator / denom

    # Step 6: Multiply by 100
    ratio_100 = ratio * 100

    # Step 7: Compute SMA(..., 9, 1)
    alpha_values = sma(ratio_100, 9, 1)

    return pd.Series(alpha_values, index=index, name='alpha_047')


def alpha047(
    code: str,
    benchmark: str = 'zz800',
    end_date: str = "2026-01-23",
    lookback: int = 350
) -> float:
    """
    Compute Alpha047 factor value for a stock at a specific date.
    """
    return run_alpha_factor(alpha_047, code, benchmark, end_date, lookback)
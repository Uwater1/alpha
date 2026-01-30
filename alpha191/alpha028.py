"""
Alpha028 factor implementation.

Formula:
    alpha_028 = 3*SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1)-2*SMA(SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1),3,1)
"""

import numpy as np
import pandas as pd
from .operators import ts_min, ts_max, sma
from .utils import run_alpha_factor


def alpha_028(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha028 factor.

    Formula:
        alpha_028 = 3*SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1)-2*SMA(SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1),3,1)
    """
    # Ensure we have required columns
    required_cols = ['close', 'high', 'low']
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

    # Step 1: Compute TSMIN(LOW, 9)
    ts_min_low = ts_min(low, 9)

    # Step 2: Compute TSMAX(HIGH, 9)
    ts_max_high = ts_max(high, 9)

    # Step 3: Compute (CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100
    range_val = ts_max_high - ts_min_low
    ratio = np.full(len(close), np.nan)
    valid_mask = ~np.isnan(range_val) & (range_val != 0) & ~np.isnan(close) & ~np.isnan(ts_min_low)
    ratio[valid_mask] = (close[valid_mask] - ts_min_low[valid_mask]) / range_val[valid_mask] * 100

    # Step 4: Compute SMA(..., 3, 1)
    sma_1 = sma(ratio, 3, 1)

    # Step 5: Compute SMA(SMA(..., 3, 1), 3, 1)
    sma_2 = sma(sma_1, 3, 1)

    # Step 6: Compute 3*SMA(...) - 2*SMA(SMA(...))
    alpha_values = 3 * sma_1 - 2 * sma_2

    return pd.Series(alpha_values, index=index, name='alpha_028')


def alpha028(
    code: str,
    benchmark: str = 'zz800',
    end_date: str = "2026-01-23",
    lookback: int = 350
) -> float:
    """
    Compute Alpha028 factor value for a stock at a specific date.
    """
    return run_alpha_factor(alpha_028, code, benchmark, end_date, lookback)

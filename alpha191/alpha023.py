"""
Alpha023 factor implementation.

Formula:
    alpha_023 = SMA((CLOSE>DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1)/(SMA((CLOSE>DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1)+SMA((CLOSE<=DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1))*100
"""

import numpy as np
import pandas as pd
from .operators import delay, ts_std, sma
from .utils import run_alpha_factor


def alpha_023(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha023 factor.

    Formula:
        alpha_023 = SMA((CLOSE>DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1)/(SMA((CLOSE>DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1)+SMA((CLOSE<=DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1))*100
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

    # Step 1: Compute DELAY(CLOSE, 1)
    delay_close = delay(close, 1)

    # Step 2: Compute STD(CLOSE, 20)
    std_close = ts_std(close, 20)

    # Step 3: Compute condition (CLOSE > DELAY(CLOSE,1))
    cond_up = close > delay_close
    cond_down = close <= delay_close

    # Step 4: Compute (CLOSE>DELAY(CLOSE,1)?STD(CLOSE,20):0)
    up_value = np.where(cond_up, std_close, 0)

    # Step 5: Compute (CLOSE<=DELAY(CLOSE,1)?STD(CLOSE,20):0)
    down_value = np.where(cond_down, std_close, 0)

    # Step 6: Compute SMA for both
    sma_up = sma(up_value, 20, 1)
    sma_down = sma(down_value, 20, 1)

    # Step 7: Compute ratio
    alpha_values = np.full(len(close), np.nan)
    denom = sma_up + sma_down
    valid_mask = ~np.isnan(denom) & (denom != 0)
    alpha_values[valid_mask] = sma_up[valid_mask] / denom[valid_mask] * 100

    return pd.Series(alpha_values, index=index, name='alpha_023')


def alpha023(
    code: str,
    benchmark: str = 'zz800',
    end_date: str = "2026-01-23",
    lookback: int = 350
) -> float:
    """
    Compute Alpha023 factor value for a stock at a specific date.
    """
    return run_alpha_factor(alpha_023, code, benchmark, end_date, lookback)

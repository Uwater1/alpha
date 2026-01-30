"""
Alpha033 factor implementation.

Formula:
    alpha_033 = ((-1 * TSMIN(LOW, 5) + DELAY(TSMIN(LOW, 5), 5)) * 
                 RANK((SUM(RET, 240) - SUM(RET, 20)) / 220)) * 
                TSRANK(VOLUME, 5)
"""

import numpy as np
import pandas as pd
from .operators import ts_min, delay, ts_sum, compute_ret, rank, ts_rank
from .utils import run_alpha_factor


def alpha_033(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha033 factor.

    Formula:
        alpha_033 = ((-1 * TSMIN(LOW, 5) + DELAY(TSMIN(LOW, 5), 5)) * 
                     RANK((SUM(RET, 240) - SUM(RET, 20)) / 220)) * 
                    TSRANK(VOLUME, 5)
    """
    # Ensure we have required columns
    required_cols = ['close', 'low', 'volume']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in DataFrame")

    if 'date' in df.columns:
        index = pd.to_datetime(df['date'])
    else:
        index = df.index

    close = df['close'].values
    low = df['low'].values
    volume = df['volume'].values

    # Step 1: Compute TSMIN(LOW, 5) and its DELAY(..., 5)
    tsmin_low = ts_min(low, 5)
    delayed_tsmin = delay(tsmin_low, 5)

    # Step 2: Compute RET and its SUM(RET, 240) and SUM(RET, 20)
    ret = compute_ret(close)
    sum_ret_240 = ts_sum(ret, 240)
    sum_ret_20 = ts_sum(ret, 20)

    # Step 3: Compute RANK((SUM(RET, 240) - SUM(RET, 20)) / 220)
    target_value = (sum_ret_240 - sum_ret_20) / 220
    ranked_value = rank(target_value)

    # Step 4: Compute TSRANK(VOLUME, 5)
    tsrank_vol = ts_rank(volume, 5)

    # Step 5: Final formula
    alpha_values = ((-1 * tsmin_low + delayed_tsmin) * ranked_value) * tsrank_vol

    return pd.Series(alpha_values, index=index, name='alpha_033')


def alpha033(
    code: str,
    benchmark: str = 'zz800',
    end_date: str = "2026-01-23",
    lookback: int = 350
) -> float:
    """
    Compute Alpha033 factor value for a stock at a specific date.
    """
    return run_alpha_factor(alpha_033, code, benchmark, end_date, lookback)

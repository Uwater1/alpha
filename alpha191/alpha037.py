"""
Alpha037 factor implementation.

Formula:
    alpha_037 = (-1*RANK(((SUM(OPEN,5)*SUM(RET,5))-DELAY((SUM(OPEN,5)*SUM(RET,5)),10))))
"""

import numpy as np
import pandas as pd
from .operators import ts_sum, delay, compute_ret, rank
from .utils import run_alpha_factor


def alpha_037(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha037 factor.

    Formula:
        alpha_037 = (-1*RANK(((SUM(OPEN,5)*SUM(RET,5))-DELAY((SUM(OPEN,5)*SUM(RET,5)),10))))
    """
    # Ensure we have required columns
    required_cols = ['open', 'close']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in DataFrame")

    if 'date' in df.columns:
        index = pd.to_datetime(df['date'])
    else:
        index = df.index

    open_price = df['open'].values
    close = df['close'].values

    # Step 1: Compute SUM(OPEN, 5)
    sum_open = ts_sum(open_price, 5)

    # Step 2: Compute RET
    ret = compute_ret(close)

    # Step 3: Compute SUM(RET, 5)
    sum_ret = ts_sum(ret, 5)

    # Step 4: Compute SUM(OPEN,5)*SUM(RET,5)
    product = sum_open * sum_ret

    # Step 5: Compute DELAY((SUM(OPEN,5)*SUM(RET,5)), 10)
    delay_product = delay(product, 10)

    # Step 6: Compute (SUM(OPEN,5)*SUM(RET,5))-DELAY(...,10)
    diff = product - delay_product

    # Step 7: Compute RANK(...)
    rank_val = rank(diff)

    # Step 8: Compute final formula
    alpha_values = -1 * rank_val

    return pd.Series(alpha_values, index=index, name='alpha_037')


def alpha037(
    code: str,
    benchmark: str = 'zz800',
    end_date: str = "2026-01-23",
    lookback: int = 350
) -> float:
    """
    Compute Alpha037 factor value for a stock at a specific date.
    """
    return run_alpha_factor(alpha_037, code, benchmark, end_date, lookback)

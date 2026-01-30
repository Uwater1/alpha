"""
Alpha038 factor implementation.

Formula:
    alpha_038 = (((SUM(HIGH,20)/20)<HIGH)?(-1*DELTA(HIGH,2)):0)
"""

import numpy as np
import pandas as pd
from .operators import ts_sum, delta
from .utils import run_alpha_factor


def alpha_038(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha038 factor.

    Formula:
        alpha_038 = (((SUM(HIGH,20)/20)<HIGH)?(-1*DELTA(HIGH,2)):0)
    """
    # Ensure we have required columns
    required_cols = ['high']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in DataFrame")

    if 'date' in df.columns:
        index = pd.to_datetime(df['date'])
    else:
        index = df.index

    high = df['high'].values

    # Step 1: Compute SUM(HIGH, 20)/20
    sum_high = ts_sum(high, 20)
    mean_high = sum_high / 20

    # Step 2: Compute condition (SUM(HIGH,20)/20 < HIGH)
    condition = mean_high < high

    # Step 3: Compute DELTA(HIGH, 2)
    delta_high = delta(high, 2)

    # Step 4: Compute -1*DELTA(HIGH,2)
    neg_delta = -1 * delta_high

    # Step 5: Apply condition
    alpha_values = np.where(condition, neg_delta, 0)

    return pd.Series(alpha_values, index=index, name='alpha_038')


def alpha038(
    code: str,
    benchmark: str = 'zz800',
    end_date: str = "2026-01-23",
    lookback: int = 350
) -> float:
    """
    Compute Alpha038 factor value for a stock at a specific date.
    """
    return run_alpha_factor(alpha_038, code, benchmark, end_date, lookback)

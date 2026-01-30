"""
Alpha006 factor implementation.

Formula:
    alpha_006 = (RANK(SIGN(DELTA((((OPEN*0.85)+(HIGH*0.15))),4)))*-1)
"""

import numpy as np
import pandas as pd
from .operators import delta, sign, rank
from .utils import run_alpha_factor


def alpha_006(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha006 factor.

    Formula:
        alpha_006 = (RANK(SIGN(DELTA((((OPEN*0.85)+(HIGH*0.15))),4)))*-1)
    """
    # Ensure we have required columns
    required_cols = ['open', 'high']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in DataFrame")

    if 'date' in df.columns:
        index = pd.to_datetime(df['date'])
    else:
        index = df.index

    open_price = df['open'].values
    high = df['high'].values

    # Step 1: Compute (OPEN*0.85)+(HIGH*0.15)
    weighted_price = (open_price * 0.85) + (high * 0.15)

    # Step 2: Compute DELTA with n=4
    delta_weighted = delta(weighted_price, 4)

    # Step 3: Compute SIGN
    sign_delta = sign(delta_weighted)

    # Step 4: Compute RANK (cross-sectional)
    # Since this is a single stock time series, rank returns 0.5 for single value
    # For time series, we use the values directly
    alpha_values = sign_delta * -1

    return pd.Series(alpha_values, index=index, name='alpha_006')


def alpha006(
    code: str,
    benchmark: str = 'zz800',
    end_date: str = "2026-01-23",
    lookback: int = 350
) -> float:
    """
    Compute Alpha006 factor value for a stock at a specific date.
    """
    return run_alpha_factor(alpha_006, code, benchmark, end_date, lookback)

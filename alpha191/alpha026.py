"""
Alpha026 factor implementation.

Formula:
    alpha_026 = ((((SUM(CLOSE,7)/7)-CLOSE))+((CORR(VWAP,DELAY(CLOSE,5),230))))
"""

import numpy as np
import pandas as pd
from .operators import ts_sum, delay, rolling_corr
from .utils import run_alpha_factor


def alpha_026(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha026 factor.

    Formula:
        alpha_026 = ((((SUM(CLOSE,7)/7)-CLOSE))+((CORR(VWAP,DELAY(CLOSE,5),230))))
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

    # Handle VWAP - use typical price if not available
    if 'vwap' in df.columns:
        vwap = df['vwap'].values
    elif 'amount' in df.columns and 'volume' in df.columns:
        vwap = df['amount'].values / df['volume'].values
    else:
        # Approximate VWAP using (open + high + low + close) / 4
        vwap = (df['open'].values + df['high'].values + df['low'].values + close) / 4

    # Step 1: Compute SUM(CLOSE, 7)/7
    sum_close = ts_sum(close, 7)
    mean_close_7 = sum_close / 7

    # Step 2: Compute (SUM(CLOSE,7)/7)-CLOSE
    diff = mean_close_7 - close

    # Step 3: Compute DELAY(CLOSE, 5)
    delay_close = delay(close, 5)

    # Step 4: Compute CORR(VWAP, DELAY(CLOSE,5), 230)
    corr_val = rolling_corr(vwap, delay_close, 230)

    # Step 5: Compute final formula
    alpha_values = diff + corr_val

    return pd.Series(alpha_values, index=index, name='alpha_026')


def alpha026(
    code: str,
    benchmark: str = 'zz800',
    end_date: str = "2026-01-23",
    lookback: int = 350
) -> float:
    """
    Compute Alpha026 factor value for a stock at a specific date.
    """
    return run_alpha_factor(alpha_026, code, benchmark, end_date, lookback)

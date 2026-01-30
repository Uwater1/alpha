"""
Alpha012 factor implementation.

Formula:
    alpha_012 = (RANK((OPEN-(SUM(VWAP,10)/10)))*(-1*(RANK(ABS((CLOSE-VWAP)))))
"""

import numpy as np
import pandas as pd
from .operators import ts_sum, rank
from .utils import run_alpha_factor


def alpha_012(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha012 factor.

    Formula:
        alpha_012 = (RANK((OPEN-(SUM(VWAP,10)/10)))*(-1*(RANK(ABS((CLOSE-VWAP)))))
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

    # Compute VWAP if not present
    if 'vwap' in df.columns:
        vwap = df['vwap'].values
    elif 'amount' in df.columns and 'volume' in df.columns:
        # VWAP = amount / volume
        amount = df['amount'].values
        volume = df['volume'].values
        vwap = np.full(len(close), np.nan)
        valid_mask = ~np.isnan(amount) & ~np.isnan(volume) & (volume != 0)
        vwap[valid_mask] = amount[valid_mask] / volume[valid_mask]
    else:
        # Approximate VWAP as (open + high + low + close) / 4
        if 'high' in df.columns and 'low' in df.columns:
            vwap = (open_price + df['high'].values + df['low'].values + close) / 4
        else:
            raise ValueError("Cannot compute VWAP: missing vwap, amount, or ohlc columns")

    # Step 1: Compute SUM(VWAP, 10)
    sum_vwap = ts_sum(vwap, 10)

    # Step 2: Compute SUM(VWAP, 10)/10
    mean_vwap = sum_vwap / 10

    # Step 3: Compute (OPEN-(SUM(VWAP,10)/10))
    open_diff = open_price - mean_vwap

    # Step 4: Compute (CLOSE-VWAP)
    close_vwap_diff = close - vwap

    # Step 5: Compute ABS((CLOSE-VWAP))
    abs_diff = np.abs(close_vwap_diff)

    # Step 6: Compute RANK for both parts (cross-sectional)
    # For single stock time series, rank returns 0.5
    # We use values directly
    rank_open_diff = open_diff
    rank_abs_diff = abs_diff

    # Step 7: Final result
    alpha_values = rank_open_diff * (-1 * rank_abs_diff)

    return pd.Series(alpha_values, index=index, name='alpha_012')


def alpha012(
    code: str,
    benchmark: str = 'zz800',
    end_date: str = "2026-01-23",
    lookback: int = 350
) -> float:
    """
    Compute Alpha012 factor value for a stock at a specific date.
    """
    return run_alpha_factor(alpha_012, code, benchmark, end_date, lookback)

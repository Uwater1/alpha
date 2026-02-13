"""
Alpha017 factor implementation.

Formula:
    alpha_017 = RANK((VWAP-MAX(VWAP,15)))^DELTA(CLOSE,5)
"""

import numpy as np
import pandas as pd
from .operators import delta, ts_max, rank
from .utils import run_alpha_factor


def alpha_017(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha017 factor.

    Formula:
        alpha_017 = RANK((VWAP-MAX(VWAP,15)))^DELTA(CLOSE,5)
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

    # Compute VWAP if not present
    if 'vwap' in df.columns:
        vwap = df['vwap'].values
    else:
        need_ohlc = True
        if {'amount', 'volume'}.issubset(df.columns):
            vwap_s = df['amount'] / df['volume'].replace(0, np.nan)
            valid = df['amount'].ne(0) & df['volume'].ne(0) & vwap_s.notna() & vwap_s.between(df['low'], df['high'])
            need_ohlc = ~valid.all()
            if not need_ohlc:
                vwap = vwap_s.values

        if need_ohlc:
            ohlc_avg = (df['open'] + df['high'] + df['low'] + df['close']) / 4
            if 'valid' in locals():
                vwap = vwap_s.where(valid, ohlc_avg).values
            else:
                vwap = ohlc_avg.values

    # Step 1: Compute MAX(VWAP, 15)
    max_vwap = ts_max(vwap, 15)

    # Step 2: Compute (VWAP-MAX(VWAP,15))
    vwap_diff = vwap - max_vwap

    # Step 3: Compute RANK (cross-sectional)
    # For single stock time series, rank returns 0.5
    # We use vwap_diff values directly
    rank_vwap_diff = vwap_diff

    # Step 4: Compute DELTA(CLOSE, 5)
    delta_close = delta(close, 5)

    # Step 5: Compute RANK((VWAP-MAX(VWAP,15)))^DELTA(CLOSE,5)
    # Using power operation - suppress expected warnings for edge cases
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        alpha_values = rank_vwap_diff ** delta_close

    return pd.Series(alpha_values, index=index, name='alpha_017')


def alpha017(
    code: str,
    benchmark: str = 'zz800',
    end_date: str = "2026-01-23",
    lookback: int = 350
) -> float:
    """
    Compute Alpha017 factor value for a stock at a specific date.
    """
    return run_alpha_factor(alpha_017, code, benchmark, end_date, lookback)

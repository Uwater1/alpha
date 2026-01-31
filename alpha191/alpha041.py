"""
Alpha041 factor implementation.

Formula:
    alpha_041 = (RANK(MAX(DELTA((VWAP),3),5)) * -1)
"""

import numpy as np
import pandas as pd
from .operators import delta, ts_max, rank
from .utils import run_alpha_factor


def alpha_041(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha041 factor.

    Formula:
        alpha_041 = (RANK(MAX(DELTA((VWAP),3),5)) * -1)
    """
    if 'date' in df.columns:
        index = pd.to_datetime(df['date'])
    else:
        index = df.index

    # Get VWAP, approximating as amount / volume if missing
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

    # Step 1: Compute DELTA(VWAP, 3)
    delta_vwap = delta(vwap, 3)

    # Step 2: Compute MAX(DELTA(VWAP, 3), 5)
    max_delta = np.maximum(delta_vwap, 5)

    # Step 3: Compute RANK(MAX(DELTA(VWAP, 3), 5))
    ranked = rank(max_delta)

    # Step 4: Multiply by -1
    alpha_values = ranked * -1

    return pd.Series(alpha_values, index=index, name='alpha_041')


def alpha041(
    code: str,
    benchmark: str = 'zz800',
    end_date: str = "2026-01-23",
    lookback: int = 350
) -> float:
    """
    Compute Alpha041 factor value for a stock at a specific date.
    """
    return run_alpha_factor(alpha_041, code, benchmark, end_date, lookback)
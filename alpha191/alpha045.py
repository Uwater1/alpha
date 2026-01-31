"""
Alpha045 factor implementation.

Formula:
    alpha_045 = (RANK(DELTA((((CLOSE*0.6)+(OPEN*0.4))),1))*RANK(CORR(VWAP,MEAN(VOLUME,150),15)))
"""

import numpy as np
import pandas as pd
from .operators import delta, rank, rolling_corr, ts_mean
from .utils import run_alpha_factor


def alpha_045(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha045 factor.

    Formula:
        alpha_045 = (RANK(DELTA((((CLOSE*0.6)+(OPEN*0.4))),1))*RANK(CORR(VWAP,MEAN(VOLUME,150),15)))
    """
    # Ensure we have required columns
    required_cols = ['close', 'open', 'volume']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in DataFrame")

    if 'date' in df.columns:
        index = pd.to_datetime(df['date'])
    else:
        index = df.index

    close = df['close'].values
    open_price = df['open'].values
    volume = df['volume'].values

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

    # Step 1: Compute (CLOSE * 0.6) + (OPEN * 0.4)
    weighted_price = (close * 0.6) + (open_price * 0.4)

    # Step 2: Compute DELTA((((CLOSE*0.6)+(OPEN*0.4))), 1)
    delta_weighted = delta(weighted_price, 1)

    # Step 3: Compute RANK(DELTA((((CLOSE*0.6)+(OPEN*0.4))), 1))
    rank_delta = rank(delta_weighted)

    # Step 4: Compute MEAN(VOLUME, 150)
    mean_volume = ts_mean(volume, 150)

    # Step 5: Compute CORR(VWAP, MEAN(VOLUME, 150), 15)
    corr_vwap_volume = rolling_corr(vwap, mean_volume, 15)

    # Step 6: Compute RANK(CORR(VWAP, MEAN(VOLUME, 150), 15))
    rank_corr = rank(corr_vwap_volume)

    # Step 7: Multiply the two ranks
    alpha_values = rank_delta * rank_corr

    return pd.Series(alpha_values, index=index, name='alpha_045')


def alpha045(
    code: str,
    benchmark: str = 'zz800',
    end_date: str = "2026-01-23",
    lookback: int = 350
) -> float:
    """
    Compute Alpha045 factor value for a stock at a specific date.
    """
    return run_alpha_factor(alpha_045, code, benchmark, end_date, lookback)
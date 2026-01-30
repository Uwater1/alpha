"""
Alpha039 factor implementation.

Formula:
    alpha_039 = ((RANK(DECAYLINEAR(DELTA((CLOSE), 2), 8)) - 
                  RANK(DECAYLINEAR(CORR(((VWAP * 0.3) + (OPEN * 0.7)), 
                  SUM(MEAN(VOLUME, 180), 37), 14), 12))) * -1)
"""

import numpy as np
import pandas as pd
from .operators import delta, decay_linear, rolling_corr, ts_mean, ts_sum, rank
from .utils import run_alpha_factor


def alpha_039(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha039 factor.

    Formula:
        alpha_039 = ((RANK(DECAYLINEAR(DELTA((CLOSE), 2), 8)) - 
                      RANK(DECAYLINEAR(CORR(((VWAP * 0.3) + (OPEN * 0.7)), 
                      SUM(MEAN(VOLUME, 180), 37), 14), 12))) * -1)
    """
    # Ensure we have required columns
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in required_cols:
        if col not in df.columns:
            # Check for amount if we need VWAP and it's missing
            if col == 'volume' and 'volume' not in df.columns:
                 raise ValueError(f"Required column '{col}' not found in DataFrame")
    
    if 'date' in df.columns:
        index = pd.to_datetime(df['date'])
    else:
        index = df.index

    close = df['close'].values
    open_price = df['open'].values
    volume = df['volume'].values
    
    # VWAP: approximate if missing
    if 'vwap' in df.columns:
        vwap = df['vwap'].values
    elif 'amount' in df.columns:
        # Avoid division by zero
        vol = volume.copy()
        vol[vol == 0] = np.nan
        vwap = df['amount'].values / vol
        # Fill NaNs from division by zero with simple average
        mask = np.isnan(vwap)
        vwap[mask] = (df['open'].values[mask] + df['high'].values[mask] + 
                      df['low'].values[mask] + df['close'].values[mask]) / 4
    else:
        vwap = (df['open'].values + df['high'].values + 
                df['low'].values + df['close'].values) / 4

    # Step 1: RANK(DECAYLINEAR(DELTA((CLOSE), 2), 8))
    delta_close = delta(close, 2)
    decay_delta = decay_linear(delta_close, 8)
    part1 = rank(decay_delta)

    # Step 2: RANK(DECAYLINEAR(CORR(((VWAP * 0.3) + (OPEN * 0.7)), SUM(MEAN(VOLUME, 180), 37), 14), 12))
    weighted_price = vwap * 0.3 + open_price * 0.7
    mean_vol_180 = ts_mean(volume, 180)
    sum_mean_vol = ts_sum(mean_vol_180, 37)
    
    correlation = rolling_corr(weighted_price, sum_mean_vol, 14)
    decay_corr = decay_linear(correlation, 12)
    part2 = rank(decay_corr)

    # Step 3: Final result
    alpha_values = (part1 - part2) * -1

    return pd.Series(alpha_values, index=index, name='alpha_039')


def alpha039(
    code: str,
    benchmark: str = 'zz800',
    end_date: str = "2026-01-23",
    lookback: int = 350
) -> float:
    """
    Compute Alpha039 factor value for a stock at a specific date.
    """
    return run_alpha_factor(alpha_039, code, benchmark, end_date, lookback)

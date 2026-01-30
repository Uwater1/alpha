"""
Alpha044 factor implementation.

Formula:
    alpha_044 = (TSRANK(DECAYLINEAR(CORR(((LOW)),MEAN(VOLUME,10),7),6),4)+TSRANK(DECAYLINEAR(DELTA((VWAP),3),10),15))
"""

import numpy as np
import pandas as pd
from .operators import rolling_corr, ts_mean, ts_rank, decay_linear, delta
from .utils import run_alpha_factor


def alpha_044(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha044 factor.

    Formula:
        alpha_044 = (TSRANK(DECAYLINEAR(CORR(((LOW)),MEAN(VOLUME,10),7),6),4)+TSRANK(DECAYLINEAR(DELTA((VWAP),3),10),15))
    """
    # Ensure we have required columns
    required_cols = ['low', 'volume']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in DataFrame")

    if 'date' in df.columns:
        index = pd.to_datetime(df['date'])
    else:
        index = df.index

    low = df['low'].values
    volume = df['volume'].values

    # Get VWAP, approximating as amount / volume if missing
    if 'vwap' not in df.columns:
        if 'amount' in df.columns and 'volume' in df.columns:
            vwap = np.where(df['volume'].values != 0, df['amount'].values / df['volume'].values, np.nan)
        else:
            raise ValueError("VWAP column not found and cannot be approximated (missing 'amount' or 'volume' columns)")
    else:
        vwap = df['vwap'].values

    # Step 1: Compute MEAN(VOLUME, 10)
    mean_volume = ts_mean(volume, 10)

    # Step 2: Compute CORR(LOW, MEAN(VOLUME, 10), 7)
    corr_low_volume = rolling_corr(low, mean_volume, 7)

    # Step 3: Compute DECAYLINEAR(CORR(LOW, MEAN(VOLUME, 10), 7), 6)
    decay_corr = decay_linear(corr_low_volume, 6)

    # Step 4: Compute TSRANK(DECAYLINEAR(CORR(LOW, MEAN(VOLUME, 10), 7), 6), 4)
    tsrank_decay_corr = ts_rank(decay_corr, 4)

    # Step 5: Compute DELTA(VWAP, 3)
    delta_vwap = delta(vwap, 3)

    # Step 6: Compute DECAYLINEAR(DELTA(VWAP, 3), 10)
    decay_delta_vwap = decay_linear(delta_vwap, 10)

    # Step 7: Compute TSRANK(DECAYLINEAR(DELTA(VWAP, 3), 10), 15)
    tsrank_decay_delta = ts_rank(decay_delta_vwap, 15)

    # Step 8: Add the two components
    alpha_values = tsrank_decay_corr + tsrank_decay_delta

    return pd.Series(alpha_values, index=index, name='alpha_044')


def alpha044(
    code: str,
    benchmark: str = 'zz800',
    end_date: str = "2026-01-23",
    lookback: int = 350
) -> float:
    """
    Compute Alpha044 factor value for a stock at a specific date.
    """
    return run_alpha_factor(alpha_044, code, benchmark, end_date, lookback)
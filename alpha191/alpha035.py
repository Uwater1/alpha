"""
Alpha035 factor implementation.

Formula:
    alpha_035 = (MIN(RANK(DECAYLINEAR(DELTA(OPEN,1),15)),RANK(DECAYLINEAR(CORR((VOLUME),((OPEN*0.65)+(OPEN*0.35)),17),7)))*-1)
"""

import numpy as np
import pandas as pd
from .operators import delta, decay_linear, rolling_corr, rank
from .utils import run_alpha_factor


def alpha_035(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha035 factor.

    Formula:
        alpha_035 = (MIN(RANK(DECAYLINEAR(DELTA(OPEN,1),15)),RANK(DECAYLINEAR(CORR((VOLUME),((OPEN*0.65)+(OPEN*0.35)),17),7)))*-1)
    """
    # Ensure we have required columns
    required_cols = ['open', 'volume']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in DataFrame")

    if 'date' in df.columns:
        index = pd.to_datetime(df['date'])
    else:
        index = df.index

    open_price = df['open'].values
    volume = df['volume'].values

    # Step 1: Compute DELTA(OPEN, 1)
    delta_open = delta(open_price, 1)

    # Step 2: Compute DECAYLINEAR(DELTA(OPEN,1), 15)
    decay_open = decay_linear(delta_open, 15)

    # Step 3: Compute RANK(DECAYLINEAR(DELTA(OPEN,1),15))
    rank_decay_open = rank(decay_open)

    # Step 4: Compute (OPEN*0.65)+(OPEN*0.35) = OPEN
    weighted_open = open_price * 0.65 + open_price * 0.35

    # Step 5: Compute CORR(VOLUME, weighted_open, 17)
    corr_vol_open = rolling_corr(volume, weighted_open, 17)

    # Step 6: Compute DECAYLINEAR(CORR(...), 7)
    decay_corr = decay_linear(corr_vol_open, 7)

    # Step 7: Compute RANK(DECAYLINEAR(CORR(...),7))
    rank_decay_corr = rank(decay_corr)

    # Step 8: Compute MIN(RANK(...), RANK(...))
    min_val = np.minimum(rank_decay_open, rank_decay_corr)

    # Step 9: Compute final formula
    alpha_values = min_val * -1

    return pd.Series(alpha_values, index=index, name='alpha_035')


def alpha035(
    code: str,
    benchmark: str = 'zz800',
    end_date: str = "2026-01-23",
    lookback: int = 350
) -> float:
    """
    Compute Alpha035 factor value for a stock at a specific date.
    """
    return run_alpha_factor(alpha_035, code, benchmark, end_date, lookback)

"""
Alpha025 factor implementation.

Formula:
    alpha_025 = ((-1*RANK((DELTA(CLOSE,7)*(1-RANK(DECAYLINEAR((VOLUME/MEAN(VOLUME,20)),9)))))*(1+RANK(SUM(RET,250))))
"""

import numpy as np
import pandas as pd
from .operators import delta, ts_mean, decay_linear, ts_sum, compute_ret, rank
from .utils import run_alpha_factor


def alpha_025(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha025 factor.

    Formula:
        alpha_025 = ((-1*RANK((DELTA(CLOSE,7)*(1-RANK(DECAYLINEAR((VOLUME/MEAN(VOLUME,20)),9)))))*(1+RANK(SUM(RET,250))))
    """
    # Ensure we have required columns
    required_cols = ['close', 'volume']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in DataFrame")

    if 'date' in df.columns:
        index = pd.to_datetime(df['date'])
    else:
        index = df.index

    close = df['close'].values
    volume = df['volume'].values

    # Step 1: Compute DELTA(CLOSE, 7)
    delta_close = delta(close, 7)

    # Step 2: Compute VOLUME/MEAN(VOLUME,20)
    mean_volume = ts_mean(volume, 20)
    vol_ratio = np.full(len(volume), np.nan)
    valid_mask = ~np.isnan(mean_volume) & (mean_volume != 0)
    vol_ratio[valid_mask] = volume[valid_mask] / mean_volume[valid_mask]

    # Step 3: Compute DECAYLINEAR(VOLUME/MEAN(VOLUME,20), 9)
    decay_vol = decay_linear(vol_ratio, 9)

    # Step 4: Compute RANK(DECAYLINEAR(...))
    rank_decay = rank(decay_vol)

    # Step 5: Compute (1-RANK(DECAYLINEAR(...)))
    one_minus_rank = 1 - rank_decay

    # Step 6: Compute DELTA(CLOSE,7)*(1-RANK(...))
    product = delta_close * one_minus_rank

    # Step 7: Compute RANK(DELTA(CLOSE,7)*(1-RANK(...)))
    rank_product = rank(product)

    # Step 8: Compute RET = delta(close, 1) / delay(close, 1)
    ret = compute_ret(close)

    # Step 9: Compute SUM(RET, 250)
    sum_ret = ts_sum(ret, 250)

    # Step 10: Compute RANK(SUM(RET, 250))
    rank_sum_ret = rank(sum_ret)

    # Step 11: Compute (1+RANK(SUM(RET,250)))
    one_plus_rank = 1 + rank_sum_ret

    # Step 12: Compute final formula
    alpha_values = -1 * rank_product * one_plus_rank

    return pd.Series(alpha_values, index=index, name='alpha_025')


def alpha025(
    code: str,
    benchmark: str = 'zz800',
    end_date: str = "2026-01-23",
    lookback: int = 350
) -> float:
    """
    Compute Alpha025 factor value for a stock at a specific date.
    """
    return run_alpha_factor(alpha_025, code, benchmark, end_date, lookback)

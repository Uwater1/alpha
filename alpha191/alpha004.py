"""
Alpha004 factor implementation.

Formula:
    alpha_004 = ((((SUM(CLOSE,8)/8)+STD(CLOSE,8))<(SUM(CLOSE,2)/2))?(-1*1):(((SUM(CLOSE,2)/2)<((SUM(CLOSE,8)/8)-STD(CLOSE,8)))?1:(((1<(VOLUME/MEAN(VOLUME,20)))||((VOLUME/MEAN(VOLUME,20))==1))?1:(-1*1))))
"""

import numpy as np
import pandas as pd
from .operators import ts_sum, ts_std, ts_mean
from .utils import run_alpha_factor


def alpha_004(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha004 factor.

    Formula:
        alpha_004 = ((((SUM(CLOSE,8)/8)+STD(CLOSE,8))<(SUM(CLOSE,2)/2))?(-1*1):(((SUM(CLOSE,2)/2)<((SUM(CLOSE,8)/8)-STD(CLOSE,8)))?1:(((1<(VOLUME/MEAN(VOLUME,20)))||((VOLUME/MEAN(VOLUME,20))==1))?1:(-1*1))))
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

    # Compute components
    # SUM(CLOSE,8)/8 - mean close over 8 days
    sum_close_8 = ts_sum(close, 8)
    mean_close_8 = sum_close_8 / 8

    # STD(CLOSE,8) - std of close over 8 days
    std_close_8 = ts_std(close, 8)

    # SUM(CLOSE,2)/2 - mean close over 2 days
    sum_close_2 = ts_sum(close, 2)
    mean_close_2 = sum_close_2 / 2

    # MEAN(VOLUME,20) - mean volume over 20 days
    mean_volume_20 = ts_mean(volume, 20)

    # Compute volume ratio, handle division by zero
    volume_ratio = np.full(len(volume), np.nan)
    valid_mask = ~np.isnan(mean_volume_20) & (mean_volume_20 != 0)
    volume_ratio[valid_mask] = volume[valid_mask] / mean_volume_20[valid_mask]

    # Calculate the boundaries
    upper_bound = mean_close_8 + std_close_8  # (SUM(CLOSE,8)/8)+STD(CLOSE,8)
    lower_bound = mean_close_8 - std_close_8  # (SUM(CLOSE,8)/8)-STD(CLOSE,8)

    # Apply the conditional logic
    # Condition 1: (mean8 + std8) < mean2
    cond1 = upper_bound < mean_close_2

    # Condition 2: mean2 < (mean8 - std8)
    cond2 = mean_close_2 < lower_bound

    # Condition 3: volume_ratio > 1 or volume_ratio == 1
    cond3 = (volume_ratio > 1) | (volume_ratio == 1)

    # Nested conditional: A?B:C in numpy
    # First level: if cond1 then -1 else check cond2
    result_values = np.where(
        cond1,
        -1.0,  # If cond1 is true, return -1
        np.where(
            cond2,
            1.0,  # If cond2 is true, return 1
            np.where(
                cond3,
                1.0,  # If cond3 is true, return 1
                -1.0  # Otherwise, return -1
            )
        )
    )

    # Set result to NaN where mean_volume_20 is NaN (not enough data)
    result_values[~valid_mask] = np.nan

    return pd.Series(result_values, index=index, name='alpha_004')


def alpha004(
    code: str,
    benchmark: str = 'zz800',
    end_date: str = "2026-01-23",
    lookback: int = 350
) -> float:
    """
    Compute Alpha004 factor value for a stock at a specific date.
    """
    return run_alpha_factor(alpha_004, code, benchmark, end_date, lookback)

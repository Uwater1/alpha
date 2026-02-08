"""
Alpha183 factor implementation.

Formula:
    MAX(SUMAC(CLOSE-MEAN(CLOSE,24)))-MIN(SUMAC(CLOSE-MEAN(CLOSE,24)))/STD(CLOSE,24)

    Note: This is interpreted as Rescaled Range (R/S) analysis.
    For a window of 24 days:
    1. Calculate deviations from the mean of the window.
    2. Calculate cumulative sum of these deviations.
    3. Range = Max(cumsum) - Min(cumsum).
    4. Result = Range / StdDev(window).
"""

import numpy as np
import pandas as pd
from numba import njit
from .utils import run_alpha_factor


@njit(cache=True)
def _alpha_183_impl(x: np.ndarray, window: int) -> np.ndarray:
    """
    Numba-accelerated implementation of Alpha183 (Rescaled Range).
    """
    n_len = len(x)
    result = np.full(n_len, np.nan, dtype=np.float64)

    if window > n_len:
        return result

    for i in range(window - 1, n_len):
        # Window indices: [start_idx, end_idx)
        start_idx = i - window + 1
        end_idx = i + 1

        # 1. Check for NaNs and calculate Mean
        w_sum = 0.0
        has_nan = False

        for j in range(start_idx, end_idx):
            val = x[j]
            if np.isnan(val):
                has_nan = True
                break
            w_sum += val

        if has_nan:
            continue

        mean = w_sum / window

        # 2. Calculate Cumsum of deviations, Min/Max of Cumsum, and SumSqDiff
        current_cumsum = 0.0
        max_cumsum = 0.0
        min_cumsum = 0.0
        sum_sq_diff = 0.0

        for j in range(start_idx, end_idx):
            val = x[j]
            dev = val - mean

            # Accumulate deviations
            current_cumsum += dev

            # Update min/max of cumulative deviations
            if current_cumsum > max_cumsum:
                max_cumsum = current_cumsum
            if current_cumsum < min_cumsum:
                min_cumsum = current_cumsum

            # Accumulate squared deviations for STD
            sum_sq_diff += dev * dev

        # 3. Calculate Standard Deviation (sample, ddof=1)
        if window > 1:
            # max(0, ...) to handle floating point small negative values
            variance = max(0.0, sum_sq_diff / (window - 1))
            std = np.sqrt(variance)
        else:
            std = 0.0

        # 4. Calculate Range = Max - Min
        # Note: Since sum of deviations is 0, the cumsum path starts at 0 (implicitly) and ends at 0.
        # However, R/S calculation typically considers the range of the partial sums S_1...S_n.
        # S_n should be 0. So the set of values includes 0.
        rng = max_cumsum - min_cumsum

        # 5. Result = Range / STD
        if std > 1e-9:  # Avoid division by zero
            result[i] = rng / std
        else:
            result[i] = 0.0

    return result


def alpha_183(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha183 factor.

    Formula:
        MAX(SUMAC(CLOSE-MEAN(CLOSE,24)))-MIN(SUMAC(CLOSE-MEAN(CLOSE,24)))/STD(CLOSE,24)
    """
    # Ensure we have required columns
    if 'close' not in df.columns:
        raise ValueError("Required column 'close' not found in DataFrame")

    # Get date index
    if 'date' in df.columns:
        index = pd.to_datetime(df['date'])
    else:
        index = df.index

    # Prepare input data
    close = df['close'].values.astype(np.float64)

    # Compute factor
    alpha_values = _alpha_183_impl(close, window=24)

    return pd.Series(alpha_values, index=index, name='alpha_183')


def alpha183(
    code: str,
    benchmark: str = 'zz800',
    end_date: str = "2026-01-23",
    lookback: int = 350
) -> float:
    """
    Compute Alpha183 factor value for a stock at a specific date.
    """
    return run_alpha_factor(alpha_183, code, benchmark, end_date, lookback)

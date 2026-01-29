"""
Operator functions for Alpha191 factors.
"""

import numpy as np
import pandas as pd
from scipy import stats


def ts_rank(x: np.ndarray, window: int = 6) -> np.ndarray:
    """
    Time-series rank within a rolling window.

    For each position i, computes the rank of x[i] within the window x[i-window+1:i+1].
    Ranks are normalized to [0, 1] scale. NaN values in the window are excluded from ranking.

    Parameters
    ----------
    x : np.ndarray
        Input array of values
    window : int, default 6
        Rolling window size

    Returns
    -------
    np.ndarray
        Array of ranks (0 to 1), with NaN for first (window-1) positions
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    result = np.full(n, np.nan)

    for i in range(window - 1, n):
        window_data = x[i - window + 1:i + 1]
        current_value = window_data[-1]

        # If current value is NaN, result is NaN
        if np.isnan(current_value):
            result[i] = np.nan
            continue

        # Create mask for valid (non-NaN) values
        valid_mask = ~np.isnan(window_data)
        valid_data = window_data[valid_mask]

        # Need at least 2 valid values to compute meaningful rank
        if len(valid_data) < 2:
            result[i] = np.nan
            continue

        # Compute ranks only on valid data
        # Using scipy.stats.rankdata with method='average' ensures:
        # - smallest → 1
        # - largest → window_size
        # - ties → average rank
        ranks = stats.rankdata(valid_data, method='average')

        # Find position of current value in valid data
        # Count how many valid values up to and including current
        current_pos = np.sum(valid_mask) - 1  # Position in valid_data array
        current_rank = ranks[current_pos]

        # Normalize to [0, 1] scale: (rank - 1) / (len(valid_data) - 1)
        result[i] = (current_rank - 1) / (len(valid_data) - 1)

    return result


def rolling_corr(a: np.ndarray, b: np.ndarray, window: int = 6) -> np.ndarray:
    """
    Rolling Pearson correlation between two arrays.

    Parameters
    ----------
    a : np.ndarray
        First input array
    b : np.ndarray
        Second input array
    window : int, default 6
        Rolling window size

    Returns
    -------
    np.ndarray
        Array of correlations, with NaN for first (window-1) positions
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if len(a) != len(b):
        raise ValueError("Input arrays must have the same length")

    n = len(a)
    result = np.full(n, np.nan)

    for i in range(window - 1, n):
        a_window = a[i - window + 1:i + 1]
        b_window = b[i - window + 1:i + 1]

        # Check for valid data (at least 2 non-NaN pairs for correlation)
        valid_mask = ~(np.isnan(a_window) | np.isnan(b_window))
        if np.sum(valid_mask) < 2:
            result[i] = np.nan
            continue

        # Compute Pearson correlation
        try:
            corr, _ = stats.pearsonr(a_window[valid_mask], b_window[valid_mask])
            result[i] = corr
        except (ValueError, RuntimeWarning):
            result[i] = np.nan

    return result

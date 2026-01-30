"""
Operator functions for Alpha191 factors.

This module provides time-series and cross-sectional operators used in
WorldQuant's Alpha191 factor library. Key operators include:

**Time-series operators (rolling window):**
- ts_rank, ts_sum, ts_mean, ts_std, ts_min, ts_max, ts_count, ts_prod

**Cross-sectional operators:**
- rank

**Utility operators:**
- delay, delta, sign, rolling_corr

**Numba Acceleration**
The following operators use Numba JIT compilation for performance:
- ts_sum, ts_mean, ts_std, ts_min, ts_max, ts_count, ts_prod

Operators using scipy (ts_rank, rolling_corr) are not JIT-accelerated
as they rely on scipy's optimized C implementations.
"""

import numpy as np
import pandas as pd
import warnings
from scipy import stats
from numba import njit


# =============================================================================
# Numba-accelerated core functions for rolling window operators
# =============================================================================

@njit(cache=True)
def _ts_sum_core(x: np.ndarray, n: int) -> np.ndarray:
    """Numba-accelerated core for rolling sum."""
    n_len = len(x)
    result = np.full(n_len, np.nan)
    
    for i in range(n - 1, n_len):
        window_sum = 0.0
        valid_count = 0
        
        for j in range(i - n + 1, i + 1):
            if not np.isnan(x[j]):
                window_sum += x[j]
                valid_count += 1
        
        if valid_count > 0:
            result[i] = window_sum
    
    return result


@njit(cache=True)
def _ts_mean_core(x: np.ndarray, n: int) -> np.ndarray:
    """Numba-accelerated core for rolling mean."""
    n_len = len(x)
    result = np.full(n_len, np.nan)
    
    for i in range(n - 1, n_len):
        window_sum = 0.0
        valid_count = 0
        
        for j in range(i - n + 1, i + 1):
            if not np.isnan(x[j]):
                window_sum += x[j]
                valid_count += 1
        
        if valid_count > 0:
            result[i] = window_sum / valid_count
    
    return result


@njit(cache=True)
def _ts_std_core(x: np.ndarray, n: int, ddof: int) -> np.ndarray:
    """Numba-accelerated core for rolling standard deviation."""
    n_len = len(x)
    result = np.full(n_len, np.nan)
    
    for i in range(n - 1, n_len):
        # First pass: compute mean
        window_sum = 0.0
        valid_count = 0
        
        for j in range(i - n + 1, i + 1):
            if not np.isnan(x[j]):
                window_sum += x[j]
                valid_count += 1
        
        if valid_count < ddof + 1:
            continue
        
        mean = window_sum / valid_count
        
        # Second pass: compute variance
        sq_diff_sum = 0.0
        for j in range(i - n + 1, i + 1):
            if not np.isnan(x[j]):
                diff = x[j] - mean
                sq_diff_sum += diff * diff
        
        variance = sq_diff_sum / (valid_count - ddof)
        result[i] = np.sqrt(variance)
    
    return result


@njit(cache=True)
def _ts_min_core(x: np.ndarray, n: int) -> np.ndarray:
    """Numba-accelerated core for rolling minimum."""
    n_len = len(x)
    result = np.full(n_len, np.nan)
    
    for i in range(n - 1, n_len):
        window_min = np.inf
        valid_count = 0
        
        for j in range(i - n + 1, i + 1):
            if not np.isnan(x[j]):
                if x[j] < window_min:
                    window_min = x[j]
                valid_count += 1
        
        if valid_count > 0:
            result[i] = window_min
    
    return result


@njit(cache=True)
def _ts_max_core(x: np.ndarray, n: int) -> np.ndarray:
    """Numba-accelerated core for rolling maximum."""
    n_len = len(x)
    result = np.full(n_len, np.nan)
    
    for i in range(n - 1, n_len):
        window_max = -np.inf
        valid_count = 0
        
        for j in range(i - n + 1, i + 1):
            if not np.isnan(x[j]):
                if x[j] > window_max:
                    window_max = x[j]
                valid_count += 1
        
        if valid_count > 0:
            result[i] = window_max
    
    return result


@njit(cache=True)
def _ts_count_core(x: np.ndarray, n: int) -> np.ndarray:
    """Numba-accelerated core for rolling count of non-zero values."""
    n_len = len(x)
    result = np.full(n_len, np.nan)
    
    for i in range(n - 1, n_len):
        count = 0
        valid_count = 0
        
        for j in range(i - n + 1, i + 1):
            if not np.isnan(x[j]):
                valid_count += 1
                if x[j] != 0:
                    count += 1
        
        if valid_count > 0:
            result[i] = float(count)
    
    return result


@njit(cache=True)
def _ts_prod_core(x: np.ndarray, n: int) -> np.ndarray:
    """Numba-accelerated core for rolling product."""
    n_len = len(x)
    result = np.full(n_len, np.nan)
    
    for i in range(n - 1, n_len):
        window_prod = 1.0
        valid_count = 0
        
        for j in range(i - n + 1, i + 1):
            if not np.isnan(x[j]):
                window_prod *= x[j]
                valid_count += 1
        
        if valid_count > 0:
            result[i] = window_prod
    
    return result


@njit(cache=True)
def _covariance_core(x: np.ndarray, y: np.ndarray, n: int, ddof: int) -> np.ndarray:
    """Numba-accelerated core for rolling covariance."""
    n_len = len(x)
    result = np.full(n_len, np.nan)
    
    for i in range(n - 1, n_len):
        # Collect valid pairs (both x and y are not NaN)
        x_sum = 0.0
        y_sum = 0.0
        valid_count = 0
        
        for j in range(i - n + 1, i + 1):
            if not np.isnan(x[j]) and not np.isnan(y[j]):
                x_sum += x[j]
                y_sum += y[j]
                valid_count += 1
        
        # Need at least (ddof + 2) valid pairs for covariance
        if valid_count < ddof + 2:
            continue
        
        # Compute means
        x_mean = x_sum / valid_count
        y_mean = y_sum / valid_count
        
        # Compute covariance
        cov_sum = 0.0
        for j in range(i - n + 1, i + 1):
            if not np.isnan(x[j]) and not np.isnan(y[j]):
                cov_sum += (x[j] - x_mean) * (y[j] - y_mean)
        
        result[i] = cov_sum / (valid_count - ddof)
    
    return result


@njit(cache=True)
def _regression_beta_core(x: np.ndarray, y: np.ndarray, n: int) -> np.ndarray:
    """Numba-accelerated core for rolling regression beta."""
    n_len = len(x)
    result = np.full(n_len, np.nan)
    
    for i in range(n - 1, n_len):
        # Collect valid pairs (both x and y are not NaN)
        x_sum = 0.0
        y_sum = 0.0
        valid_count = 0
        
        for j in range(i - n + 1, i + 1):
            if not np.isnan(x[j]) and not np.isnan(y[j]):
                x_sum += x[j]
                y_sum += y[j]
                valid_count += 1
        
        # Need at least 2 valid pairs
        if valid_count < 2:
            continue
        
        # Compute means
        x_mean = x_sum / valid_count
        y_mean = y_sum / valid_count
        
        # Compute covariance and variance
        cov_xy = 0.0
        var_y = 0.0
        for j in range(i - n + 1, i + 1):
            if not np.isnan(x[j]) and not np.isnan(y[j]):
                x_diff = x[j] - x_mean
                y_diff = y[j] - y_mean
                cov_xy += x_diff * y_diff
                var_y += y_diff * y_diff
        
        # Avoid division by zero
        if var_y == 0:
            continue
        
        result[i] = cov_xy / var_y
    
    return result


@njit(cache=True)
def _regression_residual_core(x: np.ndarray, y: np.ndarray, n: int) -> np.ndarray:
    """Numba-accelerated core for rolling regression residual."""
    n_len = len(x)
    result = np.full(n_len, np.nan)
    
    for i in range(n - 1, n_len):
        # Check if current values are valid
        if np.isnan(x[i]) or np.isnan(y[i]):
            continue
        
        # Collect valid pairs for the window (both x and y are not NaN)
        x_sum = 0.0
        y_sum = 0.0
        valid_count = 0
        
        for j in range(i - n + 1, i + 1):
            if not np.isnan(x[j]) and not np.isnan(y[j]):
                x_sum += x[j]
                y_sum += y[j]
                valid_count += 1
        
        # Need at least 2 valid pairs
        if valid_count < 2:
            continue
        
        # Compute means
        x_mean = x_sum / valid_count
        y_mean = y_sum / valid_count
        
        # Compute covariance and variance
        cov_xy = 0.0
        var_y = 0.0
        for j in range(i - n + 1, i + 1):
            if not np.isnan(x[j]) and not np.isnan(y[j]):
                x_diff = x[j] - x_mean
                y_diff = y[j] - y_mean
                cov_xy += x_diff * y_diff
                var_y += y_diff * y_diff
        
        # Avoid division by zero
        if var_y == 0:
            continue
        
        # Compute beta and alpha
        beta = cov_xy / var_y
        alpha = x_mean - beta * y_mean
        
        # Compute residual: x[i] - (alpha + beta * y[i])
        result[i] = x[i] - (alpha + beta * y[i])
    
    return result


# =============================================================================
# Public operator functions
# =============================================================================

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

    Notes
    -----
    This function uses scipy.stats.rankdata for ranking and is not JIT-accelerated
    as scipy's implementation is already highly optimized C code.
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
        ranks = stats.rankdata(valid_data, method='average')

        # Find position of current value in valid data
        current_pos = np.sum(valid_mask) - 1
        current_rank = ranks[current_pos]

        # Normalize to [0, 1] scale
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

    Notes
    -----
    This function uses scipy.stats.pearsonr for correlation and is not JIT-accelerated
    as scipy's implementation is already highly optimized.
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
            # Suppress ConstantInputWarning for constant arrays
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=stats.ConstantInputWarning)
                corr, _ = stats.pearsonr(a_window[valid_mask], b_window[valid_mask])
            result[i] = corr
        except (ValueError, RuntimeWarning):
            result[i] = np.nan

    return result


def delay(x: np.ndarray, n: int) -> np.ndarray:
    """
    Shift array backward by n periods.

    Formula: DELAY(A, n) = A[t-n]

    This is equivalent to pandas shift(n) or np.roll with NaN fill.
    Returns an array where the first n values are NaN, and subsequent
    values are the original array shifted back by n positions.

    Parameters
    ----------
    x : np.ndarray
        Input array
    n : int
        Number of periods to shift back. Must be non-negative.

    Returns
    -------
    np.ndarray
        Shifted array, first n values are NaN

    Examples
    --------
    >>> delay(np.array([1, 2, 3, 4, 5]), 2)
    array([nan, nan, 1., 2., 3.])

    >>> delay(np.array([1, 2, 3, 4, 5]), 0)
    array([1., 2., 3., 4., 5.])

    Notes
    -----
    - n = 0 returns a copy of the original array
    - n < 0 raises ValueError
    - n >= len(x) returns an all-NaN array
    - NaN values in input are preserved at their shifted positions
    - This function uses vectorized NumPy operations (no JIT needed)
    """
    x = np.asarray(x, dtype=float)

    if n < 0:
        raise ValueError("n must be non-negative")

    n = int(n)
    result_len = len(x)

    # Handle edge case: n >= len(x)
    if n >= result_len:
        return np.full(result_len, np.nan)

    # Handle edge case: n = 0
    if n == 0:
        return x.copy()

    # Create result array filled with NaN
    result = np.full(result_len, np.nan)

    # Fill in the shifted values
    result[n:] = x[:-n]

    return result


def delta(x: np.ndarray, n: int) -> np.ndarray:
    """
    Compute difference between current value and value n periods ago.

    Formula: DELTA(A, n) = A[t] - A[t-n]

    Computes the first difference (when n=1) or nth difference between
    the current value and the value n periods back.

    Parameters
    ----------
    x : np.ndarray
        Input array
    n : int
        Period offset. Must be non-negative.

    Returns
    -------
    np.ndarray
        Difference array, first n values are NaN

    Examples
    --------
    >>> delta(np.array([1, 2, 4, 7, 11]), 1)
    array([nan, 1., 2., 3., 4.])

    >>> delta(np.array([10, 11, 13, 16, 20]), 2)
    array([nan, nan, 3., 5., 7.])

    Notes
    -----
    - First n values are NaN because there is no previous value to subtract
    - n = 0 returns all zeros (x - x)
    - n < 0 raises ValueError
    - NaN values propagate: if either x[t] or x[t-n] is NaN, result is NaN
    - This function uses vectorized NumPy operations (no JIT needed)
    """
    x = np.asarray(x, dtype=float)

    if n < 0:
        raise ValueError("n must be non-negative")

    # delta(x, n) = x - delay(x, n)
    delayed = delay(x, n)
    return x - delayed


def rank(x: np.ndarray) -> np.ndarray:
    """
    Cross-sectional rank of elements, normalized to [0, 1].

    This ranks elements in the array from smallest (0) to largest (1).
    Ties are assigned average rank. This is a cross-sectional operation,
    meaning it ranks across elements at the same time point (unlike ts_rank
    which ranks over time).

    Formula:
        ranks = rankdata(x, method='average')
        normalized = (ranks - 1) / (len(valid_data) - 1)

    Parameters
    ----------
    x : np.ndarray
        Input array (cross-section of values at one time point)

    Returns
    -------
    np.ndarray
        Ranked values normalized to [0, 1]
        NaN values are excluded from ranking and remain NaN

    Examples
    --------
    >>> rank(np.array([10, 5, 15, 5]))
    array([0.5, 0.0, 1.0, 0.0])

    >>> rank(np.array([3, 1, 2, np.nan, 4]))
    array([0.666..., 0.0, 0.333..., nan, 1.0])

    Notes
    -----
    - This is CROSS-SECTIONAL ranking (across stocks at same time)
    - Different from ts_rank which ranks over time
    - NaN values are excluded from ranking and remain NaN in output
    - If only 1 valid value, returns 0.5 for that value
    - If all values are NaN, returns all NaN
    - Ties receive the average of their ranks
    - This function uses scipy.stats.rankdata (no JIT needed)
    """
    x = np.asarray(x, dtype=float)

    # Handle empty array
    if len(x) == 0:
        return np.array([], dtype=float)

    # Create result array filled with NaN
    result = np.full_like(x, np.nan, dtype=float)

    # Create mask for valid (non-NaN) values
    valid_mask = ~np.isnan(x)
    valid_data = x[valid_mask]

    # Handle all-NaN case
    if len(valid_data) == 0:
        return result

    # Handle single-value case
    if len(valid_data) == 1:
        result[valid_mask] = 0.5
        return result

    # Compute ranks using scipy.stats.rankdata with 'average' method
    ranks = stats.rankdata(valid_data, method='average')

    # Normalize to [0, 1] scale
    normalized_ranks = (ranks - 1) / (len(valid_data) - 1)

    # Assign normalized ranks back to valid positions
    result[valid_mask] = normalized_ranks

    return result


def sign(x: np.ndarray) -> np.ndarray:
    """
    Sign function.

    Formula: SIGN(A) = 1 if A > 0, 0 if A = 0, -1 if A < 0

    This is a wrapper around np.sign for consistency with other
    Alpha191 operators. It returns the sign of each element in the array.

    Parameters
    ----------
    x : np.ndarray
        Input array

    Returns
    -------
    np.ndarray
        Sign array: 1, 0, or -1 for each element
        NaN values remain NaN

    Examples
    --------
    >>> sign(np.array([1.5, -2.0, 0.0, 3.0]))
    array([1., -1., 0., 1.])

    >>> sign(np.array([1.0, np.nan, -3.0]))
    array([1., nan, -1.])

    Notes
    -----
    - np.sign returns NaN for NaN inputs (preserves NaN)
    - np.sign returns 0 for 0 inputs
    - This is a simple element-wise operation (no JIT needed)
    """
    x = np.asarray(x, dtype=float)
    return np.sign(x)


def ts_sum(x: np.ndarray, n: int) -> np.ndarray:
    """
    Sum over rolling window (Numba-accelerated).

    Formula: SUM(A, n) = sum(A[t-n+1:t+1])

    Computes the sum of values within a rolling window of size n.
    NaN values in the window are excluded from computation.

    Parameters
    ----------
    x : np.ndarray
        Input array of values
    n : int
        Rolling window size. Must be positive.

    Returns
    -------
    np.ndarray
        Array of rolling sums, with NaN for first (n-1) positions
        and positions with insufficient valid data

    Examples
    --------
    >>> ts_sum(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), 3)
    array([nan, nan, 6., 9., 12.])

    >>> ts_sum(np.array([1.0, np.nan, 3.0, 4.0, 5.0]), 3)
    array([nan, nan, 4., 7., 12.])

    Notes
    -----
    - First (n-1) values are NaN due to insufficient data
    - NaN values in window are excluded from sum
    - Returns NaN if no valid values in window
    - Minimum 1 valid value required for non-NaN result
    - This function is accelerated using Numba JIT compilation
    """
    x = np.asarray(x, dtype=float)
    n = int(n)

    if n <= 0:
        raise ValueError("n must be positive")

    return _ts_sum_core(x, n)


def ts_mean(x: np.ndarray, n: int) -> np.ndarray:
    """
    Mean over rolling window (Numba-accelerated).

    Formula: MEAN(A, n) = mean(A[t-n+1:t+1])

    Computes the arithmetic mean of values within a rolling window of size n.
    NaN values in the window are excluded from computation.

    Parameters
    ----------
    x : np.ndarray
        Input array of values
    n : int
        Rolling window size. Must be positive.

    Returns
    -------
    np.ndarray
        Array of rolling means, with NaN for first (n-1) positions
        and positions with insufficient valid data

    Examples
    --------
    >>> ts_mean(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), 3)
    array([nan, nan, 2., 3., 4.])

    >>> ts_mean(np.array([1.0, np.nan, 3.0, 4.0, 5.0]), 3)
    array([nan, nan, 2., 3.5, 4.5])

    Notes
    -----
    - First (n-1) values are NaN due to insufficient data
    - NaN values in window are excluded from mean calculation
    - Returns NaN if no valid values in window
    - Minimum 1 valid value required for non-NaN result
    - This function is accelerated using Numba JIT compilation
    """
    x = np.asarray(x, dtype=float)
    n = int(n)

    if n <= 0:
        raise ValueError("n must be positive")

    return _ts_mean_core(x, n)


def ts_std(x: np.ndarray, n: int, ddof: int = 1) -> np.ndarray:
    """
    Standard deviation over rolling window (Numba-accelerated).

    Formula: STD(A, n) = std(A[t-n+1:t+1])

    Computes the standard deviation of values within a rolling window of size n.
    NaN values in the window are excluded from computation.

    Parameters
    ----------
    x : np.ndarray
        Input array of values
    n : int
        Rolling window size. Must be positive.
    ddof : int, default 1
        Delta degrees of freedom for standard deviation calculation.
        ddof=1 computes sample standard deviation (N-1 denominator).
        ddof=0 computes population standard deviation (N denominator).

    Returns
    -------
    np.ndarray
        Array of rolling standard deviations, with NaN for first (n-1) positions
        and positions with insufficient valid data

    Examples
    --------
    >>> ts_std(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), 3)
    array([nan, nan, 1., 1., 1.])

    >>> ts_std(np.array([1.0, np.nan, 3.0, 4.0, 5.0]), 3)
    array([nan, nan, 1.414..., 0.707..., 0.707...])

    Notes
    -----
    - First (n-1) values are NaN due to insufficient data
    - NaN values in window are excluded from std calculation
    - Returns NaN if fewer than (ddof + 1) valid values in window
    - Default ddof=1 for sample standard deviation
    - Minimum 2 valid values required for default ddof=1
    - This function is accelerated using Numba JIT compilation
    """
    x = np.asarray(x, dtype=float)
    n = int(n)

    if n <= 0:
        raise ValueError("n must be positive")

    return _ts_std_core(x, n, ddof)


def ts_min(x: np.ndarray, n: int) -> np.ndarray:
    """
    Minimum over rolling window (Numba-accelerated).

    Formula: TSMIN(A, n) = min(A[t-n+1:t+1])

    Computes the minimum value within a rolling window of size n.
    NaN values in the window are excluded from computation.

    Parameters
    ----------
    x : np.ndarray
        Input array of values
    n : int
        Rolling window size. Must be positive.

    Returns
    -------
    np.ndarray
        Array of rolling minimums, with NaN for first (n-1) positions
        and positions with no valid data

    Examples
    --------
    >>> ts_min(np.array([5.0, 2.0, 8.0, 1.0, 9.0]), 3)
    array([nan, nan, 2., 1., 1.])

    >>> ts_min(np.array([5.0, np.nan, 8.0, 1.0, 9.0]), 3)
    array([nan, nan, 5., 1., 1.])

    Notes
    -----
    - First (n-1) values are NaN due to insufficient data
    - NaN values in window are excluded from min calculation
    - Returns NaN if no valid values in window
    - Minimum 1 valid value required for non-NaN result
    - This function is accelerated using Numba JIT compilation
    """
    x = np.asarray(x, dtype=float)
    n = int(n)

    if n <= 0:
        raise ValueError("n must be positive")

    return _ts_min_core(x, n)


def ts_max(x: np.ndarray, n: int) -> np.ndarray:
    """
    Maximum over rolling window (Numba-accelerated).

    Formula: TSMAX(A, n) = max(A[t-n+1:t+1])

    Computes the maximum value within a rolling window of size n.
    NaN values in the window are excluded from computation.

    Parameters
    ----------
    x : np.ndarray
        Input array of values
    n : int
        Rolling window size. Must be positive.

    Returns
    -------
    np.ndarray
        Array of rolling maximums, with NaN for first (n-1) positions
        and positions with no valid data

    Examples
    --------
    >>> ts_max(np.array([5.0, 2.0, 8.0, 1.0, 9.0]), 3)
    array([nan, nan, 8., 8., 9.])

    >>> ts_max(np.array([5.0, np.nan, 8.0, 1.0, 9.0]), 3)
    array([nan, nan, 8., 8., 9.])

    Notes
    -----
    - First (n-1) values are NaN due to insufficient data
    - NaN values in window are excluded from max calculation
    - Returns NaN if no valid values in window
    - Minimum 1 valid value required for non-NaN result
    - This function is accelerated using Numba JIT compilation
    """
    x = np.asarray(x, dtype=float)
    n = int(n)

    if n <= 0:
        raise ValueError("n must be positive")

    return _ts_max_core(x, n)


def ts_count(condition: np.ndarray, n: int) -> np.ndarray:
    """
    Count True values in rolling window (Numba-accelerated).

    Formula: COUNT(condition, n) = sum(condition[t-n+1:t+1] == True)

    Counts the number of True values within a rolling window of size n.
    NaN values in the condition array are treated as False.

    Parameters
    ----------
    condition : np.ndarray
        Input array of boolean values (or convertible to boolean).
        NaN values are treated as False.
    n : int
        Rolling window size. Must be positive.

    Returns
    -------
    np.ndarray
        Array of rolling counts (as float), with NaN for first (n-1) positions
        and positions with no valid data

    Examples
    --------
    >>> ts_count(np.array([True, False, True, True, False]), 3)
    array([nan, nan, 2., 2., 1.])

    >>> ts_count(np.array([1.0, 0.0, 3.0, np.nan, 2.0]), 3)  # non-zero is True
    array([nan, nan, 2., 1., 1.])

    Notes
    -----
    - First (n-1) values are NaN due to insufficient data
    - NaN values are treated as False (not counted)
    - Returns 0.0 if all values in window are False/NaN
    - Minimum 1 non-NaN value required for non-NaN result
    - Result is returned as float array (to support NaN)
    - This function is accelerated using Numba JIT compilation
    """
    condition = np.asarray(condition, dtype=float)
    n = int(n)

    if n <= 0:
        raise ValueError("n must be positive")

    return _ts_count_core(condition, n)


def ts_prod(x: np.ndarray, n: int) -> np.ndarray:
    """
    Product over rolling window (Numba-accelerated).

    Formula: PROD(A, n) = product(A[t-n+1:t+1])

    Computes the product of values within a rolling window of size n.
    NaN values in the window are excluded from computation.

    Parameters
    ----------
    x : np.ndarray
        Input array of values
    n : int
        Rolling window size. Must be positive.

    Returns
    -------
    np.ndarray
        Array of rolling products, with NaN for first (n-1) positions
        and positions with insufficient valid data

    Examples
    --------
    >>> ts_prod(np.array([2.0, 3.0, 4.0, 5.0]), 3)
    array([nan, nan, 24., 60.])

    >>> ts_prod(np.array([2.0, np.nan, 4.0, 5.0]), 3)
    array([nan, nan, 8., 20.])

    Notes
    -----
    - First (n-1) values are NaN due to insufficient data
    - NaN values in window are excluded from product
    - Returns NaN if no valid values in window
    - Minimum 1 valid value required for non-NaN result
    - Can overflow easily with large values or large windows
    - Zero values will result in zero product
    - This function is accelerated using Numba JIT compilation
    """
    x = np.asarray(x, dtype=float)
    n = int(n)

    if n <= 0:
        raise ValueError("n must be positive")

    return _ts_prod_core(x, n)


def covariance(x: np.ndarray, y: np.ndarray, n: int, ddof: int = 1) -> np.ndarray:
    """
    Rolling covariance between two arrays.
    
    Formula: COV(A, B, n) = E[(A - mean(A)) * (B - mean(B))]
    
    Computes the sample covariance between two arrays within a rolling window
    of size n. NaN values in either array are excluded pairwise.
    
    Parameters
    ----------
    x : np.ndarray
        First input array
    y : np.ndarray
        Second input array (same length as x)
    n : int
        Window size. Must be positive.
    ddof : int, default 1
        Delta degrees of freedom. ddof=1 for sample covariance (N-1 denominator),
        ddof=0 for population covariance (N denominator).
        
    Returns
    -------
    np.ndarray
        Rolling covariance, first (n-1) values are NaN
        
    Examples
    --------
    >>> x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    >>> covariance(x, y, 3)
    array([nan, nan, 1., 1., 1.])
    
    >>> covariance(np.array([1.0, np.nan, 3.0, 4.0]), np.array([2.0, 4.0, 6.0, 8.0]), 3)
    array([nan, nan, 2., 2.])
    
    Notes
    -----
    - Requires at least (ddof + 2) valid pairs in window
    - NaN in either array at same position → that pair is excluded
    - Uses sample covariance (ddof=1) by default
    - This function is accelerated using Numba JIT compilation
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    
    if len(x) != len(y):
        raise ValueError("Input arrays must have the same length")
    
    n = int(n)
    
    if n <= 0:
        raise ValueError("n must be positive")
    
    return _covariance_core(x, y, n, ddof)


def regression_beta(x: np.ndarray, y: np.ndarray, n: int) -> np.ndarray:
    """
    Rolling regression beta coefficient.
    
    Formula: REGBETA(A, B, n) where A ~ B
             β = cov(A, B) / var(B)
    
    This computes the slope when regressing x on y:
        x = α + β*y + ε
    
    Parameters
    ----------
    x : np.ndarray
        Dependent variable
    y : np.ndarray
        Independent variable
    n : int
        Window size. Must be positive.
        
    Returns
    -------
    np.ndarray
        Rolling beta coefficient, first (n-1) values are NaN
        
    Examples
    --------
    >>> x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    >>> regression_beta(x, y, 3)
    array([nan, nan, 0.5, 0.5, 0.5])
    
    Notes
    -----
    - Requires at least 2 valid pairs in window
    - If var(y) = 0, returns NaN for that position
    - NaN in either array at same position → that pair is excluded
    - This function is accelerated using Numba JIT compilation
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    
    if len(x) != len(y):
        raise ValueError("Input arrays must have the same length")
    
    n = int(n)
    
    if n <= 0:
        raise ValueError("n must be positive")
    
    return _regression_beta_core(x, y, n)


def regression_residual(x: np.ndarray, y: np.ndarray, n: int) -> np.ndarray:
    """
    Rolling regression residuals.
    
    Formula: REGRESI(A, B, n)
             Residual = A - (α + β*B)
             where β = cov(A,B)/var(B)
                   α = mean(A) - β*mean(B)
    
    For each position i, computes the residual from regressing x on y
    using the window [i-n+1, i]. The residual at position i is:
        ε[i] = x[i] - (α + β*y[i])
    
    Parameters
    ----------
    x : np.ndarray
        Dependent variable
    y : np.ndarray
        Independent variable
    n : int
        Window size. Must be positive.
        
    Returns
    -------
    np.ndarray
        Rolling regression residuals, first (n-1) values are NaN
        
    Examples
    --------
    >>> x = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    >>> y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> regression_residual(x, y, 3)
    # x = 2*y, so residuals should be approximately 0
    array([nan, nan, 0., 0., 0.])
    
    Notes
    -----
    - Requires at least 2 valid pairs in window
    - If var(y) = 0 or current values are NaN, returns NaN
    - NaN in either array at same position → that pair is excluded from window
    - The residual is computed using the regression coefficients from the window
      applied to the current values at position i
    - This function is accelerated using Numba JIT compilation
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    
    if len(x) != len(y):
        raise ValueError("Input arrays must have the same length")
    
    n = int(n)
    
    if n <= 0:
        raise ValueError("n must be positive")
    
    return _regression_residual_core(x, y, n)


@njit(cache=True)
def _wma_core(x: np.ndarray, n: int) -> np.ndarray:
    """Numba-accelerated core for weighted moving average with exponential decay weights."""
    n_len = len(x)
    result = np.full(n_len, np.nan)
    
    # Compute weights: [0.9^(n-1), 0.9^(n-2), ..., 0.9^0]
    weights = np.empty(n, dtype=np.float64)
    for i in range(n):
        weights[i] = 0.9 ** (n - 1 - i)
    
    for i in range(n - 1, n_len):
        window_sum = 0.0
        weight_sum = 0.0
        
        for j in range(n):
            idx = i - n + 1 + j
            if not np.isnan(x[idx]):
                window_sum += x[idx] * weights[j]
                weight_sum += weights[j]
        
        if weight_sum > 0:
            result[i] = window_sum / weight_sum
    
    return result


@njit(cache=True)
def _decay_linear_core(x: np.ndarray, d: int) -> np.ndarray:
    """Numba-accelerated core for linear decay weighted average."""
    n_len = len(x)
    result = np.full(n_len, np.nan)
    
    # Compute weights: [d, d-1, d-2, ..., 2, 1]
    # Sum of weights = d * (d + 1) / 2
    weight_sum_total = d * (d + 1) / 2.0
    
    for i in range(d - 1, n_len):
        window_sum = 0.0
        weight_sum = 0.0
        
        for j in range(d):
            idx = i - d + 1 + j
            if not np.isnan(x[idx]):
                # Weight for position j: (j + 1) since j goes from 0 to d-1
                # Most recent (j = d-1) has weight d
                weight = j + 1
                window_sum += x[idx] * weight
                weight_sum += weight
        
        if weight_sum > 0:
            result[i] = window_sum / weight_sum
    
    return result


def sma(x: np.ndarray, n: int, m: int) -> np.ndarray:
    """
    Special Moving Average with memory (exponential-like).
    
    Formula: SMA(A, n, m)
             Y[0] = A[0]
             Y[t] = (m*A[t] + (n-m)*Y[t-1]) / n
    
    This is similar to EMA but with specific weight parameters.
    
    Parameters
    ----------
    x : np.ndarray
        Input array
    n : int
        Denominator (total weight)
    m : int
        Weight for current value
        
    Returns
    -------
    np.ndarray
        Special moving average
        
    Examples
    --------
    >>> sma(np.array([1, 2, 3, 4, 5]), n=3, m=1)
    array([1., 1.33, 1.89, 2.59, 3.40])
    
    Notes
    -----
    - Y[0] is initialized to A[0]
    - Requires iterative computation (not vectorizable)
    - NaN propagates forward once encountered (breaks the chain)
    - If n <= 0, m <= 0, or m > n, raises ValueError
    """
    x = np.asarray(x, dtype=float)
    n = int(n)
    m = int(m)
    
    if n <= 0:
        raise ValueError("n must be positive")
    if m <= 0:
        raise ValueError("m must be positive")
    if m > n:
        raise ValueError("m must not exceed n")
    
    n_len = len(x)
    
    if n_len == 0:
        return np.array([], dtype=float)
    
    result = np.full(n_len, np.nan, dtype=float)
    
    # Find first non-NaN value to start the chain
    start_idx = -1
    for i in range(n_len):
        if not np.isnan(x[i]):
            start_idx = i
            break
            
    if start_idx == -1:
        return result
        
    result[start_idx] = x[start_idx]
    
    for i in range(start_idx + 1, n_len):
        if np.isnan(x[i]):
            # If current value is NaN, we have a choice:
            # 1. Propagate NaN (current implementation)
            # 2. Keep last valid (like EMA)
            # Standard JoinQuant SMA usually skips NaNs and uses last valid
            result[i] = result[i-1]
        else:
            # Y[t] = (m*A[t] + (n-m)*Y[t-1]) / n
            # If result[i-1] is NaN (shouldn't happen now after start_idx), handle it
            prev_y = result[i-1] if not np.isnan(result[i-1]) else x[i]
            result[i] = (m * x[i] + (n - m) * prev_y) / n
    
    return result


def wma(x: np.ndarray, n: int) -> np.ndarray:
    """
    Weighted Moving Average with exponential decay weights.
    
    Formula: WMA(A, n)
             weights = [0.9^(n-1), 0.9^(n-2), ..., 0.9^0]
             WMA[t] = sum(A[t-n+1:t+1] * weights) / sum(weights)
    
    Parameters
    ----------
    x : np.ndarray
        Input array
    n : int
        Window size. Must be positive.
        
    Returns
    -------
    np.ndarray
        Weighted moving average
        
    Examples
    --------
    >>> wma(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), 3)
    # weights = [0.81, 0.9, 1.0] (normalized)
    
    Notes
    -----
    - More recent values have higher weight (0.9^0 = 1.0)
    - Oldest value has weight 0.9^(n-1)
    - NaN values in the window are excluded and weights are renormalized
    - First (n-1) values are NaN due to insufficient data
    - This function is accelerated using Numba JIT compilation
    """
    x = np.asarray(x, dtype=float)
    n = int(n)
    
    if n <= 0:
        raise ValueError("n must be positive")
    
    return _wma_core(x, n)


def decay_linear(x: np.ndarray, d: int) -> np.ndarray:
    """
    Linear decay weighted average.
    
    Formula: DECAYLINEAR(A, d)
             weights = [d, d-1, d-2, ..., 2, 1]
             normalized_weights = weights / sum(weights)
             result[t] = sum(A[t-d+1:t+1] * normalized_weights)
    
    Parameters
    ----------
    x : np.ndarray
        Input array
    d : int
        Decay period (window size). Must be positive.
        
    Returns
    -------
    np.ndarray
        Linear decay weighted average
        
    Examples
    --------
    >>> decay_linear(np.array([1, 2, 3, 4, 5]), 3)
    # weights = [3, 2, 1] / 6 = [0.5, 0.333, 0.167]
    # At t=2: 1*0.5 + 2*0.333 + 3*0.167 = 1.833
    
    Notes
    -----
    - Most recent value gets highest weight (d)
    - Oldest value gets lowest weight (1)
    - NaN values in the window are excluded and weights are renormalized
    - First (d-1) values are NaN due to insufficient data
    - This function is accelerated using Numba JIT compilation
    """
    x = np.asarray(x, dtype=float)
    d = int(d)
    
    if d <= 0:
        raise ValueError("d must be positive")
    
    return _decay_linear_core(x, d)


# =============================================================================
# Numba-accelerated core functions for conditional and special operators
# =============================================================================

@njit(cache=True)
def _sum_if_core(x: np.ndarray, n: int, condition: np.ndarray) -> np.ndarray:
    """Numba-accelerated core for rolling conditional sum."""
    n_len = len(x)
    result = np.full(n_len, np.nan)
    
    for i in range(n - 1, n_len):
        window_sum = 0.0
        true_count = 0
        valid_condition_count = 0
        
        for j in range(i - n + 1, i + 1):
            # Check if condition is valid (not NaN)
            if not np.isnan(condition[j]):
                valid_condition_count += 1
                # Only sum if condition is True (not 0)
                if condition[j] != 0 and not np.isnan(x[j]):
                    window_sum += x[j]
                    true_count += 1
        
        # If there are valid conditions (not NaN), return sum (0 if all False)
        # If all conditions are NaN, keep NaN
        if valid_condition_count > 0:
            result[i] = window_sum
        # else: keep NaN (all conditions in window are NaN)
    
    return result


@njit(cache=True)
def _high_day_core(x: np.ndarray, n: int) -> np.ndarray:
    """Numba-accelerated core for high_day."""
    n_len = len(x)
    result = np.full(n_len, np.nan)
    
    for i in range(n - 1, n_len):
        window_max = -np.inf
        max_idx_in_window = -1
        valid_count = 0
        
        # First pass: find max value and its index
        for j in range(i - n + 1, i + 1):
            if not np.isnan(x[j]):
                valid_count += 1
                if x[j] > window_max:
                    window_max = x[j]
                    max_idx_in_window = j
        
        if valid_count == 0:
            result[i] = np.nan
        else:
            # Convert to days since high
            # Position in window: (i - n + 1) to i
            # Days since = i - max_idx_in_window
            result[i] = float(i - max_idx_in_window)
    
    return result


@njit(cache=True)
def _low_day_core(x: np.ndarray, n: int) -> np.ndarray:
    """Numba-accelerated core for low_day."""
    n_len = len(x)
    result = np.full(n_len, np.nan)
    
    for i in range(n - 1, n_len):
        window_min = np.inf
        min_idx_in_window = -1
        valid_count = 0
        
        # First pass: find min value and its index
        for j in range(i - n + 1, i + 1):
            if not np.isnan(x[j]):
                valid_count += 1
                if x[j] < window_min:
                    window_min = x[j]
                    min_idx_in_window = j
        
        if valid_count == 0:
            result[i] = np.nan
        else:
            # Convert to days since low
            # Days since = i - min_idx_in_window
            result[i] = float(i - min_idx_in_window)
    
    return result


# =============================================================================
# Conditional and special operators
# =============================================================================

def sum_if(x: np.ndarray, n: int, condition: np.ndarray) -> np.ndarray:
    """
    Rolling sum of x where condition is True.
    
    Formula: SUMIF(A, n, condition)
             = sum(A[t-n+1:t+1] where condition[t-n+1:t+1] is True)
    
    Parameters
    ----------
    x : np.ndarray
        Values to sum
    n : int
        Window size
    condition : np.ndarray
        Boolean array (same length as x)
        
    Returns
    -------
    np.ndarray
        Rolling conditional sum
        
    Examples
    --------
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> condition = np.array([True, False, True, True, False])
    >>> sum_if(x, 3, condition)
    array([nan, nan, 4., 7., 7.])
    #              (1+3) (2+3+4) (3+4)
    
    Notes
    -----
    - Only sums x values where corresponding condition is True
    - NaN in x is excluded
    - False or NaN in condition → exclude that value
    - If all conditions are False in window, returns 0 (not NaN)
    - This function is accelerated using Numba JIT compilation
    """
    x = np.asarray(x, dtype=float)
    condition = np.asarray(condition, dtype=float)
    n = int(n)
    
    if n <= 0:
        raise ValueError("n must be positive")
    
    if len(x) != len(condition):
        raise ValueError("x and condition must have the same length")
    
    return _sum_if_core(x, n, condition)


def filter_array(x: np.ndarray, condition: np.ndarray) -> np.ndarray:
    """
    Filter array to keep only elements where condition is True.
    
    Formula: FILTER(A, condition) = A[condition]
    
    Parameters
    ----------
    x : np.ndarray
        Input array
    condition : np.ndarray
        Boolean array (same length as x)
        
    Returns
    -------
    np.ndarray
        Filtered array (length <= original length)
        
    Examples
    --------
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> condition = np.array([True, False, True, False, True])
    >>> filter_array(x, condition)
    array([1., 3., 5.])
    
    Notes
    -----
    - Returns variable-length array
    - NaN in condition treated as False
    - Used in formulas like alpha_149
    - This is a simple vectorized operation (no JIT needed)
    """
    x = np.asarray(x, dtype=float)
    condition = np.asarray(condition)
    
    if len(x) != len(condition):
        raise ValueError("x and condition must have the same length")
    
    # Handle NaN in condition by treating as False
    # Create a mask where True means keep the value
    if condition.dtype == bool:
        mask = condition
    else:
        # For non-boolean arrays, treat non-zero, non-NaN as True
        mask = (~np.isnan(condition)) & (condition != 0)
    
    return x[mask]


def high_day(x: np.ndarray, n: int) -> np.ndarray:
    """
    Number of days since the highest value in past n periods.
    
    Formula: HIGHDAY(A, n)
             = index of max in window, counting back from current
             = 0 if max is at current position
             = n-1 if max is n periods ago
    
    Parameters
    ----------
    x : np.ndarray
        Input array
    n : int
        Window size
        
    Returns
    -------
    np.ndarray
        Days since high in each window
        
    Examples
    --------
    >>> high_day(np.array([1, 3, 2, 5, 4]), 3)
    array([nan, nan, 1., 0., 1.])
    #              max@pos1  max@current  max@pos3 (1 ago)
    
    Notes
    -----
    - Returns 0 when current value is the max
    - If multiple maxes, uses first occurrence
    - NaN values are excluded when finding max
    - This function is accelerated using Numba JIT compilation
    """
    x = np.asarray(x, dtype=float)
    n = int(n)
    
    if n <= 0:
        raise ValueError("n must be positive")
    
    return _high_day_core(x, n)


def low_day(x: np.ndarray, n: int) -> np.ndarray:
    """
    Number of days since the lowest value in past n periods.
    
    Formula: LOWDAY(A, n)
             = index of min in window, counting back from current
             = 0 if min is at current position
             = n-1 if min is n periods ago
    
    Parameters
    ----------
    x : np.ndarray
        Input array
    n : int
        Window size
        
    Returns
    -------
    np.ndarray
        Days since low in each window
        
    Examples
    --------
    >>> low_day(np.array([5, 3, 4, 1, 2]), 3)
    array([nan, nan, 1., 0., 1.])
    #              min@pos1  min@current  min@pos3 (1 ago)
    
    Notes
    -----
    - Returns 0 when current value is the min
    - If multiple mins, uses first occurrence
    - NaN values are excluded when finding min
    - This function is accelerated using Numba JIT compilation
    """
    x = np.asarray(x, dtype=float)
    n = int(n)
    
    if n <= 0:
        raise ValueError("n must be positive")
    
    return _low_day_core(x, n)


def sequence(n: int) -> np.ndarray:
    """
    Generate sequence from 1 to n.
    
    Formula: SEQUENCE(n) = [1, 2, 3, ..., n]
    
    Parameters
    ----------
    n : int
        Length of sequence
        
    Returns
    -------
    np.ndarray
        Array [1, 2, ..., n]
        
    Examples
    --------
    >>> sequence(5)
    array([1., 2., 3., 4., 5.])
    
    Notes
    -----
    - Used in formulas like REGBETA(MEAN(CLOSE,6), SEQUENCE(6))
    - Returns float array for consistency
    - Raises ValueError if n <= 0
    - This is a simple vectorized operation (no JIT needed)
    """
    n = int(n)
    
    if n <= 0:
        raise ValueError("n must be positive")
    
    return np.arange(1, n + 1, dtype=float)


# =============================================================================
# Derived field operators for OHLCV data
# =============================================================================

def compute_ret(close: np.ndarray) -> np.ndarray:
    """
    Compute daily returns.
    
    Formula: RET = CLOSE[t]/CLOSE[t-1] - 1
    
    Parameters
    ----------
    close : np.ndarray
        Close prices
        
    Returns
    -------
    np.ndarray
        Daily returns, first value is NaN
        
    Examples
    --------
    >>> compute_ret(np.array([100, 102, 99, 105]))
    array([nan, 0.02, -0.0294, 0.0606])
    
    Notes
    -----
    - First value is NaN because there is no previous close
    - Uses delay() internally which handles NaN propagation
    - This is a simple vectorized operation (no JIT needed)
    """
    close = np.asarray(close, dtype=float)
    return close / delay(close, 1) - 1


def compute_dtm(open_price: np.ndarray, high: np.ndarray) -> np.ndarray:
    """
    Compute DTM (directional movement indicator).
    
    Formula: DTM = (OPEN <= DELAY(OPEN,1) ? 
                    0 : 
                    MAX((HIGH-OPEN), (OPEN-DELAY(OPEN,1))))
    
    Parameters
    ----------
    open_price : np.ndarray
        Open prices
    high : np.ndarray
        High prices
        
    Returns
    -------
    np.ndarray
        DTM values
        
    Examples
    --------
    >>> open_p = np.array([10, 11, 10, 12])
    >>> high = np.array([12, 13, 11, 14])
    >>> compute_dtm(open_p, high)
    # DTM[1] = MAX(13-11, 11-10) = MAX(2, 1) = 2 (OPEN > DELAY(OPEN,1))
    
    Notes
    -----
    - Returns 0 when open <= previous open (no upward movement)
    - Otherwise returns the maximum of (high-open) and (open-prev_open)
    - Used in directional movement indicators
    - This is a simple vectorized operation (no JIT needed)
    """
    open_price = np.asarray(open_price, dtype=float)
    high = np.asarray(high, dtype=float)
    
    prev_open = delay(open_price, 1)
    
    # Condition: OPEN <= DELAY(OPEN, 1)
    condition = open_price <= prev_open
    
    # If condition is True, return 0, else return max
    dtm_value = np.maximum(high - open_price, open_price - prev_open)
    
    return np.where(condition, 0, dtm_value)


def compute_dbm(open_price: np.ndarray, low: np.ndarray) -> np.ndarray:
    """
    Compute DBM (directional movement indicator).
    
    Formula: DBM = (OPEN >= DELAY(OPEN,1) ? 
                    0 : 
                    MAX((OPEN-LOW), (OPEN-DELAY(OPEN,1))))
    
    Parameters
    ----------
    open_price : np.ndarray
        Open prices
    low : np.ndarray
        Low prices
        
    Returns
    -------
    np.ndarray
        DBM values
        
    Examples
    --------
    >>> open_p = np.array([10, 9, 11, 8])
    >>> low = np.array([9, 8, 10, 7])
    >>> compute_dbm(open_p, low)
    # DBM[1] = MAX(9-8, 9-10) = MAX(1, -1) = 1 (OPEN < DELAY(OPEN,1))
    
    Notes
    -----
    - Returns 0 when open >= previous open (no downward movement)
    - Otherwise returns the maximum of (open-low) and (open-prev_open)
    - Used in directional movement indicators
    - This is a simple vectorized operation (no JIT needed)
    """
    open_price = np.asarray(open_price, dtype=float)
    low = np.asarray(low, dtype=float)
    
    prev_open = delay(open_price, 1)
    
    # Condition: OPEN >= DELAY(OPEN, 1)
    condition = open_price >= prev_open
    
    # If condition is True, return 0, else return max
    dbm_value = np.maximum(open_price - low, open_price - prev_open)
    
    return np.where(condition, 0, dbm_value)


def compute_tr(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """
    Compute True Range.
    
    Formula: TR = MAX(
                    MAX(HIGH-LOW, ABS(HIGH-DELAY(CLOSE,1))),
                    ABS(LOW-DELAY(CLOSE,1))
                  )
    
    Parameters
    ----------
    high : np.ndarray
        High prices
    low : np.ndarray
        Low prices
    close : np.ndarray
        Close prices
        
    Returns
    -------
    np.ndarray
        True range values
        
    Examples
    --------
    >>> high = np.array([12, 13, 11, 14])
    >>> low = np.array([10, 11, 9, 12])
    >>> close = np.array([11, 12, 10, 13])
    >>> compute_tr(high, low, close)
    
    Notes
    -----
    True Range is used in ATR (Average True Range) calculations.
    It accounts for gaps between periods by comparing current high/low
    to previous close. The three components are:
    1. Current high - current low
    2. Absolute value of (current high - previous close)
    3. Absolute value of (current low - previous close)
    - This is a simple vectorized operation (no JIT needed)
    """
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    close = np.asarray(close, dtype=float)
    
    prev_close = delay(close, 1)
    
    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)
    
    return np.maximum(np.maximum(tr1, tr2), tr3)


def compute_hd(high: np.ndarray) -> np.ndarray:
    """
    Compute HD (high difference).
    
    Formula: HD = HIGH[t] - HIGH[t-1]
    
    Parameters
    ----------
    high : np.ndarray
        High prices
        
    Returns
    -------
    np.ndarray
        Difference in high prices
        
    Examples
    --------
    >>> compute_hd(np.array([10, 12, 11, 13, 12]))
    array([nan, 2., -1., 2., -1.])
    
    Notes
    -----
    - First value is NaN because there is no previous high
    - Equivalent to delta(high, 1)
    - Positive values indicate higher highs
    - Negative values indicate lower highs
    - This is a simple vectorized operation (no JIT needed)
    """
    high = np.asarray(high, dtype=float)
    return delta(high, 1)


def compute_ld(low: np.ndarray) -> np.ndarray:
    """
    Compute LD (low difference, inverted).
    
    Formula: LD = LOW[t-1] - LOW[t]
    
    Parameters
    ----------
    low : np.ndarray
        Low prices
        
    Returns
    -------
    np.ndarray
        Inverted difference in low prices
        
    Examples
    --------
    >>> compute_ld(np.array([10, 8, 9, 7, 8]))
    array([nan, 2., -1., 2., -1.])
    
    Notes
    -----
    - First value is NaN because there is no previous low
    - This is the inverse of delta(low, 1): LD = -delta(low, 1)
    - Positive values indicate lower lows (previous low > current low)
    - Negative values indicate higher lows (previous low < current low)
    - This is a simple vectorized operation (no JIT needed)
    """
    low = np.asarray(low, dtype=float)
    return delay(low, 1) - low

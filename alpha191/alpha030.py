"""
Alpha030 factor implementation.

Formula:
    alpha_030 = WMA((REGRESI(CLOSE/DELAY(CLOSE)-1, MKT, SMB, HML, 60))^2, 20)
"""

import numpy as np
import pandas as pd
from numba import njit
from .operators import wma, delay
from .utils import run_alpha_factor


@njit
def _rolling_ols_residuals_multivariate_core(y: np.ndarray, x: np.ndarray, window: int) -> np.ndarray:
    """
    Numba-accelerated rolling OLS residuals for multiple independent variables.

    Parameters
    ----------
    y : np.ndarray
        Dependent variable (1D array of shape N)
    x : np.ndarray
        Independent variables (2D array of shape N x K)
    window : int
        Rolling window size

    Returns
    -------
    np.ndarray
        Array of residuals corresponding to the last point in each window.
        First (window-1) points are NaN.
    """
    n_len = len(y)
    n_factors = x.shape[1]
    result = np.full(n_len, np.nan)

    if window > n_len:
        return result

    # Pre-allocate X matrix for the window: window rows, n_factors + 1 columns (for intercept)
    X_window = np.empty((window, n_factors + 1))
    X_window[:, 0] = 1.0  # Intercept column is always 1

    for i in range(window - 1, n_len):
        # Extract Y window
        y_window = y[i - window + 1 : i + 1]

        # Check for NaNs in Y window
        if np.isnan(y_window).any():
            continue

        # Extract X window and fill the pre-allocated matrix
        x_slice = x[i - window + 1 : i + 1, :]

        # Check for NaNs in X window
        has_nan = False
        for r in range(window):
            for c in range(n_factors):
                val = x_slice[r, c]
                if np.isnan(val):
                    has_nan = True
                    break
            if has_nan:
                break

        if has_nan:
            continue

        # Copy x data into X_window (columns 1 onwards)
        X_window[:, 1:] = x_slice

        # Solve OLS: coeffs = (X'X)^-1 X'Y
        # Using lstsq for stability
        # coeffs, residuals, rank, s = np.linalg.lstsq(X_window, y_window)
        # Numba's lstsq returns a tuple/struct-like object.
        # We assume recent numba/numpy behavior where it returns 4 values.
        # However, to be safe and efficient in Numba, we can rely on np.linalg.lstsq

        try:
            coeffs, _, _, _ = np.linalg.lstsq(X_window, y_window)
        except np.linalg.LinAlgError:
            # Handle singular matrix or other errors
            continue

        # Calculate residual for the current point (index i)
        # Current X is the last row of X_window
        current_X = X_window[-1, :]
        current_y = y_window[-1]

        # Predicted y
        y_pred = 0.0
        for k in range(n_factors + 1):
            y_pred += current_X[k] * coeffs[k]

        result[i] = current_y - y_pred

    return result


def alpha_030(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha030 factor.

    Formula:
        alpha_030 = WMA((REGRESI(CLOSE/DELAY(CLOSE)-1, MKT, SMB, HML, 60))^2, 20)

    Requirements:
        The input DataFrame must contain 'mkt', 'smb', 'hml' columns in addition to OHLC.
    """
    # Ensure we have required columns
    required_cols = ['close']
    factor_cols = ['mkt', 'smb', 'hml']

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in DataFrame")

    # Check for factor columns (case-insensitive check might be nice, but strict for now)
    # We allow upper or lower case for flexibility, converting to lower for internal use
    df_cols_lower = {c.lower(): c for c in df.columns}

    missing_factors = []
    x_data_cols = []

    for f in factor_cols:
        if f in df_cols_lower:
            x_data_cols.append(df_cols_lower[f])
        else:
            missing_factors.append(f)

    if missing_factors:
        raise ValueError(
            f"Alpha030 requires Fama-French factors {factor_cols}. "
            f"Missing columns: {missing_factors}. "
            "Please merge these factors into the DataFrame before calculating this alpha."
        )

    # Get date index
    if 'date' in df.columns:
        index = pd.to_datetime(df['date'])
    else:
        index = df.index

    # Step 1: Calculate Returns: CLOSE / DELAY(CLOSE, 1) - 1
    close = df['close'].values.astype(float)
    # Use delay from operators to handle shift consistently
    prev_close = delay(close, 1)
    returns = np.where(prev_close == 0, np.nan, close / prev_close - 1)

    # Step 2: Prepare independent variables matrix
    # Extract MKT, SMB, HML as numpy array (N x 3)
    factors_data = df[x_data_cols].values.astype(float)

    # Step 3: Rolling Regression Residuals (Window 60)
    residuals = _rolling_ols_residuals_multivariate_core(returns, factors_data, window=60)

    # Step 4: Square residuals
    residuals_sq = residuals ** 2

    # Step 5: WMA (Window 20)
    alpha_values = wma(residuals_sq, n=20)

    return pd.Series(alpha_values, index=index, name='alpha_030')


def alpha030(
    code: str,
    benchmark: str = 'zz800',
    end_date: str = "2026-01-23",
    lookback: int = 350
) -> float:
    """
    Compute Alpha030 factor value for a stock at a specific date.

    Note: This requires the underlying data source to provide 'mkt', 'smb', 'hml' columns.
    If using standard load_stock_csv, this will likely fail unless data is augmented.
    """
    return run_alpha_factor(alpha_030, code, benchmark, end_date, lookback)

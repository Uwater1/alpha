"""
Alpha021 factor implementation.

Formula:
    alpha_021 = REGBETA(MEAN(CLOSE,6), SEQUENCE(6))
"""

import numpy as np
import pandas as pd
from .operators import ts_mean, regression_beta, sequence
from .utils import run_alpha_factor


def alpha_021(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha021 factor.

    Formula:
        alpha_021 = REGBETA(MEAN(CLOSE,6), SEQUENCE(6))
    """
    # Ensure we have required columns
    required_cols = ['close']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in DataFrame")

    if 'date' in df.columns:
        index = pd.to_datetime(df['date'])
    else:
        index = df.index

    close = df['close'].values

    # Step 1: Compute MEAN(CLOSE, 6)
    mean_close = ts_mean(close, 6)

    # Step 2: Generate sequence for regression (matching close length)
    seq = np.arange(len(mean_close), dtype=float)

    # Step 3: Compute REGBETA(MEAN(CLOSE,6), SEQUENCE(6))
    # In these formulas, SEQUENCE(n) as the second argument to REGBETA(A, B, n)
    # usually means the time index for the rolling regression.
    alpha_values = regression_beta(mean_close, seq, 6)

    return pd.Series(alpha_values, index=index, name='alpha_021')


def alpha021(
    code: str,
    benchmark: str = 'zz800',
    end_date: str = "2026-01-23",
    lookback: int = 350
) -> float:
    """
    Compute Alpha021 factor value for a stock at a specific date.
    """
    return run_alpha_factor(alpha_021, code, benchmark, end_date, lookback)

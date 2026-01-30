"""
Alpha043 factor implementation.

Formula:
    alpha_043 = SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:(CLOSE<DELAY(CLOSE,1)?-VOLUME:0)),6)
"""

import numpy as np
import pandas as pd
from .operators import delay, ts_sum
from .utils import run_alpha_factor


def alpha_043(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha043 factor.

    Formula:
        alpha_043 = SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:(CLOSE<DELAY(CLOSE,1)?-VOLUME:0)),6)
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

    # Step 1: Compute DELAY(CLOSE, 1)
    delayed_close = delay(close, 1)

    # Step 2: Compute the conditional expression
    # (CLOSE > DELAY(CLOSE, 1) ? VOLUME : (CLOSE < DELAY(CLOSE, 1) ? -VOLUME : 0))
    # Using np.where for nested conditionals
    condition1 = close > delayed_close
    condition2 = close < delayed_close
    
    # Where condition1 is True, use volume
    # Where condition2 is True, use -volume
    # Where both are False (i.e., equal), use 0
    signed_volume = np.where(condition1, volume,
                            np.where(condition2, -volume, 0.0))
    
    # Handle NaNs from delayed_close
    signed_volume[np.isnan(delayed_close)] = np.nan

    # Step 3: Compute SUM over 6 days
    alpha_values = ts_sum(signed_volume, 6)

    return pd.Series(alpha_values, index=index, name='alpha_043')


def alpha043(
    code: str,
    benchmark: str = 'zz800',
    end_date: str = "2026-01-23",
    lookback: int = 350
) -> float:
    """
    Compute Alpha043 factor value for a stock at a specific date.
    """
    return run_alpha_factor(alpha_043, code, benchmark, end_date, lookback)
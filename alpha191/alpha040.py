"""
Alpha040 factor implementation.

Formula:
    alpha_040 = SUM((CLOSE > DELAY(CLOSE, 1) ? VOLUME : 0), 26) / 
                SUM((CLOSE <= DELAY(CLOSE, 1) ? VOLUME : 0), 26) * 100
"""

import numpy as np
import pandas as pd
from .operators import delay, ts_sum
from .utils import run_alpha_factor


def alpha_040(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha040 factor.

    Formula:
        alpha_040 = SUM((CLOSE > DELAY(CLOSE, 1) ? VOLUME : 0), 26) / 
                    SUM((CLOSE <= DELAY(CLOSE, 1) ? VOLUME : 0), 26) * 100
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

    # Step 2: Compute components for numerators and denominators
    # (CLOSE > DELAY(CLOSE, 1) ? VOLUME : 0)
    up_vol = np.where(close > delayed_close, volume, 0.0)
    # Handle NaNs from delayed_close
    up_vol[np.isnan(delayed_close)] = np.nan

    # (CLOSE <= DELAY(CLOSE, 1) ? VOLUME : 0)
    down_vol = np.where(close <= delayed_close, volume, 0.0)
    # Handle NaNs from delayed_close
    down_vol[np.isnan(delayed_close)] = np.nan

    # Step 3: Compute sums over 26 days
    sum_up = ts_sum(up_vol, 26)
    sum_down = ts_sum(down_vol, 26)

    # Step 4: Final result with protection against division by zero
    denom = sum_down
    # In some contexts, we might want to handle denom=0 specially.
    # Following Instruction 4:
    denom_protected = denom.copy()
    denom_protected[denom_protected == 0] = np.nan
    
    alpha_values = (sum_up / denom_protected) * 100

    return pd.Series(alpha_values, index=index, name='alpha_040')


def alpha040(
    code: str,
    benchmark: str = 'zz800',
    end_date: str = "2026-01-23",
    lookback: int = 350
) -> float:
    """
    Compute Alpha040 factor value for a stock at a specific date.
    """
    return run_alpha_factor(alpha_040, code, benchmark, end_date, lookback)

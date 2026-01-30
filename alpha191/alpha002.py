import numpy as np
import pandas as pd
from .operators import delta
from .utils import run_alpha_factor


def alpha_002(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha002 factor.
    Formula: -1 * delta((((close-low)-(high-close))/((high-low))), 1)
    """
    # Ensure we have required columns
    required_cols = ['high', 'low', 'close']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in DataFrame")

    if 'date' in df.columns:
        index = pd.to_datetime(df['date'])
    else:
        index = df.index

    # Core logic
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values

    # Avoid division by zero
    diff = high - low
    diff[diff == 0] = np.nan
    
    val = ((close - low) - (high - close)) / diff
    result_values = -1 * delta(val, 1)

    return pd.Series(result_values, index=index, name='alpha_002')


def alpha002(
    code: str,
    benchmark: str = 'zz800',
    end_date: str = "2026-01-23",
    lookback: int = 350
) -> float:
    """
    Compute Alpha002 factor value for a stock at a specific date.
    """
    return run_alpha_factor(alpha_002, code, benchmark, end_date, lookback)

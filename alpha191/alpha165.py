import numpy as np
import pandas as pd
from .operators import rolling_cumsum_range, ts_std
from .utils import run_alpha_factor

def alpha_165(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha165 factor.
    Formula: MAX(SUMAC(CLOSE-MEAN(CLOSE,48)))-MIN(SUMAC(CLOSE-MEAN(CLOSE,48)))/STD(CLOSE,48)
    Interpretation: Rescaled Range (R/S) analysis over 48-day window.
    """
    close = df['close'].values
    window = 48

    # Calculate Range of Cumulative Deviations
    range_cum_dev = rolling_cumsum_range(close, window)

    # Calculate Standard Deviation
    std_dev = ts_std(close, window)

    # R/S
    with np.errstate(divide='ignore', invalid='ignore'):
        result = range_cum_dev / std_dev

    return pd.Series(result, index=df.index, name='alpha_165')

def alpha165(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_165, code, benchmark, end_date, lookback)

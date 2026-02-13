import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_176(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha176 factor.
    Formula: CORR(RANK(((CLOSE-TS_MIN(LOW,12))/(TS_MAX(HIGH,12)-TS_MIN(LOW,12)))),RANK(VOLUME),6)
    """
    # Extract values as numpy arrays
    close = df['close'].values
    low = df['low'].values
    high = df['high'].values
    volume = df['volume'].values
    
    # Calculate TS_MIN(LOW,12)
    min_low_12 = ts_min(low, 12)
    
    # Calculate TS_MAX(HIGH,12)
    max_high_12 = ts_max(high, 12)
    
    # Calculate CLOSE-TS_MIN(LOW,12)
    close_min_diff = close - min_low_12
    
    # Calculate TS_MAX(HIGH,12)-TS_MIN(LOW,12)
    range_diff = max_high_12 - min_low_12
    
    # Protect against division by zero
    range_diff[range_diff == 0] = np.nan
    
    # Calculate (CLOSE-TS_MIN(LOW,12))/(TS_MAX(HIGH,12)-TS_MIN(LOW,12))
    ratio = close_min_diff / range_diff
    
    # Calculate RANK(...)
    rank_ratio = rank(ratio)
    
    # Calculate RANK(VOLUME)
    rank_volume = rank(volume)
    
    # Calculate CORR(RANK(...),RANK(VOLUME),6) using Numba-accelerated rolling_corr
    corr_values = rolling_corr(rank_ratio, rank_volume, 6)
    
    return pd.Series(corr_values, index=df.index, name='alpha_176')

def alpha176(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_176, code, benchmark, end_date, lookback)
import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_180(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha180 factor.
    Formula: ((MEAN(VOLUME,20)<VOLUME)?((-1*TSRANK(ABS(DELTA(CLOSE,7)),60))*SIGN(DELTA(CLOSE,7)):(-1*VOLUME/MEAN(VOLUME,20)))
    Note: The formula seems to have a syntax error. Assuming it's a conditional expression.
    """
    # Extract values as numpy arrays
    close = df['close'].values
    volume = df['volume'].values
    
    # Calculate MEAN(VOLUME,20)
    mean_volume_20 = ts_mean(volume, 20)
    
    # Calculate MEAN(VOLUME,20)<VOLUME
    condition = mean_volume_20 < volume
    
    # Calculate DELTA(CLOSE,7)
    delta_close_7 = delta(close, 7)
    
    # Calculate ABS(DELTA(CLOSE,7))
    abs_delta_close_7 = np.abs(delta_close_7)
    
    # Calculate TSRANK(ABS(DELTA(CLOSE,7)),60)
    # Using Numba-optimized ts_rank operator for significantly better performance.
    # We scale the normalized [0, 1] rank back to [1, window_size] to match original behavior.
    window = 60
    valid_count = ts_count(~np.isnan(abs_delta_close_7), window)
    tsrank_values = ts_rank(abs_delta_close_7, window) * (valid_count - 1) + 1
    
    # Calculate SIGN(DELTA(CLOSE,7))
    sign_delta_close_7 = np.sign(delta_close_7)
    
    # Calculate (-1*TSRANK(ABS(DELTA(CLOSE,7)),60))*SIGN(DELTA(CLOSE,7))
    true_branch = (-1 * tsrank_values) * sign_delta_close_7
    
    # Calculate (-1*VOLUME/MEAN(VOLUME,20))
    false_branch = -1 * volume / mean_volume_20
    
    # Calculate final result using conditional
    result = np.where(condition, true_branch, false_branch)
    
    return pd.Series(result, index=df.index, name='alpha_180')

def alpha180(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_180, code, benchmark, end_date, lookback)
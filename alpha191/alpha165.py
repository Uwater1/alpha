import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_165(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha165 factor.
    Formula: MAX(SUMAC(CLOSE-MEAN(CLOSE,48)))-MIN(SUMAC(CLOSE-MEAN(CLOSE,48)))/STD(CLOSE,48)
    Note: This factor uses SUMAC which is a cumulative sum function.
    """
    # Extract values as numpy arrays
    close = df['close'].values
    
    # Calculate MEAN(CLOSE,48)
    mean_close_48 = ts_mean(close, 48)
    
    # Calculate CLOSE-MEAN(CLOSE,48)
    diff = close - mean_close_48
    
    # Calculate SUMAC(CLOSE-MEAN(CLOSE,48)) - cumulative sum
    sumac = np.cumsum(diff)
    
    # Calculate MAX(SUMAC(...))
    max_sumac = ts_max(sumac, len(df))
    
    # Calculate MIN(SUMAC(...))
    min_sumac = ts_min(sumac, len(df))
    
    # Calculate STD(CLOSE,48)
    std_close_48 = ts_std(close, 48)
    
    # Calculate numerator: MAX(SUMAC(...)) - MIN(SUMAC(...))
    numerator = max_sumac - min_sumac
    
    # Calculate final result: numerator / STD(CLOSE,48)
    result = numerator / std_close_48
    
    return pd.Series(result, index=df.index, name='alpha_165')

def alpha165(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_165, code, benchmark, end_date, lookback)
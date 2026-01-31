import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_183(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha183 factor.
    Formula: MAX(SUMAC(CLOSE-MEAN(CLOSE,24)))-MIN(SUMAC(CLOSE-MEAN(CLOSE,24)))/STD(CLOSE,24)
    Note: This factor uses SUMAC which is a cumulative sum function.
    """
    # Extract values as numpy arrays
    close = df['close'].values
    
    # Calculate MEAN(CLOSE,24)
    mean_close_24 = ts_mean(close, 24)
    
    # Calculate CLOSE-MEAN(CLOSE,24)
    diff = close - mean_close_24
    
    # Calculate SUMAC(CLOSE-MEAN(CLOSE,24)) - cumulative sum
    sumac = np.cumsum(diff)
    
    # Calculate MAX(SUMAC(...))
    max_sumac = ts_max(sumac, len(df))
    
    # Calculate MIN(SUMAC(...))
    min_sumac = ts_min(sumac, len(df))
    
    # Calculate STD(CLOSE,24)
    std_close_24 = ts_std(close, 24)
    
    # Calculate numerator: MAX(SUMAC(...)) - MIN(SUMAC(...))
    numerator = max_sumac - min_sumac
    
    # Calculate final result: numerator / STD(CLOSE,24)
    result = numerator / std_close_24
    
    return pd.Series(result, index=df.index, name='alpha_183')

def alpha183(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_183, code, benchmark, end_date, lookback)
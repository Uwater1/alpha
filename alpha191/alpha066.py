import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_066(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha066 factor.
    Formula: (CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6)*100
    """
    # Calculate MEAN(CLOSE, 6)
    mean_close = ts_mean(df['close'], 6)
    
    # Calculate CLOSE - MEAN(CLOSE, 6)
    diff = df['close'] - mean_close
    
    # Handle division by zero
    mean_close[mean_close == 0] = np.nan
    
    # Calculate final result
    result = (diff / mean_close) * 100
    
    return pd.Series(result, index=df.index, name='alpha_066')

def alpha066(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_066, code, benchmark, end_date, lookback)
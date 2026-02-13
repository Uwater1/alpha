import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_153(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha153 factor (inverted).
    Formula: (MEAN(CLOSE,3)+MEAN(CLOSE,6)+MEAN(CLOSE,12)+MEAN(CLOSE,24))/4
    """
    # Extract values as numpy arrays
    close = df['close'].values
    
    # Calculate moving averages
    mean_close_3 = ts_mean(close, 3)
    mean_close_6 = ts_mean(close, 6)
    mean_close_12 = ts_mean(close, 12)
    mean_close_24 = ts_mean(close, 24)
    
    # Calculate final result
    result = -((mean_close_3 + mean_close_6 + mean_close_12 + mean_close_24) / 4)
    
    return pd.Series(result, index=df.index, name='alpha_153')

def alpha153(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_153, code, benchmark, end_date, lookback)
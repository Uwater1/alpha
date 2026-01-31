import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_065(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha065 factor.
    Formula: MEAN(CLOSE,6)/CLOSE
    """
    # Calculate MEAN(CLOSE, 6)
    mean_close = ts_mean(df['close'], 6)
    
    # Calculate final result
    result = mean_close / df['close']
    
    return pd.Series(result, index=df.index, name='alpha_065')

def alpha065(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_065, code, benchmark, end_date, lookback)
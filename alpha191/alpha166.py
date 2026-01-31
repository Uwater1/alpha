import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_166(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha166 factor.
    Formula: -20*(20-1)^1.5*SUM(CLOSE/DELAY(CLOSE,1)-1-MEAN(CLOSE/DELAY(CLOSE,1)-1,20),20)/((20-1)*(20-2)(SUM((CLOSE/DELAY(CLOSE,1),20)^2,20))^1.5)
    """
    # Extract values as numpy arrays
    close = df['close'].values
    
    # Calculate CLOSE/DELAY(CLOSE,1)-1
    delay_close = delay(close, 1)
    close_ratio = close / delay_close - 1
    
    # Calculate MEAN(CLOSE/DELAY(CLOSE,1)-1,20)
    mean_close_ratio = ts_mean(close_ratio, 20)
    
    # Calculate CLOSE/DELAY(CLOSE,1)-1-MEAN(CLOSE/DELAY(CLOSE,1)-1,20)
    diff = close_ratio - mean_close_ratio
    
    # Calculate SUM(...,20)
    sum_diff = ts_sum(diff, 20)
    
    # Calculate (CLOSE/DELAY(CLOSE,1),20)^2
    # Note: There seems to be a typo in the formula, it should be (CLOSE/DELAY(CLOSE,1)-1,20)^2
    squared_ratio = close_ratio ** 2
    
    # Calculate SUM((CLOSE/DELAY(CLOSE,1)-1,20)^2,20)
    sum_squared = ts_sum(squared_ratio, 20)
    
    # Calculate constants
    const1 = -20 * (20 - 1) ** 1.5
    const2 = (20 - 1) * (20 - 2)
    
    # Calculate denominator: const2 * (sum_squared)^1.5
    denominator = const2 * (sum_squared ** 1.5)
    
    # Calculate final result: const1 * sum_diff / denominator
    result = const1 * sum_diff / denominator
    
    return pd.Series(result, index=df.index, name='alpha_166')

def alpha166(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_166, code, benchmark, end_date, lookback)
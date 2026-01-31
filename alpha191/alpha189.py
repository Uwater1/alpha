import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_189(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha189 factor.
    Formula: (SUM(RET-MEAN(RET,6),20)*SUM(RET-MEAN(RET,6),2)/SUM((RET-MEAN(RET,6))^2,20))
    """
    # Extract values as numpy arrays
    close = df['close'].values
    
    # Calculate RET = CLOSE/DELAY(CLOSE,1)-1
    delay_close = delay(close, 1)
    ret = close / delay_close - 1
    
    # Calculate MEAN(RET,6)
    mean_ret_6 = ts_mean(ret, 6)
    
    # Calculate RET-MEAN(RET,6)
    ret_diff = ret - mean_ret_6
    
    # Calculate SUM(RET-MEAN(RET,6),20)
    sum_ret_diff_20 = ts_sum(ret_diff, 20)
    
    # Calculate SUM(RET-MEAN(RET,6),2)
    sum_ret_diff_2 = ts_sum(ret_diff, 2)
    
    # Calculate (RET-MEAN(RET,6))^2
    ret_diff_squared = ret_diff ** 2
    
    # Calculate SUM((RET-MEAN(RET,6))^2,20)
    sum_ret_diff_squared_20 = ts_sum(ret_diff_squared, 20)
    
    # Calculate numerator: sum_ret_diff_20 * sum_ret_diff_2
    numerator = sum_ret_diff_20 * sum_ret_diff_2
    
    # Calculate final result: numerator / sum_ret_diff_squared_20
    result = numerator / sum_ret_diff_squared_20
    
    return pd.Series(result, index=df.index, name='alpha_189')

def alpha189(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_189, code, benchmark, end_date, lookback)
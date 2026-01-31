import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_110(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha110 factor.
    Formula: SUM(MAX(0,HIGH-DELAY(CLOSE,1)),20)/SUM(MAX(0,DELAY(CLOSE,1)-LOW),20)*100
    """
    # Extract values as numpy arrays
    high = df['high'].values
    close = df['close'].values
    low = df['low'].values
    
    # Calculate delay of close
    delay_close = delay(close, 1)
    
    # Calculate maximum of 0 and high minus delay of close
    max_high_delay_close = np.maximum(0, high - delay_close)
    
    # Calculate maximum of 0 and delay of close minus low
    max_delay_close_low = np.maximum(0, delay_close - low)
    
    # Calculate sum of maximum of 0 and high minus delay of close
    sum_max_high_delay_close = ts_sum(max_high_delay_close, 20)
    
    # Calculate sum of maximum of 0 and delay of close minus low
    sum_max_delay_close_low = ts_sum(max_delay_close_low, 20)
    
    # Calculate final result
    result = sum_max_high_delay_close / sum_max_delay_close_low * 100
    
    return pd.Series(result, index=df.index, name='alpha_110')

def alpha110(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_110, code, benchmark, end_date, lookback)
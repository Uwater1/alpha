import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_118(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha118 factor.
    Formula: SUM(HIGH-OPEN,20)/SUM(OPEN-LOW,20)*100
    """
    # Extract values as numpy arrays
    high = df['high'].values
    open_price = df['open'].values
    low = df['low'].values
    
    # Calculate high minus open
    high_minus_open = high - open_price
    
    # Calculate open minus low
    open_minus_low = open_price - low
    
    # Calculate sum of high minus open
    sum_high_minus_open = ts_sum(high_minus_open, 20)
    
    # Calculate sum of open minus low
    sum_open_minus_low = ts_sum(open_minus_low, 20)
    
    # Calculate final result
    # Protect against division by zero
    denom = sum_open_minus_low.copy()
    denom[denom == 0] = np.nan
    result = sum_high_minus_open / denom * 100
    
    return pd.Series(result, index=df.index, name='alpha_118')

def alpha118(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_118, code, benchmark, end_date, lookback)
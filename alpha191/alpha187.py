import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_187(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha187 factor.
    Formula: SUM((OPEN<=DELAY(OPEN,1)?0:MAX((OPEN-LOW),(OPEN-DELAY(OPEN,1)))),20)
    """
    # Extract values as numpy arrays
    open_price = df['open'].values
    low = df['low'].values
    
    # Calculate DELAY(OPEN,1)
    delay_open = delay(open_price, 1)
    
    # Calculate OPEN<=DELAY(OPEN,1)
    condition = open_price <= delay_open
    
    # Calculate OPEN-LOW
    open_low_diff = open_price - low
    
    # Calculate OPEN-DELAY(OPEN,1)
    open_delay_diff = open_price - delay_open
    
    # Calculate MAX((OPEN-LOW),(OPEN-DELAY(OPEN,1)))
    max_diff = np.maximum(open_low_diff, open_delay_diff)
    
    # Calculate (OPEN<=DELAY(OPEN,1)?0:MAX((OPEN-LOW),(OPEN-DELAY(OPEN,1))))
    conditional_value = np.where(condition, 0, max_diff)
    
    # Calculate SUM(...,20)
    result = ts_sum(conditional_value, 20)
    
    return pd.Series(result, index=df.index, name='alpha_187')

def alpha187(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_187, code, benchmark, end_date, lookback)
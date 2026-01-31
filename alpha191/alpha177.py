import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_177(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha177 factor.
    Formula: ((20-HIGH+TS_MIN(HIGH,20))/(20-LOW+TS_MAX(LOW,20)))*100
    """
    # Extract values as numpy arrays
    high = df['high'].values
    low = df['low'].values
    
    # Calculate TS_MIN(HIGH,20)
    min_high_20 = ts_min(high, 20)
    
    # Calculate TS_MAX(LOW,20)
    max_low_20 = ts_max(low, 20)
    
    # Calculate 20-HIGH+TS_MIN(HIGH,20)
    numerator = 20 - high + min_high_20
    
    # Calculate 20-LOW+TS_MAX(LOW,20)
    denominator = 20 - low + max_low_20
    
    # Calculate final result: (numerator/denominator)*100
    result = (numerator / denominator) * 100
    
    return pd.Series(result, index=df.index, name='alpha_177')

def alpha177(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_177, code, benchmark, end_date, lookback)
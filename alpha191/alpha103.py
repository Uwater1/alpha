import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_103(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha103 factor.
    Formula: ((20-LOWDAY(LOW,20))/20)*100
    """
    # Extract values as numpy arrays
    low = df['low'].values
    
    # Calculate LOWDAY
    lowday_result = low_day(low, 20)
    
    # Calculate final result
    result = ((20 - lowday_result) / 20) * 100
    
    return pd.Series(result, index=df.index, name='alpha_103')

def alpha103(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_103, code, benchmark, end_date, lookback)
import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_139(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha139 factor.
    Formula: (-1*CORR(OPEN,VOLUME,10))
    """
    # Extract values as numpy arrays
    open_price = df['open'].values
    volume = df['volume'].values
    
    # Calculate correlation between open and volume
    corr_open_volume = rolling_corr(open_price, volume, 10)
    
    # Calculate final result
    result = -1 * corr_open_volume
    
    return pd.Series(result, index=df.index, name='alpha_139')

def alpha139(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_139, code, benchmark, end_date, lookback)
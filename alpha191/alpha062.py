import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_062(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha062 factor.
    Formula: (-1*CORR(HIGH,RANK(VOLUME),5))
    """
    # Calculate RANK(VOLUME)
    ranked_volume = rank(df['volume'])
    
    # Calculate CORR(HIGH, RANK(VOLUME), 5)
    correlation = rolling_corr(df['high'], ranked_volume, 5)
    
    # Apply negative sign
    result = -1 * correlation
    
    return pd.Series(result, index=df.index, name='alpha_062')

def alpha062(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_062, code, benchmark, end_date, lookback)
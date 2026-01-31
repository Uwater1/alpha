import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_168(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha168 factor.
    Formula: (-1*VOLUME/MEAN(VOLUME,20))
    """
    # Extract values as numpy arrays
    volume = df['volume'].values
    
    # Calculate MEAN(VOLUME,20)
    mean_volume_20 = ts_mean(volume, 20)
    
    # Calculate VOLUME/MEAN(VOLUME,20)
    ratio = volume / mean_volume_20
    
    # Calculate final result: -1 * ratio
    result = -1 * ratio
    
    return pd.Series(result, index=df.index, name='alpha_168')

def alpha168(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_168, code, benchmark, end_date, lookback)
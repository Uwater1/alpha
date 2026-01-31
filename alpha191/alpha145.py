import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_145(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha145 factor.
    Formula: (MEAN(VOLUME,9)-MEAN(VOLUME,26))/MEAN(VOLUME,12)*100
    """
    # Extract values as numpy arrays
    volume = df['volume'].values
    
    # Calculate moving averages
    mean_volume_9 = ts_mean(volume, 9)
    mean_volume_26 = ts_mean(volume, 26)
    mean_volume_12 = ts_mean(volume, 12)
    
    # Calculate the factor
    result = (mean_volume_9 - mean_volume_26) / mean_volume_12 * 100
    
    return pd.Series(result, index=df.index, name='alpha_145')

def alpha145(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_145, code, benchmark, end_date, lookback)
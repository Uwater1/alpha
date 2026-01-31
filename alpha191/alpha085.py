import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_085(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha085 factor.
    Formula: (TSRANK((VOLUME/MEAN(VOLUME,20)),20)*TSRANK((-1*DELTA(CLOSE,7)),8))
    """
    volume = df['volume'].values
    close = df['close'].values
    
    # Calculate VOLUME/MEAN(VOLUME,20)
    mean_volume_20 = ts_mean(volume, 20)
    
    # Handle division by zero
    mean_volume_20[mean_volume_20 == 0] = np.nan
    ratio = volume / mean_volume_20
    
    # Calculate TSRANK((VOLUME/MEAN(VOLUME,20)),20)
    tsrank1 = ts_rank(ratio, 20)
    
    # Calculate -1*DELTA(CLOSE,7)
    delta_close = -1 * delta(close, 7)
    
    # Calculate TSRANK((-1*DELTA(CLOSE,7)),8)
    tsrank2 = ts_rank(delta_close, 8)
    
    # Calculate product
    result = tsrank1 * tsrank2
    
    return pd.Series(result, index=df.index, name='alpha_085')

def alpha085(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_085, code, benchmark, end_date, lookback)
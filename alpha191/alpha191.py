import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_191(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha191 factor.
    Formula: ((RANK((CLOSE-MAX(CLOSE,5)))*RANK(CORR((MEAN(VOLUME,40)),LOW,5)))*-1)
    """
    # Extract values as numpy arrays
    close = df['close'].values
    volume = df['volume'].values
    low = df['low'].values
    
    # Calculate MAX(CLOSE,5)
    max_close_5 = ts_max(close, 5)
    
    # Calculate CLOSE-MAX(CLOSE,5)
    close_diff = close - max_close_5
    
    # Calculate RANK((CLOSE-MAX(CLOSE,5)))
    rank_close_diff = rank(close_diff)
    
    # Calculate MEAN(VOLUME,40)
    mean_volume_40 = ts_mean(volume, 40)
    
    # Calculate CORR((MEAN(VOLUME,40)),LOW,5) using Numba-accelerated rolling_corr
    corr_values = rolling_corr(mean_volume_40, low, 5)
    
    # Calculate RANK(CORR(...))
    rank_corr = rank(corr_values)
    
    # Calculate final result: RANK((CLOSE-MAX(CLOSE,5))) * RANK(CORR(...)) * -1
    result = rank_close_diff * rank_corr * -1
    
    return pd.Series(result, index=df.index, name='alpha_191')

def alpha191(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_191, code, benchmark, end_date, lookback)
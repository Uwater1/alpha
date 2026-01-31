import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_091(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha091 factor.
    Formula: ((RANK((CLOSE-MAX(CLOSE,5)))*RANK(CORR((MEAN(VOLUME,40)),LOW,5)))*-1)
    """
    close = df['close'].values
    volume = df['volume'].values
    low = df['low'].values
    
    # Calculate CLOSE-MAX(CLOSE,5)
    max_close_5 = ts_max(close, 5)
    diff = close - max_close_5
    
    # Calculate RANK(CLOSE-MAX(CLOSE,5))
    rank_diff = rank(diff)
    
    # Calculate MEAN(VOLUME,40)
    mean_volume_40 = ts_mean(volume, 40)
    
    # Calculate CORR((MEAN(VOLUME,40)),LOW,5)
    corr = rolling_corr(mean_volume_40, low, 5)
    
    # Calculate RANK(CORR(...))
    rank_corr = rank(corr)
    
    # Calculate final result
    result = rank_diff * rank_corr * -1
    
    return pd.Series(result, index=df.index, name='alpha_091')

def alpha091(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_091, code, benchmark, end_date, lookback)
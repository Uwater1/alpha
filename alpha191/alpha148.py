import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_148(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha148 factor.
    Formula: ((RANK(CORR((OPEN),SUM(MEAN(VOLUME,60),9),6))<RANK((OPEN-TSMIN(OPEN,14))))*-1)
    """
    # Extract values as numpy arrays
    open_price = df['open'].values
    volume = df['volume'].values
    
    # Calculate MEAN(VOLUME,60)
    mean_volume_60 = ts_mean(volume, 60)
    
    # Calculate SUM(MEAN(VOLUME,60),9)
    sum_mean_volume = ts_sum(mean_volume_60, 9)
    
    # Calculate CORR(OPEN, SUM(MEAN(VOLUME,60),9), 6)
    corr_result = rolling_corr(open_price, sum_mean_volume, 6)
    
    # Calculate OPEN - TSMIN(OPEN,14)
    min_open_14 = ts_min(open_price, 14)
    open_minus_min = open_price - min_open_14
    
    # Calculate ranks
    rank_corr = rank(corr_result)
    rank_open_minus_min = rank(open_minus_min)
    
    # Calculate final result: (RANK(CORR) < RANK(OPEN-TSMIN)) * -1
    result = np.where(rank_corr < rank_open_minus_min, -1, 1)
    
    return pd.Series(result, index=df.index, name='alpha_148')

def alpha148(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_148, code, benchmark, end_date, lookback)
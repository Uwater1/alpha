import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_141(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha141 factor.
    Formula: (RANK(CORR(RANK(HIGH),RANK(MEAN(VOLUME,15)),9))* -1)
    """
    # Extract values as numpy arrays
    high = df['high'].values
    volume = df['volume'].values
    
    # Calculate rank of high
    rank_high = rank(high)
    
    # Calculate mean volume
    mean_volume = ts_mean(volume, 15)
    
    # Calculate rank of mean volume
    rank_mean_volume = rank(mean_volume)
    
    # Calculate correlation between rank of high and rank of mean volume
    corr_rank_high_rank_mean_volume = rolling_corr(rank_high, rank_mean_volume, 9)
    
    # Calculate rank of correlation
    rank_corr = rank(corr_rank_high_rank_mean_volume)
    
    # Calculate final result
    result = rank_corr * -1
    
    return pd.Series(result, index=df.index, name='alpha_141')

def alpha141(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_141, code, benchmark, end_date, lookback)
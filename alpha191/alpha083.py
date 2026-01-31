import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_083(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha083 factor.
    Formula: (-1*RANK(COVIANCE(RANK(HIGH),RANK(VOLUME),5)))
    """
    high = df['high'].values
    volume = df['volume'].values
    
    # Calculate RANK(HIGH)
    rank_high = rank(high)
    
    # Calculate RANK(VOLUME)
    rank_volume = rank(volume)
    
    # Calculate COVIANCE(RANK(HIGH),RANK(VOLUME),5)
    cov = covariance(rank_high, rank_volume, 5)
    
    # Calculate RANK(COVIANCE(...))
    rank_cov = rank(cov)
    
    # Calculate -1 * RANK(...)
    result = -1 * rank_cov
    
    return pd.Series(result, index=df.index, name='alpha_083')

def alpha083(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_083, code, benchmark, end_date, lookback)
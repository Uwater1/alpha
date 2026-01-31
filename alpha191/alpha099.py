import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_099(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha099 factor.
    Formula: (-1*RANK(COVIANCE(RANK(CLOSE),RANK(VOLUME),5)))
    """
    close = df['close'].values
    volume = df['volume'].values
    
    # Calculate RANK(CLOSE)
    rank_close = rank(close)
    
    # Calculate RANK(VOLUME)
    rank_volume = rank(volume)
    
    # Calculate COVIANCE(RANK(CLOSE),RANK(VOLUME),5)
    cov = covariance(rank_close, rank_volume, 5)
    
    # Calculate RANK(COVIANCE(...))
    rank_cov = rank(cov)
    
    # Calculate -1 * RANK(...)
    result = -1 * rank_cov
    
    return pd.Series(result, index=df.index, name='alpha_099')

def alpha099(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_099, code, benchmark, end_date, lookback)
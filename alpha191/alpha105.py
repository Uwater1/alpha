import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_105(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha105 factor.
    Formula: (-1*CORR(RANK(OPEN),RANK(VOLUME),10))
    """
    # Extract values as numpy arrays
    open_price = df['open'].values
    volume = df['volume'].values
    
    # Calculate rank of open price
    rank_open = rank(open_price)
    
    # Calculate rank of volume
    rank_volume = rank(volume)
    
    # Calculate correlation between rank of open price and rank of volume
    corr_rank_open_volume = rolling_corr(rank_open, rank_volume, 10)
    
    # Calculate final result
    result = -1 * corr_rank_open_volume
    
    return pd.Series(result, index=df.index, name='alpha_105')

def alpha105(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_105, code, benchmark, end_date, lookback)
import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_136(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha136 factor.
    Formula: ((-1*RANK(DELTA(RET,3)))*CORR(OPEN,VOLUME,10))
    """
    # Extract values as numpy arrays
    ret = df['ret'].values
    open_price = df['open'].values
    volume = df['volume'].values
    
    # Calculate delta of ret
    delta_ret = delta(ret, 3)
    
    # Calculate rank of delta of ret
    rank_delta_ret = rank(delta_ret)
    
    # Calculate correlation between open and volume
    corr_open_volume = rolling_corr(open_price, volume, 10)
    
    # Calculate final result
    result = (-1 * rank_delta_ret) * corr_open_volume
    
    return pd.Series(result, index=df.index, name='alpha_136')

def alpha136(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_136, code, benchmark, end_date, lookback)
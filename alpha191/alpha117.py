import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_117(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha117 factor.
    Formula: ((TSRANK(VOLUME,32)*(1-TSRANK(((CLOSE+HIGH)-LOW),16)))*(1-TSRANK(RET,32)))
    """
    # Extract values as numpy arrays
    volume = df['volume'].values
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    ret = df['ret'].values
    
    # Calculate TSRANK of volume
    tsrank_volume = ts_rank(volume, 32)
    
    # Calculate TSRANK of close plus high minus low
    tsrank_close_high_low = ts_rank((close + high) - low, 16)
    
    # Calculate TSRANK of ret
    tsrank_ret = ts_rank(ret, 32)
    
    # Calculate final result
    result = tsrank_volume * (1 - tsrank_close_high_low) * (1 - tsrank_ret)
    
    return pd.Series(result, index=df.index, name='alpha_117')

def alpha117(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_117, code, benchmark, end_date, lookback)
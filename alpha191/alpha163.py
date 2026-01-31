import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_163(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha163 factor.
    Formula: RANK(((((-1*RET)_MEAN(VOLUME,20))*VWAP)_(HIGH-CLOSE)))
    """
    # Extract values as numpy arrays
    close = df['close'].values
    volume = df['volume'].values
    vwap = df['vwap'].values
    high = df['high'].values
    
    # Calculate RET
    ret = compute_ret(close)
    
    # Calculate -1*RET
    neg_ret = -1 * ret
    
    # Calculate MEAN(VOLUME,20)
    mean_volume_20 = ts_mean(volume, 20)
    
    # Calculate (-1*RET)_MEAN(VOLUME,20)
    # Using element-wise multiplication
    product1 = neg_ret * mean_volume_20
    
    # Calculate (HIGH-CLOSE)
    high_close_diff = high - close
    
    # Calculate (((-1*RET)_MEAN(VOLUME,20))*VWAP)
    product2 = product1 * vwap
    
    # Calculate ((((-1*RET)_MEAN(VOLUME,20))*VWAP)_(HIGH-CLOSE))
    # Using element-wise multiplication
    final_product = product2 * high_close_diff
    
    # Calculate RANK
    result = rank(final_product)
    
    return pd.Series(result, index=df.index, name='alpha_163')

def alpha163(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_163, code, benchmark, end_date, lookback)
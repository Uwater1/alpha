import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_076(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha076 factor.
    Formula: STD(ABS((CLOSE/DELAY(CLOSE,1)-1))/VOLUME,20)/MEAN(ABS((CLOSE/DELAY(CLOSE,1)-1))/VOLUME,20)
    """
    close = df['close'].values
    volume = df['volume'].values
    
    # Calculate (CLOSE/DELAY(CLOSE,1)-1)
    ret = close / delay(close, 1) - 1
    
    # Calculate ABS((CLOSE/DELAY(CLOSE,1)-1))/VOLUME
    abs_ret = np.abs(ret)
    ratio = abs_ret / volume
    
    # Handle division by zero
    ratio[volume == 0] = np.nan
    
    # Calculate STD and MEAN over 20 periods
    std_20 = ts_std(ratio, 20)
    mean_20 = ts_mean(ratio, 20)
    
    # Handle division by zero
    mean_20[mean_20 == 0] = np.nan
    result = std_20 / mean_20
    
    return pd.Series(result, index=df.index, name='alpha_076')

def alpha076(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_076, code, benchmark, end_date, lookback)
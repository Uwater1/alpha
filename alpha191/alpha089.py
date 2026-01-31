import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_089(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha089 factor.
    Formula: 2*(SMA(CLOSE,13,2)-SMA(CLOSE,27,2)-SMA(SMA(CLOSE,13,2)-SMA(CLOSE,27,2),10,2))
    """
    close = df['close'].values
    
    # Calculate SMA(CLOSE,13,2)
    sma_13_2 = sma(close, 13, 2)
    
    # Calculate SMA(CLOSE,27,2)
    sma_27_2 = sma(close, 27, 2)
    
    # Calculate SMA(CLOSE,13,2)-SMA(CLOSE,27,2)
    diff = sma_13_2 - sma_27_2
    
    # Calculate SMA(SMA(CLOSE,13,2)-SMA(CLOSE,27,2),10,2)
    sma_diff = sma(diff, 10, 2)
    
    # Calculate final result
    result = 2 * (sma_13_2 - sma_27_2 - sma_diff)
    
    return pd.Series(result, index=df.index, name='alpha_089')

def alpha089(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_089, code, benchmark, end_date, lookback)
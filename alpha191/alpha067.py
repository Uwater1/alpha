import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_067(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha067 factor.
    Formula: SMA(MAX(CLOSE-DELAY(CLOSE,1),0),24,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),24,1)*100
    """
    # Calculate DELAY(CLOSE, 1)
    delayed_close = delay(df['close'], 1)
    
    # Calculate CLOSE - DELAY(CLOSE, 1)
    price_diff = df['close'] - delayed_close
    
    # Calculate MAX(CLOSE - DELAY(CLOSE, 1), 0)
    max_diff = np.maximum(price_diff, 0)
    
    # Calculate ABS(CLOSE - DELAY(CLOSE, 1))
    abs_diff = np.abs(price_diff)
    
    # Calculate SMA(MAX(CLOSE - DELAY(CLOSE, 1), 0), 24, 1)
    sma_max = sma(max_diff, 24, 1)
    
    # Calculate SMA(ABS(CLOSE - DELAY(CLOSE, 1)), 24, 1)
    sma_abs = sma(abs_diff, 24, 1)
    
    # Handle division by zero
    sma_abs[sma_abs == 0] = np.nan
    
    # Calculate final result
    result = (sma_max / sma_abs) * 100
    
    return pd.Series(result, index=df.index, name='alpha_067')

def alpha067(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_067, code, benchmark, end_date, lookback)
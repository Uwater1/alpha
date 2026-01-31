import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_079(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha079 factor.
    Formula: SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100
    """
    close = df['close'].values
    
    # Calculate CLOSE-DELAY(CLOSE,1)
    close_diff = close - delay(close, 1)
    
    # Calculate MAX(CLOSE-DELAY(CLOSE,1),0)
    max_diff = np.maximum(close_diff, 0)
    
    # Calculate ABS(CLOSE-DELAY(CLOSE,1))
    abs_diff = np.abs(close_diff)
    
    # Calculate SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)
    sma_max = sma(max_diff, 12, 1)
    
    # Calculate SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)
    sma_abs = sma(abs_diff, 12, 1)
    
    # Handle division by zero
    sma_abs[sma_abs == 0] = np.nan
    result = sma_max / sma_abs * 100
    
    return pd.Series(result, index=df.index, name='alpha_079')

def alpha079(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_079, code, benchmark, end_date, lookback)
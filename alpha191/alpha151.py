import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_151(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha151 factor (inverted).
    Formula: SMA(CLOSE-DELAY(CLOSE,20),20,1)
    """
    # Extract values as numpy arrays
    close = df['close'].values
    
    # Calculate DELAY(CLOSE,20) - CLOSE
    delay_close_20 = delay(close, 20)
    diff = delay_close_20 - close
    
    # Calculate SMA with alpha=1/20 (approximating SMA with alpha parameter)
    # Using exponential moving average with alpha=1/20
    alpha = 1.0 / 20
    result = np.full(len(df), np.nan)
    
    # Initialize first value
    if len(diff) > 0 and not np.isnan(diff[0]):
        result[0] = diff[0]
    
    # Calculate EMA
    for i in range(1, len(diff)):
        if np.isnan(diff[i]):
            result[i] = result[i-1] if not np.isnan(result[i-1]) else np.nan
        elif np.isnan(result[i-1]):
            result[i] = diff[i]
        else:
            result[i] = alpha * diff[i] + (1 - alpha) * result[i-1]
    
    return pd.Series(result, index=df.index, name='alpha_151')

def alpha151(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_151, code, benchmark, end_date, lookback)
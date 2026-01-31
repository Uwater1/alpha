import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_160(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha160 factor.
    Formula: SMA((CLOSE<=DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1)
    """
    # Extract values as numpy arrays
    close = df['close'].values
    
    # Calculate DELAY(CLOSE,1)
    delay_close = delay(close, 1)
    
    # Calculate CLOSE<=DELAY(CLOSE,1)
    condition = close <= delay_close
    
    # Calculate STD(CLOSE,20)
    std_close_20 = ts_std(close, 20)
    
    # Calculate (CLOSE<=DELAY(CLOSE,1)?STD(CLOSE,20):0)
    conditional_std = np.where(condition, std_close_20, 0)
    
    # Calculate SMA with alpha=1/20 (approximating SMA with alpha parameter)
    # Using exponential moving average with alpha=1/20
    alpha = 1.0 / 20
    result = np.full(len(df), np.nan)
    
    # Initialize first value
    if len(conditional_std) > 0 and not np.isnan(conditional_std[0]):
        result[0] = conditional_std[0]
    
    # Calculate EMA
    for i in range(1, len(conditional_std)):
        if np.isnan(conditional_std[i]):
            result[i] = result[i-1] if not np.isnan(result[i-1]) else np.nan
        elif np.isnan(result[i-1]):
            result[i] = conditional_std[i]
        else:
            result[i] = alpha * conditional_std[i] + (1 - alpha) * result[i-1]
    
    return pd.Series(result, index=df.index, name='alpha_160')

def alpha160(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_160, code, benchmark, end_date, lookback)
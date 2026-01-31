import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_158(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha158 factor.
    Formula: ((HIGH-SMA(CLOSE,15,2))-(LOW-SMA(CLOSE,15,2)))/CLOSE
    """
    # Extract values as numpy arrays
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    
    # Calculate SMA(CLOSE,15,2) - EMA with alpha=2/15
    alpha = 2.0 / 15
    sma_close = np.full(len(df), np.nan)
    if len(close) > 0 and not np.isnan(close[0]):
        sma_close[0] = close[0]
    for i in range(1, len(close)):
        if np.isnan(close[i]):
            sma_close[i] = sma_close[i-1] if not np.isnan(sma_close[i-1]) else np.nan
        elif np.isnan(sma_close[i-1]):
            sma_close[i] = close[i]
        else:
            sma_close[i] = alpha * close[i] + (1 - alpha) * sma_close[i-1]
    
    # Calculate (HIGH-SMA(CLOSE,15,2))-(LOW-SMA(CLOSE,15,2))
    numerator = (high - sma_close) - (low - sma_close)
    
    # Calculate final result: numerator / CLOSE
    result = numerator / close
    
    return pd.Series(result, index=df.index, name='alpha_158')

def alpha158(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_158, code, benchmark, end_date, lookback)
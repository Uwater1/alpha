import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_188(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha188 factor.
    Formula: ((HIGH-LOW–SMA((CLOSE-MEAN((HIGH+LOW+CLOSE)/3)),20))/(HIGH+LOW–SMA((CLOSE-MEAN((HIGH+LOW+CLOSE)/3)),20)))
    """
    # Extract values as numpy arrays
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    
    # Calculate (HIGH+LOW+CLOSE)/3
    avg_price = (high + low + close) / 3
    
    # Calculate MEAN((HIGH+LOW+CLOSE)/3)
    mean_avg_price = ts_mean(avg_price, 20)
    
    # Calculate CLOSE-MEAN((HIGH+LOW+CLOSE)/3)
    close_diff = close - mean_avg_price
    
    # Calculate SMA((CLOSE-MEAN((HIGH+LOW+CLOSE)/3)),20)
    # Using exponential moving average with alpha=1/20
    alpha = 1.0 / 20
    sma_close_diff = np.full(len(df), np.nan)
    
    # Initialize first value
    if len(close_diff) > 0 and not np.isnan(close_diff[0]):
        sma_close_diff[0] = close_diff[0]
    
    # Calculate EMA
    for i in range(1, len(close_diff)):
        if np.isnan(close_diff[i]):
            sma_close_diff[i] = sma_close_diff[i-1] if not np.isnan(sma_close_diff[i-1]) else np.nan
        elif np.isnan(sma_close_diff[i-1]):
            sma_close_diff[i] = close_diff[i]
        else:
            sma_close_diff[i] = alpha * close_diff[i] + (1 - alpha) * sma_close_diff[i-1]
    
    # Calculate HIGH-LOW
    high_low_diff = high - low
    
    # Calculate numerator: HIGH-LOW - SMA(...)
    numerator = high_low_diff - sma_close_diff
    
    # Calculate denominator: HIGH+LOW - SMA(...)
    denominator = high + low - sma_close_diff
    
    # Calculate final result: numerator / denominator
    result = numerator / denominator
    
    return pd.Series(result, index=df.index, name='alpha_188')

def alpha188(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_188, code, benchmark, end_date, lookback)
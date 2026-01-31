import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_169(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha169 factor.
    Formula: SMA(MEAN(DELAY(SMA(CLOSE-DELAY(CLOSE,1),9,1),1),12)-MEAN(DELAY(SMA(CLOSE-DELAY(CLOSE,1),9,1),1),26),10,1)
    """
    # Extract values as numpy arrays
    close = df['close'].values
    
    # Calculate CLOSE-DELAY(CLOSE,1)
    delay_close = delay(close, 1)
    close_diff = close - delay_close
    
    # Calculate SMA(CLOSE-DELAY(CLOSE,1),9,1) - EMA with alpha=1/9
    alpha_9 = 1.0 / 9
    sma_9 = np.full(len(df), np.nan)
    if len(close_diff) > 0 and not np.isnan(close_diff[0]):
        sma_9[0] = close_diff[0]
    for i in range(1, len(close_diff)):
        if np.isnan(close_diff[i]):
            sma_9[i] = sma_9[i-1] if not np.isnan(sma_9[i-1]) else np.nan
        elif np.isnan(sma_9[i-1]):
            sma_9[i] = close_diff[i]
        else:
            sma_9[i] = alpha_9 * close_diff[i] + (1 - alpha_9) * sma_9[i-1]
    
    # Calculate DELAY(SMA(CLOSE-DELAY(CLOSE,1),9,1),1)
    delay_sma_9 = delay(sma_9, 1)
    
    # Calculate MEAN(DELAY(SMA(CLOSE-DELAY(CLOSE,1),9,1),1),12)
    mean_12 = ts_mean(delay_sma_9, 12)
    
    # Calculate MEAN(DELAY(SMA(CLOSE-DELAY(CLOSE,1),9,1),1),26)
    mean_26 = ts_mean(delay_sma_9, 26)
    
    # Calculate difference
    diff = mean_12 - mean_26
    
    # Calculate SMA with alpha=1/10 (approximating SMA with alpha parameter)
    # Using exponential moving average with alpha=1/10
    alpha_final = 1.0 / 10
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
            result[i] = alpha_final * diff[i] + (1 - alpha_final) * result[i-1]
    
    return pd.Series(result, index=df.index, name='alpha_169')

def alpha169(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_169, code, benchmark, end_date, lookback)
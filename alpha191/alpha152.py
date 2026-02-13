import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_152(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha152 factor (inverted).
    Formula: SMA(MEAN(DELAY(SMA(DELAY(CLOSE/DELAY(CLOSE,9),1),9,1),1),12)-MEAN(DELAY(SMA(DELAY(CLOSE/DELAY(CLOSE,9),1),9,1),1),26),9,1)
    """
    # Extract values as numpy arrays
    close = df['close'].values
    
    # Calculate CLOSE/DELAY(CLOSE,9)
    delay_close_9 = delay(close, 9)
    close_ratio = close / delay_close_9
    
    # Calculate DELAY(CLOSE/DELAY(CLOSE,9),1)
    delay_close_ratio = delay(close_ratio, 1)
    
    # Calculate SMA(DELAY(CLOSE/DELAY(CLOSE,9),1),9,1)
    # Using EMA with alpha=1/9
    alpha_9 = 1.0 / 9
    sma_9 = np.full(len(df), np.nan)
    if len(delay_close_ratio) > 0 and not np.isnan(delay_close_ratio[0]):
        sma_9[0] = delay_close_ratio[0]
    for i in range(1, len(delay_close_ratio)):
        if np.isnan(delay_close_ratio[i]):
            sma_9[i] = sma_9[i-1] if not np.isnan(sma_9[i-1]) else np.nan
        elif np.isnan(sma_9[i-1]):
            sma_9[i] = delay_close_ratio[i]
        else:
            sma_9[i] = alpha_9 * delay_close_ratio[i] + (1 - alpha_9) * sma_9[i-1]
    
    # Calculate DELAY(SMA(DELAY(CLOSE/DELAY(CLOSE,9),1),9,1),1)
    delay_sma_9 = delay(sma_9, 1)
    
    # Calculate MEAN(DELAY(SMA(DELAY(CLOSE/DELAY(CLOSE,9),1),9,1),1),12)
    mean_12 = ts_mean(delay_sma_9, 12)
    
    # Calculate MEAN(DELAY(SMA(DELAY(CLOSE/DELAY(CLOSE,9),1),9,1),1),26)
    mean_26 = ts_mean(delay_sma_9, 26)
    
    # Calculate negative difference
    diff = -(mean_12 - mean_26)
    
    # Calculate final SMA with alpha=1/9
    alpha_final = 1.0 / 9
    result = np.full(len(df), np.nan)
    if len(diff) > 0 and not np.isnan(diff[0]):
        result[0] = diff[0]
    for i in range(1, len(diff)):
        if np.isnan(diff[i]):
            result[i] = result[i-1] if not np.isnan(result[i-1]) else np.nan
        elif np.isnan(result[i-1]):
            result[i] = diff[i]
        else:
            result[i] = alpha_final * diff[i] + (1 - alpha_final) * result[i-1]
    
    return pd.Series(result, index=df.index, name='alpha_152')

def alpha152(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_152, code, benchmark, end_date, lookback)
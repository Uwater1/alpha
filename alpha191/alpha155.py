import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_155(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha155 factor.
    Formula: SMA(VOLUME,13,2)-SMA(VOLUME,27,2)-SMA(SMA(VOLUME,13,2)-SMA(VOLUME,27,2),10,2)
    """
    # Extract values as numpy arrays
    volume = df['volume'].values
    
    # Calculate SMA(VOLUME,13,2) - EMA with alpha=2/13
    alpha_13 = 2.0 / 13
    sma_13 = np.full(len(df), np.nan)
    if len(volume) > 0 and not np.isnan(volume[0]):
        sma_13[0] = volume[0]
    for i in range(1, len(volume)):
        if np.isnan(volume[i]):
            sma_13[i] = sma_13[i-1] if not np.isnan(sma_13[i-1]) else np.nan
        elif np.isnan(sma_13[i-1]):
            sma_13[i] = volume[i]
        else:
            sma_13[i] = alpha_13 * volume[i] + (1 - alpha_13) * sma_13[i-1]
    
    # Calculate SMA(VOLUME,27,2) - EMA with alpha=2/27
    alpha_27 = 2.0 / 27
    sma_27 = np.full(len(df), np.nan)
    if len(volume) > 0 and not np.isnan(volume[0]):
        sma_27[0] = volume[0]
    for i in range(1, len(volume)):
        if np.isnan(volume[i]):
            sma_27[i] = sma_27[i-1] if not np.isnan(sma_27[i-1]) else np.nan
        elif np.isnan(sma_27[i-1]):
            sma_27[i] = volume[i]
        else:
            sma_27[i] = alpha_27 * volume[i] + (1 - alpha_27) * sma_27[i-1]
    
    # Calculate difference
    diff = sma_13 - sma_27
    
    # Calculate SMA(diff,10,2) - EMA with alpha=2/10
    alpha_10 = 2.0 / 10
    sma_diff = np.full(len(df), np.nan)
    if len(diff) > 0 and not np.isnan(diff[0]):
        sma_diff[0] = diff[0]
    for i in range(1, len(diff)):
        if np.isnan(diff[i]):
            sma_diff[i] = sma_diff[i-1] if not np.isnan(sma_diff[i-1]) else np.nan
        elif np.isnan(sma_diff[i-1]):
            sma_diff[i] = diff[i]
        else:
            sma_diff[i] = alpha_10 * diff[i] + (1 - alpha_10) * sma_diff[i-1]
    
    # Calculate final result
    result = sma_13 - sma_27 - sma_diff
    
    return pd.Series(result, index=df.index, name='alpha_155')

def alpha155(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_155, code, benchmark, end_date, lookback)
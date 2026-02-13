import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_162(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha162 factor.
    Formula: (SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100-MIN(SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100,12))/(MAX(SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100,12)-MIN(SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100,12))
    """
    # Extract values as numpy arrays
    close = df['close'].values
    
    # Calculate DELAY(CLOSE,1)
    delay_close = delay(close, 1)
    
    # Calculate CLOSE-DELAY(CLOSE,1)
    close_diff = close - delay_close
    
    # Calculate MAX(CLOSE-DELAY(CLOSE,1),0)
    max_close_diff = np.maximum(close_diff, 0)
    
    # Calculate ABS(CLOSE-DELAY(CLOSE,1))
    abs_close_diff = np.abs(close_diff)
    
    # Calculate SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1) - EMA with alpha=1/12
    alpha = 1.0 / 12
    sma_max = np.full(len(df), np.nan)
    if len(max_close_diff) > 0 and not np.isnan(max_close_diff[0]):
        sma_max[0] = max_close_diff[0]
    for i in range(1, len(max_close_diff)):
        if np.isnan(max_close_diff[i]):
            sma_max[i] = sma_max[i-1] if not np.isnan(sma_max[i-1]) else np.nan
        elif np.isnan(sma_max[i-1]):
            sma_max[i] = max_close_diff[i]
        else:
            sma_max[i] = alpha * max_close_diff[i] + (1 - alpha) * sma_max[i-1]
    
    # Calculate SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1) - EMA with alpha=1/12
    sma_abs = np.full(len(df), np.nan)
    if len(abs_close_diff) > 0 and not np.isnan(abs_close_diff[0]):
        sma_abs[0] = abs_close_diff[0]
    for i in range(1, len(abs_close_diff)):
        if np.isnan(abs_close_diff[i]):
            sma_abs[i] = sma_abs[i-1] if not np.isnan(sma_abs[i-1]) else np.nan
        elif np.isnan(sma_abs[i-1]):
            sma_abs[i] = abs_close_diff[i]
        else:
            sma_abs[i] = alpha * abs_close_diff[i] + (1 - alpha) * sma_abs[i-1]
    
    # Calculate SMA(...)*100
    # Protect against division by zero
    sma_abs_safe = sma_abs.copy()
    sma_abs_safe[sma_abs_safe == 0] = np.nan
    sma_ratio = sma_max / sma_abs_safe * 100
    
    # Calculate MIN(SMA(...)*100,12)
    min_sma_ratio = ts_min(sma_ratio, 12)
    
    # Calculate MAX(SMA(...)*100,12)
    max_sma_ratio = ts_max(sma_ratio, 12)
    
    # Calculate numerator: SMA(...)*100 - MIN(...)
    numerator = sma_ratio - min_sma_ratio
    
    # Calculate denominator: MAX(...) - MIN(...)
    denominator = max_sma_ratio - min_sma_ratio
    
    # Protect against division by zero
    denominator[denominator == 0] = np.nan
    
    # Calculate final result: numerator / denominator
    result = numerator / denominator
    
    return pd.Series(result, index=df.index, name='alpha_162')

def alpha162(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_162, code, benchmark, end_date, lookback)
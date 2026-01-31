import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_164(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha164 factor.
    Formula: SMA((((CLOSE>DELAY(CLOSE,1))?1/(CLOSE-DELAY(CLOSE,1)):1)-MIN(((CLOSE>DELAY(CLOSE,1))?1/(CLOSE-DELAY(CLOSE,1)):1),12))/(HIGH-LOW)*100,13,2)
    """
    # Extract values as numpy arrays
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    
    # Calculate DELAY(CLOSE,1)
    delay_close = delay(close, 1)
    
    # Calculate CLOSE>DELAY(CLOSE,1)
    condition = close > delay_close
    
    # Calculate (CLOSE>DELAY(CLOSE,1))?1/(CLOSE-DELAY(CLOSE,1)):1
    close_diff = close - delay_close
    # Protect against division by zero
    close_diff[close_diff == 0] = np.nan
    conditional_value = np.where(condition, 1 / close_diff, 1)
    
    # Calculate MIN(((CLOSE>DELAY(CLOSE,1))?1/(CLOSE-DELAY(CLOSE,1)):1),12)
    min_conditional = ts_min(conditional_value, 12)
    
    # Calculate numerator: conditional_value - min_conditional
    numerator = conditional_value - min_conditional
    
    # Calculate (HIGH-LOW)
    high_low_diff = high - low
    
    # Calculate numerator/(HIGH-LOW)*100
    ratio = numerator / high_low_diff * 100
    
    # Calculate SMA with alpha=2/13 (approximating SMA with alpha parameter)
    # Using exponential moving average with alpha=2/13
    alpha = 2.0 / 13
    result = np.full(len(df), np.nan)
    
    # Initialize first value
    if len(ratio) > 0 and not np.isnan(ratio[0]):
        result[0] = ratio[0]
    
    # Calculate EMA
    for i in range(1, len(ratio)):
        if np.isnan(ratio[i]):
            result[i] = result[i-1] if not np.isnan(result[i-1]) else np.nan
        elif np.isnan(result[i-1]):
            result[i] = ratio[i]
        else:
            result[i] = alpha * ratio[i] + (1 - alpha) * result[i-1]
    
    return pd.Series(result, index=df.index, name='alpha_164')

def alpha164(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_164, code, benchmark, end_date, lookback)
import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_161(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha161 factor.
    Formula: MEAN(MAX(MAX((HIGH-LOW),ABS(DELAY(CLOSE,1)-HIGH)),ABS(DELAY(CLOSE,1)-LOW)),12)
    """
    # Extract values as numpy arrays
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    
    # Calculate DELAY(CLOSE,1)
    delay_close = delay(close, 1)
    
    # Calculate HIGH-LOW
    high_low_diff = high - low
    
    # Calculate ABS(DELAY(CLOSE,1)-HIGH)
    abs_delay_high = np.abs(delay_close - high)
    
    # Calculate ABS(DELAY(CLOSE,1)-LOW)
    abs_delay_low = np.abs(delay_close - low)
    
    # Calculate MAX((HIGH-LOW),ABS(DELAY(CLOSE,1)-HIGH))
    max1 = np.maximum(high_low_diff, abs_delay_high)
    
    # Calculate MAX(MAX((HIGH-LOW),ABS(DELAY(CLOSE,1)-HIGH)),ABS(DELAY(CLOSE,1)-LOW))
    max2 = np.maximum(max1, abs_delay_low)
    
    # Calculate MEAN(...,12)
    result = ts_mean(max2, 12)
    
    return pd.Series(result, index=df.index, name='alpha_161')

def alpha161(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_161, code, benchmark, end_date, lookback)
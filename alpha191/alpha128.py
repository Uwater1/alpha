import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_128(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha128 factor.
    Formula: 100-(100/(1+SUM(((HIGH+LOW+CLOSE)/3>DELAY((HIGH+LOW+CLOSE)/3,1)?(HIGH+LOW+CLOSE)/3*VOLUME:0),14)/SUM(((HIGH+LOW+CLOSE)/3<DELAY((HIGH+LOW+CLOSE)/3,1)?(HIGH+LOW+CLOSE)/3*VOLUME:0),14)))
    """
    # Extract values as numpy arrays
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    volume = df['volume'].values
    
    # Calculate average of high, low, and close
    avg_high_low_close = (high + low + close) / 3
    
    # Calculate delay of average of high, low, and close
    delay_avg_high_low_close = delay(avg_high_low_close, 1)
    
    # Calculate positive volume
    positive_volume = np.where(avg_high_low_close > delay_avg_high_low_close, avg_high_low_close * volume, 0)
    
    # Calculate negative volume
    negative_volume = np.where(avg_high_low_close < delay_avg_high_low_close, avg_high_low_close * volume, 0)
    
    # Calculate sum of positive volume
    sum_positive_volume = ts_sum(positive_volume, 14)
    
    # Calculate sum of negative volume
    sum_negative_volume = ts_sum(negative_volume, 14)
    
    # Protect against division by zero
    denom = sum_negative_volume.copy()
    denom[denom == 0] = np.nan
    
    # Calculate final result
    result = 100 - (100 / (1 + sum_positive_volume / denom))
    
    return pd.Series(result, index=df.index, name='alpha_128')

def alpha128(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_128, code, benchmark, end_date, lookback)
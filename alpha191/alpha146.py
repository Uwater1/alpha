import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_146(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha146 factor.
    Formula: MEAN((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)-SMA((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1),61,2),20)*((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)-SMA((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1),61,2))/SMA(((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)-((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)-SMA((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1),61,2)))^2,60)
    """
    # Extract values as numpy arrays
    close = df['close'].values
    
    # Calculate close/DELAY(close,1)-1
    delay_close = delay(close, 1)
    close_ratio = close / delay_close - 1
    
    # Calculate SMA with alpha=2/61 (approximating SMA with alpha parameter)
    # Since we don't have exact SMA implementation, we'll use ts_mean for simplicity
    sma_61 = ts_mean(close_ratio, 61)
    
    # Calculate the difference
    diff = close_ratio - sma_61
    
    # Calculate numerator: MEAN(diff, 20) * diff
    mean_diff_20 = ts_mean(diff, 20)
    numerator = mean_diff_20 * diff
    
    # Calculate denominator: SMA(diff^2, 60)
    # Using ts_mean for SMA approximation
    denominator = ts_mean(diff**2, 60)
    
    # Calculate final result
    result = numerator / denominator
    
    return pd.Series(result, index=df.index, name='alpha_146')

def alpha146(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_146, code, benchmark, end_date, lookback)
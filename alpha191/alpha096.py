import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_096(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha096 factor.
    Formula: SMA(SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1),3,1)
    """
    close = df['close'].values
    low = df['low'].values
    high = df['high'].values
    
    # Calculate CLOSE-TSMIN(LOW,9)
    min_low_9 = ts_min(low, 9)
    numerator = close - min_low_9
    
    # Calculate TSMAX(HIGH,9)-TSMIN(LOW,9)
    max_high_9 = ts_max(high, 9)
    denominator = max_high_9 - min_low_9
    
    # Handle division by zero
    denominator[denominator == 0] = np.nan
    ratio = numerator / denominator * 100
    
    # Calculate SMA(...,3,1)
    sma1 = sma(ratio, 3, 1)
    
    # Calculate SMA(SMA(...,3,1),3,1)
    result = sma(sma1, 3, 1)
    
    return pd.Series(result, index=df.index, name='alpha_096')

def alpha096(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_096, code, benchmark, end_date, lookback)
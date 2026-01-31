import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_072(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha072 factor.
    Formula: SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))*100,15,1)
    """
    # Calculate SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))*100,15,1)
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    
    max_high_6 = ts_max(high, 6)
    min_low_6 = ts_min(low, 6)
    
    numerator = max_high_6 - close
    denominator = max_high_6 - min_low_6
    
    # Handle division by zero
    denominator[denominator == 0] = np.nan
    ratio = numerator / denominator * 100
    
    result = sma(ratio, 15, 1)
    
    return pd.Series(result, index=df.index, name='alpha_072')

def alpha072(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_072, code, benchmark, end_date, lookback)
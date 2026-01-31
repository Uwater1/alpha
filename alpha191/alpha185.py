import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_185(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha185 factor.
    Formula: RANK((-1*(((LOW-CLOSE)*(OPEN^5)))/((CLOSE-HIGH)*(CLOSE^5))))
    """
    # Extract values as numpy arrays
    low = df['low'].values
    close = df['close'].values
    open_price = df['open'].values
    high = df['high'].values
    
    # Calculate LOW-CLOSE
    low_close_diff = low - close
    
    # Calculate OPEN^5
    open_power_5 = open_price ** 5
    
    # Calculate (LOW-CLOSE)*(OPEN^5)
    numerator_part1 = low_close_diff * open_power_5
    
    # Calculate CLOSE-HIGH
    close_high_diff = close - high
    
    # Calculate CLOSE^5
    close_power_5 = close ** 5
    
    # Calculate (CLOSE-HIGH)*(CLOSE^5)
    denominator_part1 = close_high_diff * close_power_5
    
    # Calculate ((LOW-CLOSE)*(OPEN^5))/((CLOSE-HIGH)*(CLOSE^5))
    # Protect against division by zero
    denom = denominator_part1.copy()
    denom[denom == 0] = np.nan
    ratio = numerator_part1 / denom
    
    # Calculate -1 * ratio
    negative_ratio = -1 * ratio
    
    # Calculate RANK(...)
    result = rank(negative_ratio)
    
    return pd.Series(result, index=df.index, name='alpha_185')

def alpha185(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_185, code, benchmark, end_date, lookback)
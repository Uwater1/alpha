import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_111(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha111 factor.
    Formula: SMA(VOL*((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW),11,2)-SMA(VOL*((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW),4,2)
    """
    # Extract values as numpy arrays
    volume = df['volume'].values
    close = df['close'].values
    low = df['low'].values
    high = df['high'].values
    
    # Calculate numerator
    numerator = (close - low) - (high - close)
    
    # Calculate denominator
    denominator = high - low
    
    # Protect against division by zero
    denominator[denominator == 0] = np.nan
    
    # Calculate ratio
    ratio = numerator / denominator
    
    # Calculate volume times ratio
    volume_ratio = volume * ratio
    
    # Calculate SMA of volume times ratio
    sma_volume_ratio_11 = sma(volume_ratio, 11, 2)
    sma_volume_ratio_4 = sma(volume_ratio, 4, 2)
    
    # Calculate final result
    result = sma_volume_ratio_11 - sma_volume_ratio_4
    
    return pd.Series(result, index=df.index, name='alpha_111')

def alpha111(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_111, code, benchmark, end_date, lookback)
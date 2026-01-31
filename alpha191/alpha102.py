import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_102(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha102 factor.
    Formula: SMA(MAX(VOLUME-DELAY(VOLUME,1),0),6,1)/SMA(ABS(VOLUME-DELAY(VOLUME,1)),6,1)*100
    """
    # Extract values as numpy arrays
    volume = df['volume'].values
    
    # Calculate volume change
    volume_change = volume - delay(volume, 1)
    
    # Calculate positive volume change
    positive_volume_change = np.maximum(volume_change, 0)
    
    # Calculate absolute volume change
    abs_volume_change = np.abs(volume_change)
    
    # Calculate SMA of positive volume change
    sma_positive_volume_change = sma(positive_volume_change, 6, 1)
    
    # Calculate SMA of absolute volume change
    sma_abs_volume_change = sma(abs_volume_change, 6, 1)
    
    # Calculate final result
    result = sma_positive_volume_change / sma_abs_volume_change * 100
    
    return pd.Series(result, index=df.index, name='alpha_102')

def alpha102(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_102, code, benchmark, end_date, lookback)
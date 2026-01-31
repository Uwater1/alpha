import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_109(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha109 factor.
    Formula: SMA(HIGH-LOW,10,2)/SMA(SMA(HIGH-LOW,10,2),10,2)
    """
    # Extract values as numpy arrays
    high = df['high'].values
    low = df['low'].values
    
    # Calculate high minus low
    high_minus_low = high - low
    
    # Calculate SMA of high minus low
    sma_high_minus_low = sma(high_minus_low, 10, 2)
    
    # Calculate SMA of SMA of high minus low
    sma_sma_high_minus_low = sma(sma_high_minus_low, 10, 2)
    
    # Calculate final result
    result = sma_high_minus_low / sma_sma_high_minus_low
    
    return pd.Series(result, index=df.index, name='alpha_109')

def alpha109(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_109, code, benchmark, end_date, lookback)
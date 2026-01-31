import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_173(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha173 factor.
    Formula: 300*(MEAN(CLOSE,6)+MEAN(CLOSE,12)+MEAN(CLOSE,24))/(MA(CLOSE,6)+MA(CLOSE,12)+MA(CLOSE,24))+100
    """
    # Extract values as numpy arrays
    close = df['close'].values
    
    # Calculate MEAN(CLOSE,6)
    mean_6 = ts_mean(close, 6)
    
    # Calculate MEAN(CLOSE,12)
    mean_12 = ts_mean(close, 12)
    
    # Calculate MEAN(CLOSE,24)
    mean_24 = ts_mean(close, 24)
    
    # Calculate MA(CLOSE,6) - Simple Moving Average
    ma_6 = ts_mean(close, 6)
    
    # Calculate MA(CLOSE,12) - Simple Moving Average
    ma_12 = ts_mean(close, 12)
    
    # Calculate MA(CLOSE,24) - Simple Moving Average
    ma_24 = ts_mean(close, 24)
    
    # Calculate numerator: MEAN(CLOSE,6) + MEAN(CLOSE,12) + MEAN(CLOSE,24)
    numerator = mean_6 + mean_12 + mean_24
    
    # Calculate denominator: MA(CLOSE,6) + MA(CLOSE,12) + MA(CLOSE,24)
    denominator = ma_6 + ma_12 + ma_24
    
    # Calculate final result: 300 * numerator / denominator + 100
    result = 300 * numerator / denominator + 100
    
    return pd.Series(result, index=df.index, name='alpha_173')

def alpha173(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_173, code, benchmark, end_date, lookback)
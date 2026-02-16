import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_147(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha147 factor.
    Formula: REGBETA(MEAN(CLOSE,12),SEQUENCE(12))
    """
    # Extract values as numpy arrays
    close = df['close'].values
    
    # Calculate moving average
    mean_close_12 = ts_mean(close, 12)
    
    # Create sequence 1,2,3,...,12
    sequence = np.arange(1, 13)
    
    # Calculate regression beta
    result = np.full(len(df), np.nan)
    
    for i in range(12, len(df)):
        # Get the window of data
        window_data = mean_close_12[i-12:i]
        
        # Skip if any NaN values in the window
        if np.isnan(window_data).any():
            continue
            
        # Calculate regression
        # Using numpy's polyfit to calculate beta (slope)
        coeffs = np.polyfit(sequence, window_data, 1)
        result[i] = coeffs[0]  # The slope (beta)
    
    return pd.Series(result, index=df.index, name='alpha_147')

def alpha147(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_147, code, benchmark, end_date, lookback)
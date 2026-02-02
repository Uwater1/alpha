import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_127(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha127 factor.
    Formula: (MEAN((100*(CLOSE-MAX(CLOSE,12))/(MAX(CLOSE,12)))^2))^(1/2)
    """
    # Extract values as numpy arrays
    close = df['close'].values
    
    # Calculate maximum of close
    max_close = ts_max(close, 12)
    
    # Calculate close minus maximum of close
    close_minus_max_close = close - max_close
    
    # Protect against division by zero
    denom = max_close.copy()
    denom[denom == 0] = np.nan
    
    # Calculate ratio
    ratio = 100 * close_minus_max_close / denom
    
    # Calculate square of ratio
    ratio_squared = ratio ** 2
    
    # Calculate mean of square of ratio
    mean_ratio_squared = ts_mean(ratio_squared, 12)
    
    # Protect against negative values before sqrt (numerical issues)
    mean_ratio_squared = np.where(mean_ratio_squared < 0, np.nan, mean_ratio_squared)
    
    # Calculate final result
    result = mean_ratio_squared ** (1/2)
    
    return pd.Series(result, index=df.index, name='alpha_127')

def alpha127(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_127, code, benchmark, end_date, lookback)
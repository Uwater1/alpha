import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_054(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha054 factor.
    Formula: (-1*RANK((STD(ABS(CLOSE-OPEN))+(CLOSE-OPEN))+CORR(CLOSE,OPEN,10)))
    """
    # Calculate ABS(CLOSE - OPEN)
    abs_diff = np.abs(df['close'] - df['open'])
    
    # Calculate STD(ABS(CLOSE - OPEN)) over 10 periods
    std_abs_diff = ts_std(abs_diff, 10)
    
    # Calculate (CLOSE - OPEN)
    price_diff = df['close'] - df['open']
    
    # Calculate STD(ABS(CLOSE-OPEN)) + (CLOSE-OPEN)
    combined = std_abs_diff + price_diff
    
    # Calculate CORR(CLOSE, OPEN, 10)
    correlation = rolling_corr(df['close'], df['open'], 10)
    
    # Add the two components
    total = combined + correlation
    
    # Apply cross-sectional rank
    ranked = rank(total)
    
    # Apply negative sign
    result = -1 * ranked
    
    return pd.Series(result, index=df.index, name='alpha_054')

def alpha054(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_054, code, benchmark, end_date, lookback)
import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_126(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha126 factor.
    Formula: (CLOSE+HIGH+LOW)/3
    """
    # Extract values as numpy arrays
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    
    # Calculate final result
    result = (close + high + low) / 3
    
    return pd.Series(result, index=df.index, name='alpha_126')

def alpha126(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_126, code, benchmark, end_date, lookback)
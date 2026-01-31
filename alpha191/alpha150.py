import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_150(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha150 factor.
    Formula: (CLOSE+HIGH+LOW)/3*VOLUME
    """
    # Extract values as numpy arrays
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    volume = df['volume'].values
    
    # Calculate (CLOSE+HIGH+LOW)/3
    typical_price = (close + high + low) / 3
    
    # Calculate final result: typical_price * volume
    result = typical_price * volume
    
    return pd.Series(result, index=df.index, name='alpha_150')

def alpha150(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_150, code, benchmark, end_date, lookback)
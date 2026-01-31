import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_135(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha135 factor.
    Formula: -SMA(DELAY(CLOSE/DELAY(CLOSE,20),1),20,1)
    """
    # Extract values as numpy arrays
    close = df['close'].values
    
    # Calculate delay of close
    delay_close = delay(close, 20)
    
    # Calculate ratio
    ratio = close / delay_close
    
    # Calculate delay of ratio
    delay_ratio = delay(ratio, 1)
    
    # Calculate SMA of delay of ratio
    result = -sma(delay_ratio, 20, 1)
    
    return pd.Series(result, index=df.index, name='alpha_135')

def alpha135(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_135, code, benchmark, end_date, lookback)
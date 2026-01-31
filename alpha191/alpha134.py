import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_134(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha134 factor.
    Formula: (CLOSE-DELAY(CLOSE,12))/DELAY(CLOSE,12)*VOLUME
    """
    # Extract values as numpy arrays
    close = df['close'].values
    volume = df['volume'].values
    
    # Calculate delay of close
    delay_close = delay(close, 12)
    
    # Calculate close change
    close_change = close - delay_close
    
    # Calculate final result
    result = close_change / delay_close * volume
    
    return pd.Series(result, index=df.index, name='alpha_134')

def alpha134(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_134, code, benchmark, end_date, lookback)
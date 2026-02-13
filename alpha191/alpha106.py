import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_106(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha106 factor (inverted).
    Formula: DELAY(CLOSE,20)-CLOSE
    """
    # Extract values as numpy arrays
    close = df['close'].values
    
    # Calculate delay of close
    delay_close = delay(close, 20)
    
    # Calculate final result
    result = delay_close - close
    
    return pd.Series(result, index=df.index, name='alpha_106')

def alpha106(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_106, code, benchmark, end_date, lookback)
import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_167(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha167 factor.
    Formula: SUM((CLOSE-DELAY(CLOSE,1)>0?CLOSE-DELAY(CLOSE,1):0),12
    Note: The formula seems to be missing a closing parenthesis, assuming it's SUM(...,12)
    """
    # Extract values as numpy arrays
    close = df['close'].values
    
    # Calculate DELAY(CLOSE,1)
    delay_close = delay(close, 1)
    
    # Calculate CLOSE-DELAY(CLOSE,1)
    close_diff = close - delay_close
    
    # Calculate CLOSE-DELAY(CLOSE,1)>0
    condition = close_diff > 0
    
    # Calculate (CLOSE-DELAY(CLOSE,1)>0?CLOSE-DELAY(CLOSE,1):0)
    conditional_value = np.where(condition, close_diff, 0)
    
    # Calculate SUM(...,12)
    result = ts_sum(conditional_value, 12)
    
    return pd.Series(result, index=df.index, name='alpha_167')

def alpha167(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_167, code, benchmark, end_date, lookback)
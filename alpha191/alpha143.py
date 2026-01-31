import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_143(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha143 factor.
    Formula: CLOSE>DELAY(CLOSE,1)?(CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)*SELF:SELF
    Note: This factor is recursive and uses the previous value of itself (SELF).
    """
    # Extract values as numpy arrays
    close = df['close'].values
    
    # Calculate delta close
    delta_close = delta(close, 1)
    
    # Calculate delay close
    delay_close = delay(close, 1)
    
    # Calculate return
    ret = delta_close / delay_close
    
    # Initialize result array
    result = np.full(len(df), np.nan)
    
    # For the first value, we don't have a previous SELF value, so we can't compute it
    # We'll start from the second value
    for i in range(1, len(df)):
        if np.isnan(ret[i]):
            result[i] = np.nan
        elif close[i] > delay_close[i]:
            # If close > delay_close, use ret * previous_self
            if np.isnan(result[i-1]):
                # If we don't have a previous SELF value, we can't compute this
                result[i] = np.nan
            else:
                result[i] = ret[i] * result[i-1]
        else:
            # If close <= delay_close, use previous_self
            if np.isnan(result[i-1]):
                # If we don't have a previous SELF value, we can't compute this
                result[i] = np.nan
            else:
                result[i] = result[i-1]
    
    return pd.Series(result, index=df.index, name='alpha_143')

def alpha143(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_143, code, benchmark, end_date, lookback)
import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_175(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha175 factor.
    Formula: SUM((CLOSE>DELAY(CLOSE,1)?0:MAX((-CLOSE+DELAY(CLOSE,1)),(-CLOSE+OPEN)))/(CLOSE-DELAY(CLOSE,1))*VOLUME,60)
    """
    # Extract values as numpy arrays
    close = df['close'].values
    open_price = df['open'].values
    volume = df['volume'].values
    
    # Calculate DELAY(CLOSE,1)
    delay_close = delay(close, 1)
    
    # Calculate CLOSE>DELAY(CLOSE,1)
    condition = close > delay_close
    
    # Calculate -CLOSE+DELAY(CLOSE,1)
    close_delay_diff = -close + delay_close
    
    # Calculate -CLOSE+OPEN
    close_open_diff = -close + open_price
    
    # Calculate MAX((-CLOSE+DELAY(CLOSE,1)),(-CLOSE+OPEN))
    max_diff = np.maximum(close_delay_diff, close_open_diff)
    
    # Calculate (CLOSE>DELAY(CLOSE,1)?0:MAX((-CLOSE+DELAY(CLOSE,1)),(-CLOSE+OPEN)))
    conditional_value = np.where(condition, 0, max_diff)
    
    # Calculate CLOSE-DELAY(CLOSE,1)
    close_diff = close - delay_close
    
    # Protect against division by zero
    denom = close_diff.copy()
    denom[denom == 0] = np.nan
    
    # Calculate (conditional_value)/(CLOSE-DELAY(CLOSE,1))*VOLUME
    ratio = conditional_value / denom * volume
    
    # Calculate SUM(...,60)
    result = ts_sum(ratio, 60)
    
    return pd.Series(result, index=df.index, name='alpha_175')

def alpha175(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_175, code, benchmark, end_date, lookback)
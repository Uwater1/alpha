import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_129(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha129 factor (inverted).
    Formula: SUM((CLOSE-DELAY(CLOSE,1)<0?ABS(CLOSE-DELAY(CLOSE,1)):0),12)
    """
    # Extract values as numpy arrays
    close = df['close'].values
    
    # Calculate close change
    close_change = close - delay(close, 1)
    
    # Calculate positive close change
    negative_close_change = np.where(close_change >= 0, np.abs(close_change), 0)
    
    # Calculate sum of negative close change
    result = ts_sum(negative_close_change, 12)
    
    return pd.Series(result, index=df.index, name='alpha_129')

def alpha129(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_129, code, benchmark, end_date, lookback)
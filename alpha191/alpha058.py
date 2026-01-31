import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_058(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha058 factor.
    Formula: COUNT(CLOSE>DELAY(CLOSE,1),20)/20*100
    """
    # Calculate DELAY(CLOSE, 1)
    delayed_close = delay(df['close'], 1)
    
    # Calculate condition: CLOSE > DELAY(CLOSE, 1)
    condition = df['close'] > delayed_close
    
    # Count True values in rolling window of 20
    count_up_days = ts_count(condition, 20)
    
    # Calculate percentage
    result = (count_up_days / 20) * 100
    
    return pd.Series(result, index=df.index, name='alpha_058')

def alpha058(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_058, code, benchmark, end_date, lookback)
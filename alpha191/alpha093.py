import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_093(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha093 factor.
    Formula: SUM((OPEN>=DELAY(OPEN,1)?0:MAX((OPEN-LOW),(OPEN-DELAY(OPEN,1)))),20)
    """
    open_price = df['open'].values
    low = df['low'].values
    
    # Calculate DELAY(OPEN,1)
    delayed_open = delay(open_price, 1)
    
    # Calculate (OPEN>=DELAY(OPEN,1)?0:MAX((OPEN-LOW),(OPEN-DELAY(OPEN,1))))
    condition = open_price >= delayed_open
    term1 = open_price - low
    term2 = open_price - delayed_open
    
    result_open = np.where(condition, 0, np.maximum(term1, term2))
    
    # Calculate SUM(...,20)
    result = ts_sum(result_open, 20)
    
    return pd.Series(result, index=df.index, name='alpha_093')

def alpha093(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_093, code, benchmark, end_date, lookback)
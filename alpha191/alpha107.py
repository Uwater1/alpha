import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_107(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha107 factor.
    Formula: (((-1*RANK((OPEN-DELAY(HIGH,1))))*RANK((OPEN-DELAY(CLOSE,1))))*RANK((OPEN-DELAY(LOW,1))))
    """
    # Extract values as numpy arrays
    open_price = df['open'].values
    high = df['high'].values
    close = df['close'].values
    low = df['low'].values
    
    # Calculate rank of open minus delay of high
    rank_open_minus_delay_high = rank(open_price - delay(high, 1))
    
    # Calculate rank of open minus delay of close
    rank_open_minus_delay_close = rank(open_price - delay(close, 1))
    
    # Calculate rank of open minus delay of low
    rank_open_minus_delay_low = rank(open_price - delay(low, 1))
    
    # Calculate final result
    result = (-1 * rank_open_minus_delay_high) * rank_open_minus_delay_close * rank_open_minus_delay_low
    
    return pd.Series(result, index=df.index, name='alpha_107')

def alpha107(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_107, code, benchmark, end_date, lookback)
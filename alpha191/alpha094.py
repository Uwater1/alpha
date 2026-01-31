import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_094(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha094 factor.
    Formula: SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:(CLOSE<DELAY(CLOSE,1)?-VOLUME:0)),30)
    """
    close = df['close'].values
    volume = df['volume'].values
    
    # Calculate DELAY(CLOSE,1)
    delayed_close = delay(close, 1)
    
    # Calculate (CLOSE>DELAY(CLOSE,1)?VOLUME:(CLOSE<DELAY(CLOSE,1)?-VOLUME:0))
    condition1 = close > delayed_close
    condition2 = close < delayed_close
    
    result_volume = np.where(condition1, volume, 
                            np.where(condition2, -volume, 0))
    
    # Calculate SUM(...,30)
    result = ts_sum(result_volume, 30)
    
    return pd.Series(result, index=df.index, name='alpha_094')

def alpha094(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_094, code, benchmark, end_date, lookback)
import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_086(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha086 factor.
    Formula: ((0.25<(((DELAY(CLOSE,20)-DELAY(CLOSE,10))/10)-((DELAY(CLOSE,10)-CLOSE)/10)))?(-1*1):(((((DELAY(CLOSE,20)-DELAY(CLOSE,10))/10)-((DELAY(CLOSE,10)-CLOSE)/10))<0)?1:((-1*1)*(CLOSE-DELAY(CLOSE,1)))))
    """
    close = df['close'].values
    
    # Calculate DELAY(CLOSE,20)
    delayed_close_20 = delay(close, 20)
    
    # Calculate DELAY(CLOSE,10)
    delayed_close_10 = delay(close, 10)
    
    # Calculate ((DELAY(CLOSE,20)-DELAY(CLOSE,10))/10)
    term1 = (delayed_close_20 - delayed_close_10) / 10
    
    # Calculate ((DELAY(CLOSE,10)-CLOSE)/10)
    term2 = (delayed_close_10 - close) / 10
    
    # Calculate the difference
    diff = term1 - term2
    
    # Calculate DELAY(CLOSE,1)
    delayed_close_1 = delay(close, 1)
    
    # Calculate (CLOSE-DELAY(CLOSE,1))
    close_diff = close - delayed_close_1
    
    # Apply the conditional logic
    condition1 = diff > 0.25
    condition2 = diff < 0
    
    result = np.where(condition1, -1,
                     np.where(condition2, 1, -1 * close_diff))
    
    return pd.Series(result, index=df.index, name='alpha_086')

def alpha086(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_086, code, benchmark, end_date, lookback)
import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_098(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha098 factor.
    Formula: ((((DELTA((SUM(CLOSE,100)/100),100))/DELAY(CLOSE,100))<0.05)||((DELTA((SUM(CLOSE,100)/100),100))/DELAY(CLOSE,100))==0.05))?(-1*(CLOSE-TSMIN(CLOSE,100))):(-1*DELTA(CLOSE,3)))
    """
    close = df['close'].values
    
    # Calculate SUM(CLOSE,100)/100
    sum_close_100 = ts_sum(close, 100)
    mean_close_100 = sum_close_100 / 100
    
    # Calculate DELTA((SUM(CLOSE,100)/100),100)
    delta_mean = delta(mean_close_100, 100)
    
    # Calculate DELAY(CLOSE,100)
    delayed_close = delay(close, 100)
    
    # Calculate (DELTA((SUM(CLOSE,100)/100),100))/DELAY(CLOSE,100)
    ratio = delta_mean / delayed_close
    
    # Handle division by zero
    ratio[delayed_close == 0] = np.nan
    
    # Calculate conditions
    condition1 = ratio < 0.05
    condition2 = ratio == 0.05
    
    # Calculate terms
    term1 = -1 * (close - ts_min(close, 100))
    term2 = -1 * delta(close, 3)
    
    # Apply conditional logic
    result = np.where(condition1 | condition2, term1, term2)
    
    return pd.Series(result, index=df.index, name='alpha_098')

def alpha098(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_098, code, benchmark, end_date, lookback)
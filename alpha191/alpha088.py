import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_088(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha088 factor.
    Formula: (CLOSE-DELAY(CLOSE,20))/DELAY(CLOSE,20)*100
    """
    close = df['close'].values
    
    # Calculate (CLOSE-DELAY(CLOSE,20))/DELAY(CLOSE,20)*100
    delayed_close = delay(close, 20)
    
    # Handle division by zero
    delayed_close[delayed_close == 0] = np.nan
    result = (close - delayed_close) / delayed_close * 100
    
    return pd.Series(result, index=df.index, name='alpha_088')

def alpha088(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_088, code, benchmark, end_date, lookback)
import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_071(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha071 factor (inverted).
    Formula: (MEAN(CLOSE,24)-CLOSE)/MEAN(CLOSE,24)*100
    """
    # Calculate (CLOSE-MEAN(CLOSE,24))/MEAN(CLOSE,24)*100
    close = df['close'].values
    mean_24 = ts_mean(close, 24)
    
    # Handle division by zero
    denom = mean_24
    denom[denom == 0] = np.nan
    result = (denom - close) / denom * 100
    
    return pd.Series(result, index=df.index, name='alpha_071')

def alpha071(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_071, code, benchmark, end_date, lookback)
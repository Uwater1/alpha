import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_078(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha078 factor.
    Formula: ((HIGH+LOW+CLOSE)/3-MA((HIGH+LOW+CLOSE)/3,12))/(0.015*MEAN(ABS(CLOSE-MEAN((HIGH+LOW+CLOSE)/3,12)),12))
    """
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    
    # Calculate (HIGH+LOW+CLOSE)/3
    hlcc3 = (high + low + close) / 3
    
    # Calculate MA((HIGH+LOW+CLOSE)/3,12)
    ma_12 = ts_mean(hlcc3, 12)
    
    # Calculate numerator: (HIGH+LOW+CLOSE)/3 - MA((HIGH+LOW+CLOSE)/3,12)
    numerator = hlcc3 - ma_12
    
    # Calculate denominator: 0.015 * MEAN(ABS(CLOSE - MEAN((HIGH+LOW+CLOSE)/3,12)),12)
    close_minus_ma = close - ma_12
    abs_close_minus_ma = np.abs(close_minus_ma)
    mean_abs = ts_mean(abs_close_minus_ma, 12)
    denominator = 0.015 * mean_abs
    
    # Handle division by zero
    denominator[denominator == 0] = np.nan
    result = numerator / denominator
    
    return pd.Series(result, index=df.index, name='alpha_078')

def alpha078(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_078, code, benchmark, end_date, lookback)
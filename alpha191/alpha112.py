import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_112(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha112 factor.
    Formula: (SUM((CLOSE-DELAY(CLOSE,1)>0?CLOSE-DELAY(CLOSE,1):0),12)-SUM((CLOSE-DELAY(CLOSE,1)<0?ABS(CLOSE-DELAY(CLOSE,1)):0),12))/(SUM((CLOSE-DELAY(CLOSE,1)>0?CLOSE-DELAY(CLOSE,1):0),12)+SUM((CLOSE-DELAY(CLOSE,1)<0?ABS(CLOSE-DELAY(CLOSE,1)):0),12))*100
    """
    # Extract values as numpy arrays
    close = df['close'].values
    
    # Calculate close change
    close_change = close - delay(close, 1)
    
    # Calculate positive close change
    positive_close_change = np.where(close_change > 0, close_change, 0)
    
    # Calculate negative close change
    negative_close_change = np.where(close_change < 0, np.abs(close_change), 0)
    
    # Calculate sum of positive close change
    sum_positive_close_change = ts_sum(positive_close_change, 12)
    
    # Calculate sum of negative close change
    sum_negative_close_change = ts_sum(negative_close_change, 12)
    
    # Calculate final result
    result = (sum_positive_close_change - sum_negative_close_change) / (sum_positive_close_change + sum_negative_close_change) * 100
    
    return pd.Series(result, index=df.index, name='alpha_112')

def alpha112(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_112, code, benchmark, end_date, lookback)
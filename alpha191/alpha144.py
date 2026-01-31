import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_144(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha144 factor.
    Formula: SUMIF(ABS(CLOSE/DELAY(CLOSE,1)-1)/AMOUNT,20,CLOSE<DELAY(CLOSE,1))/COUNT(CLOSE<DELAY(CLOSE,1),20)
    """
    # Extract values as numpy arrays
    close = df['close'].values
    amount = df['amount'].values
    
    # Calculate close/DELAY(close,1)-1
    delay_close = delay(close, 1)
    close_ratio = close / delay_close - 1
    
    # Calculate ABS(close/DELAY(close,1)-1)/amount
    numerator = np.abs(close_ratio) / amount
    
    # Calculate CLOSE < DELAY(CLOSE,1) condition
    condition = close < delay_close
    
    # Calculate SUMIF: sum of numerator where condition is True, over last 20 days
    sumif_result = np.full(len(df), np.nan)
    for i in range(20, len(df)):
        if condition[i-20:i].any():
            sumif_result[i] = np.sum(numerator[i-20:i][condition[i-20:i]])
    
    # Calculate COUNT: count of True conditions over last 20 days
    count_result = np.full(len(df), np.nan)
    for i in range(20, len(df)):
        count_result[i] = np.sum(condition[i-20:i])
    
    # Calculate final result: sumif / count
    result = np.where(count_result != 0, sumif_result / count_result, np.nan)
    
    return pd.Series(result, index=df.index, name='alpha_144')

def alpha144(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_144, code, benchmark, end_date, lookback)
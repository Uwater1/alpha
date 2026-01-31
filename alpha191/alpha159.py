import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_159(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha159 factor.
    Formula: ((CLOSE-SUM(MIN(LOW,DELAY(CLOSE,1)),6))/SUM(MAX(HGIH,DELAY(CLOSE,1))-MIN(LOW,DELAY(CLOSE,1)),6)*12*24+(CLOSE-SUM(MIN(LOW,DELAY(CLOSE,1)),12))/SUM(MAX(HGIH,DELAY(CLOSE,1))-MIN(LOW,DELAY(CLOSE,1)),12)*6*24+(CLOSE-SUM(MIN(LOW,DELAY(CLOSE,1)),24))/SUM(MAX(HGIH,DELAY(CLOSE,1))-MIN(LOW,DELAY(CLOSE,1)),24)*6*24)*100/(6*12+6*24+12*24)
    """
    # Extract values as numpy arrays
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    
    # Calculate DELAY(CLOSE,1)
    delay_close = delay(close, 1)
    
    # Calculate MIN(LOW,DELAY(CLOSE,1))
    min_low_delay = np.minimum(low, delay_close)
    
    # Calculate MAX(HIGH,DELAY(CLOSE,1))
    max_high_delay = np.maximum(high, delay_close)
    
    # Calculate MAX(HIGH,DELAY(CLOSE,1))-MIN(LOW,DELAY(CLOSE,1))
    diff = max_high_delay - min_low_delay
    
    # Calculate SUM(MIN(LOW,DELAY(CLOSE,1)),6)
    sum_min_6 = ts_sum(min_low_delay, 6)
    
    # Calculate SUM(MAX(HIGH,DELAY(CLOSE,1))-MIN(LOW,DELAY(CLOSE,1)),6)
    sum_diff_6 = ts_sum(diff, 6)
    
    # Calculate SUM(MIN(LOW,DELAY(CLOSE,1)),12)
    sum_min_12 = ts_sum(min_low_delay, 12)
    
    # Calculate SUM(MAX(HIGH,DELAY(CLOSE,1))-MIN(LOW,DELAY(CLOSE,1)),12)
    sum_diff_12 = ts_sum(diff, 12)
    
    # Calculate SUM(MIN(LOW,DELAY(CLOSE,1)),24)
    sum_min_24 = ts_sum(min_low_delay, 24)
    
    # Calculate SUM(MAX(HIGH,DELAY(CLOSE,1))-MIN(LOW,DELAY(CLOSE,1)),24)
    sum_diff_24 = ts_sum(diff, 24)
    
    # Calculate each term
    # Protect against division by zero
    sum_diff_6_safe = sum_diff_6.copy()
    sum_diff_6_safe[sum_diff_6_safe == 0] = np.nan
    sum_diff_12_safe = sum_diff_12.copy()
    sum_diff_12_safe[sum_diff_12_safe == 0] = np.nan
    sum_diff_24_safe = sum_diff_24.copy()
    sum_diff_24_safe[sum_diff_24_safe == 0] = np.nan
    
    term1 = (close - sum_min_6) / sum_diff_6_safe * 12 * 24
    term2 = (close - sum_min_12) / sum_diff_12_safe * 6 * 24
    term3 = (close - sum_min_24) / sum_diff_24_safe * 6 * 24
    
    # Calculate denominator
    denominator = 6 * 12 + 6 * 24 + 12 * 24
    
    # Calculate final result
    result = (term1 + term2 + term3) * 100 / denominator
    
    return pd.Series(result, index=df.index, name='alpha_159')

def alpha159(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_159, code, benchmark, end_date, lookback)
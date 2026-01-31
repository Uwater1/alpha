import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_052(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha052 factor.
    Formula: SUM(MAX(0,HIGH-DELAY((HIGH+LOW+CLOSE)/3,1)),26)/SUM(MAX(0,DELAY((HIGH+LOW+CLOSE)/3,1)-LOW),26)*100
    """
    # Calculate VWAP-like value (HIGH+LOW+CLOSE)/3
    vwap_like = (df['high'] + df['low'] + df['close']) / 3
    
    # Calculate DELAY((HIGH+LOW+CLOSE)/3, 1)
    delayed_vwap = delay(vwap_like, 1)
    
    # Calculate numerator: MAX(0, HIGH - DELAY((HIGH+LOW+CLOSE)/3, 1))
    numerator = np.maximum(0, df['high'] - delayed_vwap)
    
    # Calculate denominator: MAX(0, DELAY((HIGH+LOW+CLOSE)/3, 1) - LOW)
    denominator = np.maximum(0, delayed_vwap - df['low'])
    
    # Calculate rolling sums over 26 periods
    sum_numerator = ts_sum(numerator, 26)
    sum_denominator = ts_sum(denominator, 26)
    
    # Handle division by zero
    sum_denominator[sum_denominator == 0] = np.nan
    
    # Calculate final result
    result = (sum_numerator / sum_denominator) * 100
    
    return pd.Series(result, index=df.index, name='alpha_052')

def alpha052(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_052, code, benchmark, end_date, lookback)
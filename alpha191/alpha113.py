import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_113(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha113 factor.
    Formula: (-1*((RANK((SUM(DELAY(CLOSE,5),20)/20))*CORR(CLOSE,VOLUME,2))*RANK(CORR(SUM(CLOSE,5),SUM(CLOSE,20),2))))
    """
    # Extract values as numpy arrays
    close = df['close'].values
    volume = df['volume'].values
    
    # Calculate sum of delay of close
    sum_delay_close = ts_sum(delay(close, 5), 20)
    
    # Calculate rank of sum of delay of close
    rank_sum_delay_close = rank(sum_delay_close / 20)
    
    # Calculate correlation between close and volume
    corr_close_volume = rolling_corr(close, volume, 2)
    
    # Calculate sum of close
    sum_close_5 = ts_sum(close, 5)
    sum_close_20 = ts_sum(close, 20)
    
    # Calculate correlation between sum of close
    corr_sum_close = rolling_corr(sum_close_5, sum_close_20, 2)
    
    # Calculate rank of correlation
    rank_corr_sum_close = rank(corr_sum_close)
    
    # Calculate final result
    result = -1 * (rank_sum_delay_close * corr_close_volume) * rank_corr_sum_close
    
    return pd.Series(result, index=df.index, name='alpha_113')

def alpha113(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_113, code, benchmark, end_date, lookback)
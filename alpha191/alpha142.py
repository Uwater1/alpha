import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_142(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha142 factor.
    Formula: (((-1*RANK(TSRANK(CLOSE,10)))*RANK(DELTA(DELTA(CLOSE,1),1)))*RANK(TSRANK((VOLUME/MEAN(VOLUME,20)),5)))
    """
    # Extract values as numpy arrays
    close = df['close'].values
    volume = df['volume'].values
    
    # Calculate ts_rank of close over 10 periods
    ts_rank_close = ts_rank(close, 10)
    
    # Calculate delta of close over 1 period
    delta_close_1 = delta(close, 1)
    
    # Calculate delta of delta_close_1 over 1 period
    delta_delta_close = delta(delta_close_1, 1)
    
    # Calculate mean volume over 20 periods
    mean_volume = ts_mean(volume, 20)
    
    # Calculate volume/mean_volume
    volume_ratio = volume / mean_volume
    
    # Calculate ts_rank of volume_ratio over 5 periods
    ts_rank_volume = ts_rank(volume_ratio, 5)
    
    # Calculate ranks
    rank_ts_rank_close = rank(ts_rank_close)
    rank_delta_delta_close = rank(delta_delta_close)
    rank_ts_rank_volume = rank(ts_rank_volume)
    
    # Calculate final result
    result = (-1 * rank_ts_rank_close) * rank_delta_delta_close * rank_ts_rank_volume
    
    return pd.Series(result, index=df.index, name='alpha_142')

def alpha142(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_142, code, benchmark, end_date, lookback)
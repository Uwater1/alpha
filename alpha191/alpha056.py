import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_056(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha056 factor.
    Formula: (RANK((OPEN-TSMIN(OPEN,12)))<RANK((RANK(CORR(SUM(((HIGH +LOW)/2),19),SUM(MEAN(VOLUME,40),19),13))^5)))
    """
    # Calculate OPEN - TSMIN(OPEN, 12)
    open_min_diff = df['open'] - ts_min(df['open'], 12)
    
    # Calculate (HIGH + LOW)/2
    high_low_avg = (df['high'] + df['low']) / 2
    
    # Calculate SUM(((HIGH + LOW)/2), 19)
    sum_high_low = ts_sum(high_low_avg, 19)
    
    # Calculate MEAN(VOLUME, 40)
    mean_volume = ts_mean(df['volume'], 40)
    
    # Calculate SUM(MEAN(VOLUME, 40), 19)
    sum_mean_volume = ts_sum(mean_volume, 19)
    
    # Calculate CORR(SUM(((HIGH + LOW)/2), 19), SUM(MEAN(VOLUME, 40), 19), 13)
    correlation = rolling_corr(sum_high_low, sum_mean_volume, 13)
    
    # Calculate RANK(CORR(...))^5
    ranked_corr_power5 = rank(correlation) ** 5
    
    # Calculate RANK(OPEN - TSMIN(OPEN, 12))
    ranked_open_diff = rank(open_min_diff)
    
    # Calculate RANK(RANK(CORR(...))^5)
    ranked_ranked_corr = rank(ranked_corr_power5)
    
    # Calculate final result: RANK(OPEN-TSMIN(OPEN,12)) < RANK((RANK(CORR(...))^5))
    result = ranked_open_diff < ranked_ranked_corr
    
    return pd.Series(result.astype(float), index=df.index, name='alpha_056')

def alpha056(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_056, code, benchmark, end_date, lookback)
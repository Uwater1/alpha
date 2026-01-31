import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_138(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha138 factor.
    Formula: ((RANK(DECAYLINEAR(DELTA((((LOW*0.7)+(VWAP*0.3))),3),20))-TSRANK(DECAYLINEAR(TSRANK(CORR(TSRANK(LOW,8),TSRANK(MEAN(VOLUME,60),17),5),19),16),7))* -1)
    """
    # Extract values as numpy arrays
    low = df['low'].values
    vwap = df['vwap'].values
    volume = df['volume'].values
    
    # Calculate weighted average of low and VWAP
    weighted_avg = (low * 0.7) + (vwap * 0.3)
    
    # Calculate delta of weighted average
    delta_weighted_avg = delta(weighted_avg, 3)
    
    # Calculate decay linear of delta
    decay_linear_delta = decay_linear(delta_weighted_avg, 20)
    
    # Calculate rank of decay linear of delta
    rank_decay_linear_delta = rank(decay_linear_delta)
    
    # Calculate TSRANK of low
    tsrank_low = ts_rank(low, 8)
    
    # Calculate mean volume
    mean_volume = ts_mean(volume, 60)
    
    # Calculate TSRANK of mean volume
    tsrank_mean_volume = ts_rank(mean_volume, 17)
    
    # Calculate correlation between TSRANK of low and TSRANK of mean volume
    corr_tsrank_low_tsrank_mean_volume = rolling_corr(tsrank_low, tsrank_mean_volume, 5)
    
    # Calculate TSRANK of correlation
    tsrank_corr = ts_rank(corr_tsrank_low_tsrank_mean_volume, 19)
    
    # Calculate decay linear of TSRANK
    decay_linear_tsrank = decay_linear(tsrank_corr, 16)
    
    # Calculate TSRANK of decay linear of TSRANK
    tsrank_decay_linear_tsrank = ts_rank(decay_linear_tsrank, 7)
    
    # Calculate final result
    result = (rank_decay_linear_delta - tsrank_decay_linear_tsrank) * -1
    
    return pd.Series(result, index=df.index, name='alpha_138')

def alpha138(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_138, code, benchmark, end_date, lookback)
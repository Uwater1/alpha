import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_119(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha119 factor.
    Formula: (RANK(DECAYLINEAR(CORR(VWAP,SUM(MEAN(VOLUME,5),26),5),7))-RANK(DECAYLINEAR(TSRANK(MIN(CORR(RANK(OPEN),RANK(MEAN(VOLUME,15)),21),9),7),8)))
    """
    # Extract values as numpy arrays
    vwap = df['vwap'].values
    volume = df['volume'].values
    open_price = df['open'].values
    
    # Calculate mean volume
    mean_volume_5 = ts_mean(volume, 5)
    
    # Calculate sum of mean volume
    sum_mean_volume_5 = ts_sum(mean_volume_5, 26)
    
    # Calculate correlation between VWAP and sum of mean volume
    corr_vwap_sum_mean_volume = rolling_corr(vwap, sum_mean_volume_5, 5)
    
    # Calculate decay linear of correlation
    decay_linear_corr = decay_linear(corr_vwap_sum_mean_volume, 7)
    
    # Calculate rank of decay linear of correlation
    rank_decay_linear_corr = rank(decay_linear_corr)
    
    # Calculate mean volume
    mean_volume_15 = ts_mean(volume, 15)
    
    # Calculate correlation between rank of open and rank of mean volume
    corr_rank_open_rank_mean_volume = rolling_corr(rank(open_price), rank(mean_volume_15), 21)
    
    # Calculate minimum of correlation
    min_corr = np.minimum(corr_rank_open_rank_mean_volume, 9)
    
    # Calculate TSRANK of minimum of correlation
    tsrank_min_corr = ts_rank(min_corr, 7)
    
    # Calculate decay linear of TSRANK
    decay_linear_tsrank = decay_linear(tsrank_min_corr, 8)
    
    # Calculate rank of decay linear of TSRANK
    rank_decay_linear_tsrank = rank(decay_linear_tsrank)
    
    # Calculate final result
    result = rank_decay_linear_corr - rank_decay_linear_tsrank
    
    return pd.Series(result, index=df.index, name='alpha_119')

def alpha119(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_119, code, benchmark, end_date, lookback)
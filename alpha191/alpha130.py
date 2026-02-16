import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_130(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha130 factor.
    Formula: (RANK(DECAYLINEAR(CORR(((HIGH+LOW)/2),MEAN(VOLUME,40),9),10))/RANK(DECAYLINEAR(CORR(RANK(VWAP),RANK(VOLUME),7),3)))
    """
    # Extract values as numpy arrays
    high = df['high'].values
    low = df['low'].values
    volume = df['volume'].values
    vwap = df['vwap'].values
    
    # Calculate average of high and low
    avg_high_low = (high + low) / 2
    
    # Calculate mean volume
    mean_volume = ts_mean(volume, 40)
    
    # Calculate correlation between average of high and low and mean volume
    corr_avg_high_low_mean_volume = rolling_corr(avg_high_low, mean_volume, 9)
    
    # Calculate decay linear of correlation
    decay_linear_corr = decay_linear(corr_avg_high_low_mean_volume, 10)
    
    # Calculate rank of decay linear of correlation
    rank_decay_linear_corr = ts_rank(decay_linear_corr, 20)
    
    # Calculate correlation between rank of VWAP and rank of volume
    corr_rank_vwap_rank_volume = rolling_corr(ts_rank(vwap, 20), ts_rank(volume, 20), 7)
    
    # Calculate decay linear of correlation
    decay_linear_corr_rank = decay_linear(corr_rank_vwap_rank_volume, 3)
    
    # Calculate rank of decay linear of correlation
    rank_decay_linear_corr_rank = ts_rank(decay_linear_corr_rank, 20)
    
    # Protect against division by zero
    denom = rank_decay_linear_corr_rank
    denom[denom == 0] = np.nan
    
    # Calculate final result
    result = rank_decay_linear_corr / denom
    
    return pd.Series(result, index=df.index, name='alpha_130')

def alpha130(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_130, code, benchmark, end_date, lookback)
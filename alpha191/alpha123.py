import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_123(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha123 factor.
    Formula: ((RANK(CORR(SUM(((HIGH+LOW)/2),20),SUM(MEAN(VOLUME,60),20),9))<RANK(CORR(LOW,VOLUME,6)))*-1)
    """
    # Extract values as numpy arrays
    high = df['high'].values
    low = df['low'].values
    volume = df['volume'].values
    
    # Calculate average of high and low
    avg_high_low = (high + low) / 2
    
    # Calculate sum of average of high and low
    sum_avg_high_low = ts_sum(avg_high_low, 20)
    
    # Calculate mean volume
    mean_volume = ts_mean(volume, 60)
    
    # Calculate sum of mean volume
    sum_mean_volume = ts_sum(mean_volume, 20)
    
    # Calculate correlation between sum of average of high and low and sum of mean volume
    corr_sum_avg_high_low_sum_mean_volume = rolling_corr(sum_avg_high_low, sum_mean_volume, 9)
    
    # Calculate rank of correlation
    rank_corr_sum_avg_high_low_sum_mean_volume = rank(corr_sum_avg_high_low_sum_mean_volume)
    
    # Calculate correlation between low and volume
    corr_low_volume = rolling_corr(low, volume, 6)
    
    # Calculate rank of correlation
    rank_corr_low_volume = rank(corr_low_volume)
    
    # Calculate final result
    result = np.where(rank_corr_sum_avg_high_low_sum_mean_volume < rank_corr_low_volume, -1, 0)
    
    return pd.Series(result, index=df.index, name='alpha_123')

def alpha123(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_123, code, benchmark, end_date, lookback)
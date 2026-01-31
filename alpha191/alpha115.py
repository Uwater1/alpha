import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_115(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha115 factor.
    Formula: (RANK(CORR(((HIGH*0.9)+(CLOSE*0.1)),MEAN(VOLUME,30),10))^RANK(CORR(TSRANK(((HIGH+LOW)/2),4),TSRANK(VOLUME,10),7)))
    """
    # Extract values as numpy arrays
    high = df['high'].values
    close = df['close'].values
    volume = df['volume'].values
    low = df['low'].values
    
    # Calculate weighted average of high and close
    weighted_avg = (high * 0.9) + (close * 0.1)
    
    # Calculate mean volume
    mean_volume = ts_mean(volume, 30)
    
    # Calculate correlation between weighted average and mean volume
    corr_weighted_avg_mean_volume = rolling_corr(weighted_avg, mean_volume, 10)
    
    # Calculate rank of correlation
    rank_corr_weighted_avg_mean_volume = rank(corr_weighted_avg_mean_volume)
    
    # Calculate average of high and low
    avg_high_low = (high + low) / 2
    
    # Calculate TSRANK of average of high and low
    tsrank_avg_high_low = ts_rank(avg_high_low, 4)
    
    # Calculate TSRANK of volume
    tsrank_volume = ts_rank(volume, 10)
    
    # Calculate correlation between TSRANK of average of high and low and TSRANK of volume
    corr_tsrank_avg_high_low_volume = rolling_corr(tsrank_avg_high_low, tsrank_volume, 7)
    
    # Calculate rank of correlation
    rank_corr_tsrank_avg_high_low_volume = rank(corr_tsrank_avg_high_low_volume)
    
    # Calculate final result
    result = rank_corr_weighted_avg_mean_volume ** rank_corr_tsrank_avg_high_low_volume
    
    return pd.Series(result, index=df.index, name='alpha_115')

def alpha115(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_115, code, benchmark, end_date, lookback)
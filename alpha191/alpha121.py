import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_121(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha121 factor.
    Formula: ((RANK((VWAP-MIN(VWAP,12)))^TSRANK(CORR(TSRANK(VWAP,20),TSRANK(MEAN(VOLUME,60),2),18),3))*-1)
    """
    # Extract values as numpy arrays
    vwap = df['vwap'].values
    volume = df['volume'].values
    
    # Calculate minimum of VWAP
    min_vwap = ts_min(vwap, 12)
    
    # Calculate VWAP minus minimum of VWAP
    vwap_minus_min_vwap = vwap - min_vwap
    
    # Calculate rank of VWAP minus minimum of VWAP
    rank_vwap_minus_min_vwap = rank(vwap_minus_min_vwap)
    
    # Calculate TSRANK of VWAP
    tsrank_vwap = ts_rank(vwap, 20)
    
    # Calculate mean volume
    mean_volume = ts_mean(volume, 60)
    
    # Calculate TSRANK of mean volume
    tsrank_mean_volume = ts_rank(mean_volume, 2)
    
    # Calculate correlation between TSRANK of VWAP and TSRANK of mean volume
    corr_tsrank_vwap_tsrank_mean_volume = rolling_corr(tsrank_vwap, tsrank_mean_volume, 18)
    
    # Calculate TSRANK of correlation
    tsrank_corr = ts_rank(corr_tsrank_vwap_tsrank_mean_volume, 3)
    
    # Calculate final result
    result = (rank_vwap_minus_min_vwap ** tsrank_corr) * -1
    
    return pd.Series(result, index=df.index, name='alpha_121')

def alpha121(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_121, code, benchmark, end_date, lookback)
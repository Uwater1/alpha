import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_131(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha131 factor.
    Formula: (RANK(DELTA(VWAP,1))^TSRANK(CORR(CLOSE,MEAN(VOLUME,50),18),18))
    """
    # Extract values as numpy arrays
    vwap = df['vwap'].values
    close = df['close'].values
    volume = df['volume'].values
    
    # Calculate delta of VWAP
    delta_vwap = delta(vwap, 1)
    
    # Calculate rank of delta of VWAP
    rank_delta_vwap = rank(delta_vwap)
    
    # Calculate mean volume
    mean_volume = ts_mean(volume, 50)
    
    # Calculate correlation between close and mean volume
    corr_close_mean_volume = rolling_corr(close, mean_volume, 18)
    
    # Calculate TSRANK of correlation
    tsrank_corr = ts_rank(corr_close_mean_volume, 18)
    
    # Protect against division by zero in power (when base is 0 and exponent is negative)
    tsrank_corr_safe = tsrank_corr.copy()
    tsrank_corr_safe[(rank_delta_vwap == 0) & (tsrank_corr < 0)] = np.nan
    
    # Calculate final result
    result = rank_delta_vwap ** tsrank_corr_safe
    
    return pd.Series(result, index=df.index, name='alpha_131')

def alpha131(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_131, code, benchmark, end_date, lookback)
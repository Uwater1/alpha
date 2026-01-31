import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_125(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha125 factor.
    Formula: (RANK(DECAYLINEAR(CORR((VWAP),MEAN(VOLUME,80),17),20))/RANK(DECAYLINEAR(DELTA(((CLOSE*0.5)+(VWAP*0.5)),3),16)))
    """
    # Extract values as numpy arrays
    vwap = df['vwap'].values
    volume = df['volume'].values
    close = df['close'].values
    
    # Calculate mean volume
    mean_volume = ts_mean(volume, 80)
    
    # Calculate correlation between VWAP and mean volume
    corr_vwap_mean_volume = rolling_corr(vwap, mean_volume, 17)
    
    # Calculate decay linear of correlation
    decay_linear_corr = decay_linear(corr_vwap_mean_volume, 20)
    
    # Calculate rank of decay linear of correlation
    rank_decay_linear_corr = rank(decay_linear_corr)
    
    # Calculate weighted average of close and VWAP
    weighted_avg = (close * 0.5) + (vwap * 0.5)
    
    # Calculate delta of weighted average
    delta_weighted_avg = delta(weighted_avg, 3)
    
    # Calculate decay linear of delta
    decay_linear_delta = decay_linear(delta_weighted_avg, 16)
    
    # Calculate rank of decay linear of delta
    rank_decay_linear_delta = rank(decay_linear_delta)
    
    # Protect against division by zero
    denom = rank_decay_linear_delta
    denom[denom == 0] = np.nan
    
    # Calculate final result
    result = rank_decay_linear_corr / denom
    
    return pd.Series(result, index=df.index, name='alpha_125')

def alpha125(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_125, code, benchmark, end_date, lookback)
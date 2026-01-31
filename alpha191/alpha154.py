import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_154(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha154 factor.
    Formula: (((VWAP-MIN(VWAP,16)))<(CORR(VWAP,MEAN(VOLUME,180),18)))
    """
    # Extract values as numpy arrays
    vwap = df['vwap'].values
    volume = df['volume'].values
    
    # Calculate MIN(VWAP,16)
    min_vwap_16 = ts_min(vwap, 16)
    
    # Calculate MEAN(VOLUME,180)
    mean_volume_180 = ts_mean(volume, 180)
    
    # Calculate CORR(VWAP, MEAN(VOLUME,180), 18)
    corr_result = rolling_corr(vwap, mean_volume_180, 18)
    
    # Calculate VWAP - MIN(VWAP,16)
    vwap_minus_min = vwap - min_vwap_16
    
    # Calculate final result: (VWAP-MIN(VWAP,16)) < CORR(VWAP,MEAN(VOLUME,180),18)
    result = np.where(vwap_minus_min < corr_result, 1, 0)
    
    return pd.Series(result, index=df.index, name='alpha_154')

def alpha154(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_154, code, benchmark, end_date, lookback)
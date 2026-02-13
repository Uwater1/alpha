import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_178(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha178 factor.
    Formula: ((RANK(CORR((CLOSE),MEAN(VOLUME,20),5))*RANK(((OPEN-CLOSE)/(CLOSE)))+RANK((OPEN-CLOSE)/CLOSE))*RANK(CORR((VWAP),MEAN(VOLUME,60),3)))
    """
    # Extract values as numpy arrays
    close = df['close'].values
    open_price = df['open'].values
    volume = df['volume'].values
    vwap = df['vwap'].values
    
    # Calculate MEAN(VOLUME,20)
    mean_volume_20 = ts_mean(volume, 20)
    
    # Calculate CORR((CLOSE),MEAN(VOLUME,20),5) using Numba-accelerated rolling_corr
    corr_close_volume = rolling_corr(close, mean_volume_20, 5)
    
    # Calculate RANK(CORR(...))
    rank_corr_close_volume = rank(corr_close_volume)
    
    # Calculate (OPEN-CLOSE)/(CLOSE)
    open_close_diff = open_price - close
    
    # Protect against division by zero
    denom = close.copy()
    denom[denom == 0] = np.nan
    
    with np.errstate(invalid='ignore', divide='ignore'):
        ratio1 = open_close_diff / denom
    
    # Calculate RANK(((OPEN-CLOSE)/(CLOSE)))
    rank_ratio1 = rank(ratio1)
    
    # Calculate RANK((OPEN-CLOSE)/CLOSE) - same as above
    rank_ratio2 = rank_ratio1
    
    # Calculate MEAN(VOLUME,60)
    mean_volume_60 = ts_mean(volume, 60)
    
    # Calculate CORR((VWAP),MEAN(VOLUME,60),3) using Numba-accelerated rolling_corr
    corr_vwap_volume = rolling_corr(vwap, mean_volume_60, 3)
    
    # Calculate RANK(CORR((VWAP),MEAN(VOLUME,60),3))
    rank_corr_vwap_volume = rank(corr_vwap_volume)
    
    # Calculate final result: (rank_corr_close_volume * rank_ratio1 + rank_ratio2) * rank_corr_vwap_volume
    result = (rank_corr_close_volume * rank_ratio1 + rank_ratio2) * rank_corr_vwap_volume
    
    return pd.Series(result, index=df.index, name='alpha_178')

def alpha178(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_178, code, benchmark, end_date, lookback)
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
    
    # Calculate CORR((CLOSE),MEAN(VOLUME,20),5)
    # Note: We'll use a rolling window correlation
    corr_close_volume = np.full(len(df), np.nan)
    window = 5
    
    for i in range(window - 1, len(df)):
        # Get the window of data
        close_window = close[i-window+1:i+1]
        volume_window = mean_volume_20[i-window+1:i+1]
        
        # Calculate correlation
        if not np.isnan(close_window).all() and not np.isnan(volume_window).all():
            # Remove NaN values
            valid_mask = ~(np.isnan(close_window) | np.isnan(volume_window))
            if valid_mask.sum() > 1:  # Need at least 2 valid points for correlation
                close_valid = close_window[valid_mask]
                volume_valid = volume_window[valid_mask]
                if len(close_valid) > 1:
                    corr = np.corrcoef(close_valid, volume_valid)[0, 1]
                    if not np.isnan(corr):
                        corr_close_volume[i] = corr
    
    # Calculate RANK(CORR(...))
    rank_corr_close_volume = rank(corr_close_volume)
    
    # Calculate (OPEN-CLOSE)/(CLOSE)
    open_close_diff = open_price - close
    
    # Protect against division by zero
    denom = close.copy()
    denom[denom == 0] = np.nan
    
    ratio1 = open_close_diff / denom
    
    # Calculate RANK(((OPEN-CLOSE)/(CLOSE)))
    rank_ratio1 = rank(ratio1)
    
    # Calculate RANK((OPEN-CLOSE)/CLOSE) - same as above
    rank_ratio2 = rank_ratio1
    
    # Calculate MEAN(VOLUME,60)
    mean_volume_60 = ts_mean(volume, 60)
    
    # Calculate CORR((VWAP),MEAN(VOLUME,60),3)
    corr_vwap_volume = np.full(len(df), np.nan)
    window = 3
    
    for i in range(window - 1, len(df)):
        # Get the window of data
        vwap_window = vwap[i-window+1:i+1]
        volume_window = mean_volume_60[i-window+1:i+1]
        
        # Calculate correlation
        if not np.isnan(vwap_window).all() and not np.isnan(volume_window).all():
            # Remove NaN values
            valid_mask = ~(np.isnan(vwap_window) | np.isnan(volume_window))
            if valid_mask.sum() > 1:  # Need at least 2 valid points for correlation
                vwap_valid = vwap_window[valid_mask]
                volume_valid = volume_window[valid_mask]
                if len(vwap_valid) > 1:
                    corr = np.corrcoef(vwap_valid, volume_valid)[0, 1]
                    if not np.isnan(corr):
                        corr_vwap_volume[i] = corr
    
    # Calculate RANK(CORR((VWAP),MEAN(VOLUME,60),3))
    rank_corr_vwap_volume = rank(corr_vwap_volume)
    
    # Calculate final result: (rank_corr_close_volume * rank_ratio1 + rank_ratio2) * rank_corr_vwap_volume
    result = (rank_corr_close_volume * rank_ratio1 + rank_ratio2) * rank_corr_vwap_volume
    
    return pd.Series(result, index=df.index, name='alpha_178')

def alpha178(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_178, code, benchmark, end_date, lookback)
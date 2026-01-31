import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_179(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha179 factor.
    Formula: (RANK(CORR(VWAP,VOLUME,4))*RANK(CORR(RANK(LOW),RANK(MEAN(VOLUME,50)),12)))
    """
    # Extract values as numpy arrays
    vwap = df['vwap'].values
    volume = df['volume'].values
    low = df['low'].values
    
    # Calculate MEAN(VOLUME,50)
    mean_volume_50 = ts_mean(volume, 50)
    
    # Calculate CORR(VWAP,VOLUME,4)
    # Note: We'll use a rolling window correlation
    corr_vwap_volume = np.full(len(df), np.nan)
    window = 4
    
    for i in range(window - 1, len(df)):
        # Get the window of data
        vwap_window = vwap[i-window+1:i+1]
        volume_window = volume[i-window+1:i+1]
        
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
    
    # Calculate RANK(CORR(VWAP,VOLUME,4))
    rank_corr_vwap_volume = rank(corr_vwap_volume)
    
    # Calculate RANK(LOW)
    rank_low = rank(low)
    
    # Calculate RANK(MEAN(VOLUME,50))
    rank_mean_volume_50 = rank(mean_volume_50)
    
    # Calculate CORR(RANK(LOW),RANK(MEAN(VOLUME,50)),12)
    corr_rank_low_volume = np.full(len(df), np.nan)
    window = 12
    
    for i in range(window - 1, len(df)):
        # Get the window of data
        rank_low_window = rank_low[i-window+1:i+1]
        rank_volume_window = rank_mean_volume_50[i-window+1:i+1]
        
        # Calculate correlation
        if not np.isnan(rank_low_window).all() and not np.isnan(rank_volume_window).all():
            # Remove NaN values
            valid_mask = ~(np.isnan(rank_low_window) | np.isnan(rank_volume_window))
            if valid_mask.sum() > 1:  # Need at least 2 valid points for correlation
                rank_low_valid = rank_low_window[valid_mask]
                rank_volume_valid = rank_volume_window[valid_mask]
                if len(rank_low_valid) > 1:
                    corr = np.corrcoef(rank_low_valid, rank_volume_valid)[0, 1]
                    if not np.isnan(corr):
                        corr_rank_low_volume[i] = corr
    
    # Calculate RANK(CORR(RANK(LOW),RANK(MEAN(VOLUME,50)),12))
    rank_corr_rank_low_volume = rank(corr_rank_low_volume)
    
    # Calculate final result: RANK(CORR(VWAP,VOLUME,4)) * RANK(CORR(RANK(LOW),RANK(MEAN(VOLUME,50)),12))
    result = rank_corr_vwap_volume * rank_corr_rank_low_volume
    
    return pd.Series(result, index=df.index, name='alpha_179')

def alpha179(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_179, code, benchmark, end_date, lookback)
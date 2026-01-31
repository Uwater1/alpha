import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_176(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha176 factor.
    Formula: CORR(RANK(((CLOSE-TS_MIN(LOW,12))/(TS_MAX(HIGH,12)-TS_MIN(LOW,12)))),RANK(VOLUME),6)
    """
    # Extract values as numpy arrays
    close = df['close'].values
    low = df['low'].values
    high = df['high'].values
    volume = df['volume'].values
    
    # Calculate TS_MIN(LOW,12)
    min_low_12 = ts_min(low, 12)
    
    # Calculate TS_MAX(HIGH,12)
    max_high_12 = ts_max(high, 12)
    
    # Calculate CLOSE-TS_MIN(LOW,12)
    close_min_diff = close - min_low_12
    
    # Calculate TS_MAX(HIGH,12)-TS_MIN(LOW,12)
    range_diff = max_high_12 - min_low_12
    
    # Calculate (CLOSE-TS_MIN(LOW,12))/(TS_MAX(HIGH,12)-TS_MIN(LOW,12))
    ratio = close_min_diff / range_diff
    
    # Calculate RANK(...)
    rank_ratio = rank(ratio)
    
    # Calculate RANK(VOLUME)
    rank_volume = rank(volume)
    
    # Calculate CORR(RANK(...),RANK(VOLUME),6)
    # Note: We'll use a rolling window correlation
    corr_values = np.full(len(df), np.nan)
    window = 6
    
    for i in range(window - 1, len(df)):
        # Get the window of data
        rank_ratio_window = rank_ratio[i-window+1:i+1]
        rank_volume_window = rank_volume[i-window+1:i+1]
        
        # Calculate correlation
        if not np.isnan(rank_ratio_window).all() and not np.isnan(rank_volume_window).all():
            # Remove NaN values
            valid_mask = ~(np.isnan(rank_ratio_window) | np.isnan(rank_volume_window))
            if valid_mask.sum() > 1:  # Need at least 2 valid points for correlation
                rank_ratio_valid = rank_ratio_window[valid_mask]
                rank_volume_valid = rank_volume_window[valid_mask]
                if len(rank_ratio_valid) > 1:
                    corr = np.corrcoef(rank_ratio_valid, rank_volume_valid)[0, 1]
                    if not np.isnan(corr):
                        corr_values[i] = corr
    
    return pd.Series(corr_values, index=df.index, name='alpha_176')

def alpha176(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_176, code, benchmark, end_date, lookback)
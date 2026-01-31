import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_180(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha180 factor.
    Formula: ((MEAN(VOLUME,20)<VOLUME)?((-1*TSRANK(ABS(DELTA(CLOSE,7)),60))*SIGN(DELTA(CLOSE,7)):(-1*VOLUME/MEAN(VOLUME,20)))
    Note: The formula seems to have a syntax error. Assuming it's a conditional expression.
    """
    # Extract values as numpy arrays
    close = df['close'].values
    volume = df['volume'].values
    
    # Calculate MEAN(VOLUME,20)
    mean_volume_20 = ts_mean(volume, 20)
    
    # Calculate MEAN(VOLUME,20)<VOLUME
    condition = mean_volume_20 < volume
    
    # Calculate DELTA(CLOSE,7)
    delta_close_7 = delta(close, 7)
    
    # Calculate ABS(DELTA(CLOSE,7))
    abs_delta_close_7 = np.abs(delta_close_7)
    
    # Calculate TSRANK(ABS(DELTA(CLOSE,7)),60)
    # Note: TSRANK is time-series rank, we'll implement it as a rolling rank
    tsrank_values = np.full(len(df), np.nan)
    window = 60
    
    for i in range(window - 1, len(df)):
        # Get the window of data
        window_data = abs_delta_close_7[i-window+1:i+1]
        
        # Calculate rank within the window
        if not np.isnan(window_data).all():
            # Remove NaN values
            valid_mask = ~np.isnan(window_data)
            if valid_mask.sum() > 0:
                valid_data = window_data[valid_mask]
                if len(valid_data) > 0:
                    # Calculate rank (1-based)
                    sorted_indices = np.argsort(valid_data)
                    ranks = np.empty_like(sorted_indices)
                    ranks[sorted_indices] = np.arange(1, len(valid_data) + 1)
                    
                    # Map back to original positions
                    tsrank_values[i] = ranks[-1]  # Rank of the last element
    
    # Calculate SIGN(DELTA(CLOSE,7))
    sign_delta_close_7 = np.sign(delta_close_7)
    
    # Calculate (-1*TSRANK(ABS(DELTA(CLOSE,7)),60))*SIGN(DELTA(CLOSE,7))
    true_branch = (-1 * tsrank_values) * sign_delta_close_7
    
    # Calculate (-1*VOLUME/MEAN(VOLUME,20))
    false_branch = -1 * volume / mean_volume_20
    
    # Calculate final result using conditional
    result = np.where(condition, true_branch, false_branch)
    
    return pd.Series(result, index=df.index, name='alpha_180')

def alpha180(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_180, code, benchmark, end_date, lookback)
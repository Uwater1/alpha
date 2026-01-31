import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_184(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha184 factor.
    Formula: RANK(CORR(DELAY((OPEN-CLOSE),1),CLOSE,200))+RANK((OPEN-CLOSE))
    """
    # Extract values as numpy arrays
    open_price = df['open'].values
    close = df['close'].values
    
    # Calculate OPEN-CLOSE
    open_close_diff = open_price - close
    
    # Calculate DELAY((OPEN-CLOSE),1)
    delay_open_close = delay(open_close_diff, 1)
    
    # Calculate CORR(DELAY((OPEN-CLOSE),1),CLOSE,200)
    # Note: We'll use a rolling window correlation
    corr_values = np.full(len(df), np.nan)
    window = 200
    
    for i in range(window - 1, len(df)):
        # Get the window of data
        delay_window = delay_open_close[i-window+1:i+1]
        close_window = close[i-window+1:i+1]
        
        # Calculate correlation
        if not np.isnan(delay_window).all() and not np.isnan(close_window).all():
            # Remove NaN values
            valid_mask = ~(np.isnan(delay_window) | np.isnan(close_window))
            if valid_mask.sum() > 1:  # Need at least 2 valid points for correlation
                delay_valid = delay_window[valid_mask]
                close_valid = close_window[valid_mask]
                if len(delay_valid) > 1:
                    corr = np.corrcoef(delay_valid, close_valid)[0, 1]
                    if not np.isnan(corr):
                        corr_values[i] = corr
    
    # Calculate RANK(CORR(...))
    rank_corr = rank(corr_values)
    
    # Calculate RANK((OPEN-CLOSE))
    rank_open_close = rank(open_close_diff)
    
    # Calculate final result: RANK(CORR(...)) + RANK((OPEN-CLOSE))
    result = rank_corr + rank_open_close
    
    return pd.Series(result, index=df.index, name='alpha_184')

def alpha184(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_184, code, benchmark, end_date, lookback)
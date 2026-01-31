import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_190(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha190 factor.
    Formula: LOG(MEAN(VOLUME,20),RANK(((CLOSE-RANK((OPEN+HIGH-LOW))/RANK(VOLUME))*128)))
    Note: This formula seems to have a syntax error. Assuming it's LOG of MEAN(VOLUME,20) with base RANK(...)
    """
    # Extract values as numpy arrays
    close = df['close'].values
    open_price = df['open'].values
    high = df['high'].values
    low = df['low'].values
    volume = df['volume'].values
    
    # Calculate OPEN+HIGH-LOW
    open_high_low = open_price + high - low
    
    # Calculate RANK((OPEN+HIGH-LOW))
    rank_open_high_low = rank(open_high_low)
    
    # Calculate CLOSE-RANK((OPEN+HIGH-LOW))
    close_diff = close - rank_open_high_low
    
    # Calculate RANK(VOLUME)
    rank_volume = rank(volume)
    
    # Protect against division by zero
    denom = rank_volume
    denom[denom == 0] = np.nan
    
    # Calculate (CLOSE-RANK((OPEN+HIGH-LOW))/RANK(VOLUME))
    ratio = close_diff / denom
    
    # Calculate (ratio)*128
    scaled_ratio = ratio * 128
    
    # Calculate RANK(scaled_ratio)
    rank_scaled_ratio = rank(scaled_ratio)
    
    # Calculate MEAN(VOLUME,20)
    mean_volume_20 = ts_mean(volume, 20)
    
    # Protect against log of zero/negative values and division by zero
    log_input = mean_volume_20.copy()
    log_input[log_input <= 0] = np.nan
    
    log_base = rank_scaled_ratio.copy()
    log_base[log_base <= 0] = np.nan
    log_base_val = np.log(log_base)
    log_base_val[log_base_val == 0] = np.nan
    
    # Calculate LOG(MEAN(VOLUME,20), RANK(...))
    # Note: This is log base RANK(...) of MEAN(VOLUME,20)
    # Using change of base formula: log_b(a) = ln(a)/ln(b)
    result = np.log(log_input) / log_base_val
    
    return pd.Series(result, index=df.index, name='alpha_190')

def alpha190(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_190, code, benchmark, end_date, lookback)
import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_149(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha149 factor.
    Formula: REGBETA(FILTER(CLOSE/DELAY(CLOSE,1)-1,BANCHMARKINDEXCLOSE<DELAY(BANCHMARKINDEXCLOSE,1)),FILTER(BANCHMARKINDEXCLOSE/DELAY(BANCHMARKINDEXCLOSE,1)-1,BANCHMARKINDEXCLOSE<DELAY(BANCHMARKINDEXCLOSE,1)),252)
    """
    # Extract values as numpy arrays
    close = df['close'].values
    benchmark_close = df['benchmark_close'].values
    
    # Calculate returns
    close_ret = close / delay(close, 1) - 1
    benchmark_ret = benchmark_close / delay(benchmark_close, 1) - 1
    
    # Create filter condition: BANCHMARKINDEXCLOSE < DELAY(BANCHMARKINDEXCLOSE,1)
    filter_condition = benchmark_close < delay(benchmark_close, 1)
    
    # Apply filter to both return series
    filtered_close_ret = np.where(filter_condition, close_ret, np.nan)
    filtered_benchmark_ret = np.where(filter_condition, benchmark_ret, np.nan)
    
    # Calculate regression beta over 252 days
    result = np.full(len(df), np.nan)
    
    for i in range(252, len(df)):
        # Get the window of filtered data
        window_close_ret = filtered_close_ret[i-252:i]
        window_benchmark_ret = filtered_benchmark_ret[i-252:i]
        
        # Remove NaN values
        valid_mask = ~(np.isnan(window_close_ret) | np.isnan(window_benchmark_ret))
        
        if valid_mask.sum() < 2:  # Need at least 2 points for regression
            continue
            
        clean_close_ret = window_close_ret[valid_mask]
        clean_benchmark_ret = window_benchmark_ret[valid_mask]
        
        # Calculate regression
        if len(clean_close_ret) > 1:
            coeffs = np.polyfit(clean_benchmark_ret, clean_close_ret, 1)
            result[i] = coeffs[0]  # The slope (beta)
    
    return pd.Series(result, index=df.index, name='alpha_149')

def alpha149(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_149, code, benchmark, end_date, lookback)
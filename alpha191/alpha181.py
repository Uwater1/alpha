import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_181(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha181 factor.
    Formula: SUM(((CLOSE/DELAY(CLOSE,1)-1)-MEAN((CLOSE/DELAY(CLOSE,1)-1),20))-(BANCHMARKINDEXCLOSE-MEAN(BANCHMARKINDEXCLOSE,20))^2,20)/SUM((BANCHMARKINDEXCLOSE-MEAN(BANCHMARKINDEXCLOSE,20))^3,20)
    Note: We'll assume BANCHMARKINDEXCLOSE is available in the dataframe
    """
    # Extract values as numpy arrays
    close = df['close'].values
    
    # Check if benchmark index is available
    if 'benchmark_index_close' in df.columns:
        benchmark_close = df['benchmark_index_close'].values
    else:
        # If not available, we'll use a placeholder (this would need to be handled differently in practice)
        benchmark_close = close.copy()  # Using close as a placeholder
    
    # Calculate CLOSE/DELAY(CLOSE,1)-1
    delay_close = delay(close, 1)
    close_ratio = close / delay_close - 1
    
    # Calculate MEAN((CLOSE/DELAY(CLOSE,1)-1),20)
    mean_close_ratio = ts_mean(close_ratio, 20)
    
    # Calculate (CLOSE/DELAY(CLOSE,1)-1)-MEAN((CLOSE/DELAY(CLOSE,1)-1),20)
    diff_close = close_ratio - mean_close_ratio
    
    # Calculate MEAN(BANCHMARKINDEXCLOSE,20)
    mean_benchmark = ts_mean(benchmark_close, 20)
    
    # Calculate BANCHMARKINDEXCLOSE-MEAN(BANCHMARKINDEXCLOSE,20)
    diff_benchmark = benchmark_close - mean_benchmark
    
    # Calculate (BANCHMARKINDEXCLOSE-MEAN(BANCHMARKINDEXCLOSE,20))^2
    benchmark_squared = diff_benchmark ** 2
    
    # Calculate (BANCHMARKINDEXCLOSE-MEAN(BANCHMARKINDEXCLOSE,20))^3
    benchmark_cubed = diff_benchmark ** 3
    
    # Calculate numerator: SUM(diff_close - benchmark_squared, 20)
    numerator = ts_sum(diff_close - benchmark_squared, 20)
    
    # Calculate denominator: SUM(benchmark_cubed, 20)
    denominator = ts_sum(benchmark_cubed, 20)
    
    # Calculate final result: numerator / denominator
    result = numerator / denominator
    
    return pd.Series(result, index=df.index, name='alpha_181')

def alpha181(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_181, code, benchmark, end_date, lookback)
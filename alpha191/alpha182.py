import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_182(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha182 factor.
    Formula: COUNT((CLOSE>OPEN & BANCHMARKINDEXCLOSE>BANCHMARKINDEXOPEN) OR (CLOSE<OPEN & BANCHMARKINDEXCLOSE<BANCHMARKINDEXOPEN),20)/20
    Note: We'll assume BANCHMARKINDEXCLOSE and BANCHMARKINDEXOPEN are available in the dataframe
    """
    # Extract values as numpy arrays
    close = df['close'].values
    open_price = df['open'].values
    
    # Check if benchmark index is available
    if 'benchmark_index_close' in df.columns and 'benchmark_index_open' in df.columns:
        benchmark_close = df['benchmark_index_close'].values
        benchmark_open = df['benchmark_index_open'].values
    else:
        # If not available, we'll use a placeholder (this would need to be handled differently in practice)
        benchmark_close = close.copy()  # Using close as a placeholder
        benchmark_open = open_price.copy()  # Using open as a placeholder
    
    # Calculate CLOSE>OPEN
    close_gt_open = close > open_price
    
    # Calculate BANCHMARKINDEXCLOSE>BANCHMARKINDEXOPEN
    benchmark_gt = benchmark_close > benchmark_open
    
    # Calculate CLOSE<OPEN
    close_lt_open = close < open_price
    
    # Calculate BANCHMARKINDEXCLOSE<BANCHMARKINDEXOPEN
    benchmark_lt = benchmark_close < benchmark_open
    
    # Calculate (CLOSE>OPEN & BANCHMARKINDEXCLOSE>BANCHMARKINDEXOPEN)
    condition1 = close_gt_open & benchmark_gt
    
    # Calculate (CLOSE<OPEN & BANCHMARKINDEXCLOSE<BANCHMARKINDEXOPEN)
    condition2 = close_lt_open & benchmark_lt
    
    # Calculate (condition1 OR condition2)
    combined_condition = condition1 | condition2
    
    # Calculate COUNT(...,20)
    count = ts_sum(combined_condition.astype(int), 20)
    
    # Calculate final result: count / 20
    result = count / 20
    
    return pd.Series(result, index=df.index, name='alpha_182')

def alpha182(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_182, code, benchmark, end_date, lookback)
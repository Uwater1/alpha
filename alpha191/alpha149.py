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
    
    # Apply filter: set to NaN where condition is False
    # FILTER returns the value where condition is True, NaN otherwise
    filtered_close_ret = np.where(filter_condition, close_ret, np.nan)
    filtered_benchmark_ret = np.where(filter_condition, benchmark_ret, np.nan)
    
    # Calculate regression beta over 252 days using the regression_beta operator
    # regression_beta handles NaN values correctly by excluding them from computation
    result = regression_beta(filtered_close_ret, filtered_benchmark_ret, 252)
    
    return pd.Series(result, index=df.index, name='alpha_149')

def alpha149(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_149, code, benchmark, end_date, lookback)

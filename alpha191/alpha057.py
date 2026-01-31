import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_057(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha057 factor.
    Formula: SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1)
    """
    # Calculate CLOSE - TSMIN(LOW, 9)
    close_min_low_diff = df['close'] - ts_min(df['low'], 9)
    
    # Calculate TSMAX(HIGH, 9) - TSMIN(LOW, 9)
    high_low_range = ts_max(df['high'], 9) - ts_min(df['low'], 9)
    
    # Handle division by zero
    high_low_range[high_low_range == 0] = np.nan
    
    # Calculate the main expression
    main_expr = (close_min_low_diff / high_low_range) * 100
    
    # Calculate SMA with parameters (3, 1)
    result = sma(main_expr, 3, 1)
    
    return pd.Series(result, index=df.index, name='alpha_057')

def alpha057(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_057, code, benchmark, end_date, lookback)
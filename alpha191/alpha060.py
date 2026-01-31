import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_060(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha060 factor.
    Formula: SUM(((CLOSE-LOW)-(HIGH-CLOSE))./(HIGH-LOW).*VOLUME,20)
    """
    # Calculate (CLOSE - LOW) - (HIGH - CLOSE)
    numerator = (df['close'] - df['low']) - (df['high'] - df['close'])
    
    # Calculate (HIGH - LOW)
    denominator = df['high'] - df['low']
    
    # Handle division by zero
    denominator[denominator == 0] = np.nan
    
    # Calculate the main expression
    main_expr = (numerator / denominator) * df['volume']
    
    # Calculate 20-period sum
    result = ts_sum(main_expr, 20)
    
    return pd.Series(result, index=df.index, name='alpha_060')

def alpha060(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_060, code, benchmark, end_date, lookback)
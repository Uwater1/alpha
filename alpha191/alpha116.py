import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_116(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha116 factor.
    Formula: REGBETA(CLOSE,SEQUENCE,20)
    """
    # Extract values as numpy arrays
    close = df['close'].values
    
    # Calculate sequence
    sequence = np.arange(len(close))
    
    # Calculate regression beta
    result = regression_beta(close, sequence, 20)
    
    return pd.Series(result, index=df.index, name='alpha_116')

def alpha116(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_116, code, benchmark, end_date, lookback)
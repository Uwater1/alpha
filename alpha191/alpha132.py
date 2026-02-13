import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_132(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha132 factor (inverted).
    Formula: MEAN(AMOUNT,20)
    """
    # Extract values as numpy arrays
    amount = df['amount'].values
    
    # Calculate -MEAN(AMOUNT,20)
    result = -ts_mean(amount, 20)
    
    return pd.Series(result, index=df.index, name='alpha_132')

def alpha132(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_132, code, benchmark, end_date, lookback)
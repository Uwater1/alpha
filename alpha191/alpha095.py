import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_095(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha095 factor.
    Formula: STD(AMOUNT,20)
    """
    # Calculate STD(AMOUNT,20)
    result = ts_std(df['amount'], 20)
    
    return pd.Series(result, index=df.index, name='alpha_095')

def alpha095(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_095, code, benchmark, end_date, lookback)
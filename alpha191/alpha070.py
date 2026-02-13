import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_070(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha070 factor (inverted).
    Formula: -STD(AMOUNT,6)
    """
    # Calculate -STD(AMOUNT, 6)
    result = -ts_std(df['amount'], 6)
    
    return pd.Series(result, index=df.index, name='alpha_070')

def alpha070(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_070, code, benchmark, end_date, lookback)
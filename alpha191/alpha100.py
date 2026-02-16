import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_100(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha100 factor.
    Formula: STD(VOLUME,20)
    """
    # Calculate STD(VOLUME,20)
    result = ts_std(df['volume'], 20)
    
    return pd.Series(result, index=df.index, name='alpha_100')

def alpha100(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_100, code, benchmark, end_date, lookback)
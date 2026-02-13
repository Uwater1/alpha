import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_097(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha097 factor (inverted).
    Formula: -STD(VOLUME,10)
    """
    # Calculate -STD(VOLUME,10)
    result = -ts_std(df['volume'], 10)
    
    return pd.Series(result, index=df.index, name='alpha_097')

def alpha097(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_097, code, benchmark, end_date, lookback)
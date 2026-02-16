import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_081(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha081 factor.
    Formula: SMA(VOLUME,21,2)
    """
    volume = df['volume'].values
    
    # Calculate SMA(VOLUME,21,2)
    result = sma(volume, 21, 2)
    
    return pd.Series(result, index=df.index, name='alpha_081')

def alpha081(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_081, code, benchmark, end_date, lookback)
import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_124(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha124 factor.
    Formula: (CLOSE-VWAP)/DECAYLINEAR(RANK(TSMAX(CLOSE,30)),2)
    """
    # Extract values as numpy arrays
    close = df['close'].values
    vwap = df['vwap'].values
    
    # Calculate close minus VWAP
    close_minus_vwap = close - vwap
    
    # Calculate TSMAX of close
    tsmax_close = ts_max(close, 30)
    
    # Calculate rank of TSMAX of close
    rank_tsmax_close = rank(tsmax_close)
    
    # Calculate decay linear of rank of TSMAX of close
    decay_linear_rank_tsmax_close = decay_linear(rank_tsmax_close, 2)
    
    # Calculate final result
    result = close_minus_vwap / decay_linear_rank_tsmax_close
    
    return pd.Series(result, index=df.index, name='alpha_124')

def alpha124(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_124, code, benchmark, end_date, lookback)
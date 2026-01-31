import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_120(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha120 factor.
    Formula: (RANK((VWAP-CLOSE))/RANK((VWAP+CLOSE)))
    """
    # Extract values as numpy arrays
    vwap = df['vwap'].values
    close = df['close'].values
    
    # Calculate VWAP minus close
    vwap_minus_close = vwap - close
    
    # Calculate VWAP plus close
    vwap_plus_close = vwap + close
    
    # Calculate rank of VWAP minus close
    rank_vwap_minus_close = rank(vwap_minus_close)
    
    # Calculate rank of VWAP plus close
    rank_vwap_plus_close = rank(vwap_plus_close)
    
    # Protect against division by zero
    denom = rank_vwap_plus_close
    denom[denom == 0] = np.nan
    
    # Calculate final result
    result = rank_vwap_minus_close / denom
    
    return pd.Series(result, index=df.index, name='alpha_120')

def alpha120(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_120, code, benchmark, end_date, lookback)
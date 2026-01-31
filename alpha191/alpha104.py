import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_104(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha104 factor.
    Formula: (-1*(DELTA(CORR(HIGH,VOLUME,5),5)*RANK(STD(CLOSE,20))))
    """
    # Extract values as numpy arrays
    high = df['high'].values
    volume = df['volume'].values
    close = df['close'].values
    
    # Calculate correlation between high and volume
    corr_high_volume = rolling_corr(high, volume, 5)
    
    # Calculate delta of correlation
    delta_corr = delta(corr_high_volume, 5)
    
    # Calculate standard deviation of close
    std_close = ts_std(close, 20)
    
    # Calculate rank of standard deviation
    rank_std_close = rank(std_close)
    
    # Calculate final result
    result = -1 * delta_corr * rank_std_close
    
    return pd.Series(result, index=df.index, name='alpha_104')

def alpha104(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_104, code, benchmark, end_date, lookback)
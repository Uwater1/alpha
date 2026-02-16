import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_122(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha122 factor.
    Formula: (SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2)-DELAY(SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2),1))/DELAY(SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2),1)
    """
    # Extract values as numpy arrays
    close = df['close'].values
    
    # Calculate log of close
    log_close = np.log(close)
    
    # Calculate SMA of log of close
    sma_log_close = sma(log_close, 13, 2)
    
    # Calculate SMA of SMA of log of close
    sma_sma_log_close = sma(sma_log_close, 13, 2)
    
    # Calculate SMA of SMA of SMA of log of close
    sma_sma_sma_log_close = sma(sma_sma_log_close, 13, 2)
    
    # Calculate delay of SMA of SMA of SMA of log of close
    delay_sma_sma_sma_log_close = delay(sma_sma_sma_log_close, 1)
    
    # Calculate final result
    result = (sma_sma_sma_log_close - delay_sma_sma_sma_log_close) / delay_sma_sma_sma_log_close
    
    return pd.Series(result, index=df.index, name='alpha_122')

def alpha122(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_122, code, benchmark, end_date, lookback)
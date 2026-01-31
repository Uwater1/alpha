import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_170(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha170 factor.
    Formula: ((((RANK((1/CLOSE))_VOLUME)/MEAN(VOLUME,20))_((HIGH*RANK((HIGH-CLOSE)))/(SUM(HIGH,5)/5)))-RANK((VWAP-DELAY(VWAP,5))))
    """
    # Extract values as numpy arrays
    close = df['close'].values
    volume = df['volume'].values
    high = df['high'].values
    vwap = df['vwap'].values
    
    # Calculate 1/CLOSE
    inv_close = 1 / close
    
    # Calculate RANK((1/CLOSE))
    rank_inv_close = rank(inv_close)
    
    # Calculate RANK((1/CLOSE))*VOLUME
    product1 = rank_inv_close * volume
    
    # Calculate MEAN(VOLUME,20)
    mean_volume_20 = ts_mean(volume, 20)
    
    # Calculate (RANK((1/CLOSE))*VOLUME)/MEAN(VOLUME,20)
    ratio1 = product1 / mean_volume_20
    
    # Calculate HIGH-CLOSE
    high_close_diff = high - close
    
    # Calculate RANK((HIGH-CLOSE))
    rank_high_close = rank(high_close_diff)
    
    # Calculate HIGH*RANK((HIGH-CLOSE))
    product2 = high * rank_high_close
    
    # Calculate SUM(HIGH,5)/5
    sum_high_5 = ts_sum(high, 5)
    mean_high_5 = sum_high_5 / 5
    
    # Calculate ((HIGH*RANK((HIGH-CLOSE)))/(SUM(HIGH,5)/5))
    ratio2 = product2 / mean_high_5
    
    # Calculate (RANK((1/CLOSE))*VOLUME)/MEAN(VOLUME,20))*((HIGH*RANK((HIGH-CLOSE)))/(SUM(HIGH,5)/5))
    product3 = ratio1 * ratio2
    
    # Calculate DELAY(VWAP,5)
    delay_vwap = delay(vwap, 5)
    
    # Calculate VWAP-DELAY(VWAP,5)
    vwap_diff = vwap - delay_vwap
    
    # Calculate RANK((VWAP-DELAY(VWAP,5)))
    rank_vwap_diff = rank(vwap_diff)
    
    # Calculate final result: product3 - RANK(...)
    result = product3 - rank_vwap_diff
    
    return pd.Series(result, index=df.index, name='alpha_170')

def alpha170(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_170, code, benchmark, end_date, lookback)
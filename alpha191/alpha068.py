import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_068(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha068 factor.
    Formula: SMA(((HIGH+LOW)/2-(DELAY(HIGH,1)+DELAY(LOW,1))/2)*(HIGH-LOW)/VOLUME,15,2)
    """
    # Calculate (HIGH + LOW) / 2
    high_low_avg = (df['high'] + df['low']) / 2
    
    # Calculate DELAY(HIGH, 1) and DELAY(LOW, 1)
    delayed_high = delay(df['high'], 1)
    delayed_low = delay(df['low'], 1)
    
    # Calculate (DELAY(HIGH, 1) + DELAY(LOW, 1)) / 2
    delayed_avg = (delayed_high + delayed_low) / 2
    
    # Calculate (HIGH + LOW) / 2 - (DELAY(HIGH, 1) + DELAY(LOW, 1)) / 2
    diff = high_low_avg - delayed_avg
    
    # Calculate (HIGH - LOW) / VOLUME
    high_low_volume_ratio = (df['high'] - df['low']) / df['volume']
    
    # Calculate the main expression
    main_expr = diff * high_low_volume_ratio
    
    # Calculate SMA with parameters (15, 2)
    result = sma(main_expr, 15, 2)
    
    return pd.Series(result, index=df.index, name='alpha_068')

def alpha068(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_068, code, benchmark, end_date, lookback)
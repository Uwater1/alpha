import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_080(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha080 factor.
    Formula: (VOLUME-DELAY(VOLUME,5))/DELAY(VOLUME,5)*100
    """
    volume = df['volume'].values
    
    # Calculate (VOLUME-DELAY(VOLUME,5))/DELAY(VOLUME,5)*100
    delayed_volume = delay(volume, 5)
    
    # Handle division by zero
    delayed_volume[delayed_volume == 0] = np.nan
    result = (volume - delayed_volume) / delayed_volume * 100
    
    return pd.Series(result, index=df.index, name='alpha_080')

def alpha080(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_080, code, benchmark, end_date, lookback)
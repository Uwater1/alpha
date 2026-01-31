import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_156(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha156 factor.
    Formula: (MAX(RANK(DECAYLINEAR(DELTA(VWAP,5),3)),RANK(DECAYLINEAR(((DELTA(((OPEN*0.15)+(LOW*0.85)),2)/((OPEN*0.15)+(LOW*0.85)))*-1),3)))*-1)
    """
    # Extract values as numpy arrays
    vwap = df['vwap'].values
    open_price = df['open'].values
    low = df['low'].values
    
    # Calculate DELTA(VWAP,5)
    delta_vwap = delta(vwap, 5)
    
    # Calculate DECAYLINEAR(DELTA(VWAP,5),3)
    decay_vwap = decay_linear(delta_vwap, 3)
    
    # Calculate (OPEN*0.15)+(LOW*0.85)
    weighted_price = open_price * 0.15 + low * 0.85
    
    # Calculate DELTA(((OPEN*0.15)+(LOW*0.85)),2)
    delta_weighted = delta(weighted_price, 2)
    
    # Calculate DELTA(((OPEN*0.15)+(LOW*0.85)),2)/((OPEN*0.15)+(LOW*0.85))
    ratio = delta_weighted / weighted_price
    
    # Calculate DECAYLINEAR(((DELTA(((OPEN*0.15)+(LOW*0.85)),2)/((OPEN*0.15)+(LOW*0.85)))*-1),3)
    decay_weighted = decay_linear(-ratio, 3)
    
    # Calculate ranks
    rank_vwap = rank(decay_vwap)
    rank_weighted = rank(decay_weighted)
    
    # Calculate MAX(RANK(DECAYLINEAR(DELTA(VWAP,5),3)),RANK(DECAYLINEAR(...)))
    max_rank = np.maximum(rank_vwap, rank_weighted)
    
    # Calculate final result: MAX(...) * -1
    result = max_rank * -1
    
    return pd.Series(result, index=df.index, name='alpha_156')

def alpha156(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_156, code, benchmark, end_date, lookback)
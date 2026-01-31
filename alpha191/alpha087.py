import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_087(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha087 factor.
    Formula: ((RANK(DECAYLINEAR(DELTA(VWAP,4),7))+TSRANK(DECAYLINEAR(((((LOW*0.9)+(LOW*0.1))-VWAP)/(OPEN-((HIGH+LOW)/2))),11),7))*-1)
    """
    # Calculate VWAP if not available
    if 'vwap' in df.columns:
        vwap = df['vwap'].values
    else:
        need_ohlc = True
        if {'amount', 'volume'}.issubset(df.columns):
            vwap_s = df['amount'] / df['volume'].replace(0, np.nan)
            valid = df['amount'].ne(0) & df['volume'].ne(0) & vwap_s.notna() & vwap_s.between(df['low'], df['high'])
            need_ohlc = ~valid.all()
            if not need_ohlc:
                vwap = vwap_s.values

        if need_ohlc:
            ohlc_avg = (df['open'] + df['high'] + df['low'] + df['close']) / 4
            if 'valid' in locals():
                vwap = vwap_s.where(valid, ohlc_avg).values
            else:
                vwap = ohlc_avg.values
    
    low = df['low'].values
    open_price = df['open'].values
    high = df['high'].values
    
    # Calculate first part: RANK(DECAYLINEAR(DELTA(VWAP,4),7))
    delta_vwap = delta(vwap, 4)
    decay1 = decay_linear(delta_vwap, 7)
    rank1 = rank(decay1)
    
    # Calculate second part: TSRANK(DECAYLINEAR(((((LOW*0.9)+(LOW*0.1))-VWAP)/(OPEN-((HIGH+LOW)/2))),11),7)
    weighted_low = (low * 0.9) + (low * 0.1)
    numerator = weighted_low - vwap
    denominator = open_price - ((high + low) / 2)
    
    # Handle division by zero
    denominator[denominator == 0] = np.nan
    ratio = numerator / denominator
    
    decay2 = decay_linear(ratio, 11)
    tsrank2 = ts_rank(decay2, 7)
    
    # Calculate final result
    result = (rank1 + tsrank2) * -1
    
    return pd.Series(result, index=df.index, name='alpha_087')

def alpha087(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_087, code, benchmark, end_date, lookback)
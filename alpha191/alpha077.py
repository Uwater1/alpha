import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_077(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha077 factor.
    Formula: MIN(RANK(DECAYLINEAR(((((HIGH+LOW)/2)+HIGH)-(VWAP+HIGH)),20)),RANK(DECAYLINEAR(CORR(((HIGH+LOW)/2),MEAN(VOLUME,40),3),6)))
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
    
    high = df['high'].values
    low = df['low'].values
    volume = df['volume'].values
    
    # Calculate first part: RANK(DECAYLINEAR(((((HIGH+LOW)/2)+HIGH)-(VWAP+HIGH)),20))
    hl_avg = (high + low) / 2
    term1 = (((hl_avg + high) - (vwap + high)))
    decay1 = decay_linear(term1, 20)
    rank1 = rank(decay1)
    
    # Calculate second part: RANK(DECAYLINEAR(CORR(((HIGH+LOW)/2),MEAN(VOLUME,40),3),6))
    mean_volume_40 = ts_mean(volume, 40)
    corr = rolling_corr(hl_avg, mean_volume_40, 3)
    decay2 = decay_linear(corr, 6)
    rank2 = rank(decay2)
    
    result = np.minimum(rank1, rank2)
    
    return pd.Series(result, index=df.index, name='alpha_077')

def alpha077(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_077, code, benchmark, end_date, lookback)
import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_073(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha073 factor.
    Formula: ((TSRANK(DECAYLINEAR(DECAYLINEAR(CORR((CLOSE),VOLUME,10),16),4),5)-RANK(DECAYLINEAR(CORR(VWAP,MEAN(VOLUME,30),4),3)))*-1)
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
    
    close = df['close'].values
    volume = df['volume'].values
    
    # Calculate first part: TSRANK(DECAYLINEAR(DECAYLINEAR(CORR((CLOSE),VOLUME,10),16),4),5)
    corr_close_volume = rolling_corr(close, volume, 10)
    decay1 = decay_linear(corr_close_volume, 16)
    decay2 = decay_linear(decay1, 4)
    tsrank_result = ts_rank(decay2, 5)
    
    # Calculate second part: RANK(DECAYLINEAR(CORR(VWAP,MEAN(VOLUME,30),4),3))
    mean_volume_30 = ts_mean(volume, 30)
    corr_vwap_volume = rolling_corr(vwap, mean_volume_30, 4)
    decay3 = decay_linear(corr_vwap_volume, 3)
    rank_result = rank(decay3)
    
    result = (tsrank_result - rank_result) * -1
    
    return pd.Series(result, index=df.index, name='alpha_073')

def alpha073(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_073, code, benchmark, end_date, lookback)
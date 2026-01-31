import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_090(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha090 factor.
    Formula: (RANK(CORR(RANK(VWAP),RANK(VOLUME),5))*-1)
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
    
    volume = df['volume'].values
    
    # Calculate RANK(VWAP)
    rank_vwap = rank(vwap)
    
    # Calculate RANK(VOLUME)
    rank_volume = rank(volume)
    
    # Calculate CORR(RANK(VWAP),RANK(VOLUME),5)
    corr = rolling_corr(rank_vwap, rank_volume, 5)
    
    # Calculate RANK(CORR(...))
    rank_corr = rank(corr)
    
    # Calculate -1 * RANK(...)
    result = -1 * rank_corr
    
    return pd.Series(result, index=df.index, name='alpha_090')

def alpha090(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_090, code, benchmark, end_date, lookback)
import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_074(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha074 factor (inverted).
    Formula: -(RANK(CORR(SUM(((LOW*0.35)+(VWAP*0.65)),20),SUM(MEAN(VOLUME,40),20),7))+RANK(CORR(RANK(VWAP),RANK(VOLUME),6)))
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
    volume = df['volume'].values
    
    # Calculate first part: RANK(CORR(SUM(((LOW*0.35)+(VWAP*0.65)),20),SUM(MEAN(VOLUME,40),20),7))
    weighted_sum = (low * 0.35) + (vwap * 0.65)
    sum_weighted = ts_sum(weighted_sum, 20)
    mean_volume_40 = ts_mean(volume, 40)
    sum_mean_volume = ts_sum(mean_volume_40, 20)
    corr1 = rolling_corr(sum_weighted, sum_mean_volume, 7)
    rank1 = rank(corr1)
    
    # Calculate second part: RANK(CORR(RANK(VWAP),RANK(VOLUME),6))
    rank_vwap = rank(vwap)
    rank_volume = rank(volume)
    corr2 = rolling_corr(rank_vwap, rank_volume, 6)
    rank2 = rank(corr2)
    
    result = -(rank1 + rank2)
    
    return pd.Series(result, index=df.index, name='alpha_074')

def alpha074(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_074, code, benchmark, end_date, lookback)
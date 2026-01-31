import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_092(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha092 factor.
    Formula: (MAX(RANK(DECAYLINEAR(DELTA(((CLOSE*0.35)+(VWAP*0.65)),2),3)),TSRANK(DECAYLINEAR(ABS(CORR((MEAN(VOLUME,180)),CLOSE,13)),5),15))*-1)
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
    
    # Calculate first part: RANK(DECAYLINEAR(DELTA(((CLOSE*0.35)+(VWAP*0.65)),2),3))
    weighted_sum = (close * 0.35) + (vwap * 0.65)
    delta_weighted = delta(weighted_sum, 2)
    decay1 = decay_linear(delta_weighted, 3)
    rank1 = rank(decay1)
    
    # Calculate second part: TSRANK(DECAYLINEAR(ABS(CORR((MEAN(VOLUME,180)),CLOSE,13)),5),15)
    mean_volume_180 = ts_mean(volume, 180)
    corr = rolling_corr(mean_volume_180, close, 13)
    abs_corr = np.abs(corr)
    decay2 = decay_linear(abs_corr, 5)
    tsrank2 = ts_rank(decay2, 15)
    
    # Calculate MAX and final result
    result = np.maximum(rank1, tsrank2) * -1
    
    return pd.Series(result, index=df.index, name='alpha_092')

def alpha092(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_092, code, benchmark, end_date, lookback)
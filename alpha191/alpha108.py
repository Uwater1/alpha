import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_108(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha108 factor.
    Formula: ((RANK((HIGH-MIN(HIGH,2)))^RANK(CORR((VWAP),(MEAN(VOLUME,120)),6))) *-1)
    """
    # Extract values as numpy arrays
    high = df['high'].values
    volume = df['volume'].values
    
    # Calculate VWAP if available, otherwise use OHLC average
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
    
    # Calculate minimum of high over 2 days
    min_high_2 = ts_min(high, 2)
    
    # Calculate rank of high minus minimum of high
    rank_high_minus_min_high = rank(high - min_high_2)
    
    # Calculate mean volume over 120 days
    mean_volume_120 = ts_mean(volume, 120)
    
    # Calculate correlation between VWAP and mean volume
    corr_vwap_mean_volume = rolling_corr(vwap, mean_volume_120, 6)
    
    # Calculate rank of correlation
    rank_corr_vwap_mean_volume = rank(corr_vwap_mean_volume)
    
    # Calculate final result
    result = (rank_high_minus_min_high ** rank_corr_vwap_mean_volume) * -1
    
    return pd.Series(result, index=df.index, name='alpha_108')

def alpha108(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_108, code, benchmark, end_date, lookback)
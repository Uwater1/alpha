import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_101(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha101 factor.
    Formula: ((RANK(CORR(CLOSE,SUM(MEAN(VOLUME,30),37),15))<RANK(CORR(RANK(((HIGH+LOW)/2)),RANK(VOLUME),12)))*-1)
    """
    # Extract values as numpy arrays
    close = df['close'].values
    volume = df['volume'].values
    high = df['high'].values
    low = df['low'].values
    
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
    
    # Calculate mean volume over 30 days
    mean_volume_30 = ts_mean(volume, 30)
    
    # Calculate sum of mean volume over 37 days
    sum_mean_volume_37 = ts_sum(mean_volume_30, 37)
    
    # Calculate correlation between close and sum of mean volume
    corr_close_volume = rolling_corr(close, sum_mean_volume_37, 15)
    
    # Calculate rank of correlation
    rank_corr_close_volume = rank(corr_close_volume)
    
    # Calculate average of high and low
    avg_high_low = (high + low) / 2
    
    # Calculate rank of average high and low
    rank_avg_high_low = rank(avg_high_low)
    
    # Calculate rank of volume
    rank_volume = rank(volume)
    
    # Calculate correlation between rank of average high and low and rank of volume
    corr_rank_high_low_volume = rolling_corr(rank_avg_high_low, rank_volume, 12)
    
    # Calculate rank of correlation
    rank_corr_rank_high_low_volume = rank(corr_rank_high_low_volume)
    
    # Calculate final result
    result = np.where(rank_corr_close_volume < rank_corr_rank_high_low_volume, -1, 0)
    
    return pd.Series(result, index=df.index, name='alpha_101')

def alpha101(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_101, code, benchmark, end_date, lookback)
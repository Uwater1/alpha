import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_114(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha114 factor.
    Formula: ((RANK(DELAY(((HIGH-LOW)/(SUM(CLOSE,5)/5)),2))*RANK(RANK(VOLUME)))/(((HIGH-LOW)/(SUM(CLOSE,5)/5))/(VWAP-CLOSE)))
    """
    # Extract values as numpy arrays
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
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
    
    # Calculate high minus low
    high_minus_low = high - low
    
    # Calculate sum of close
    sum_close = ts_sum(close, 5)
    
    # Calculate ratio
    ratio = high_minus_low / (sum_close / 5)
    
    # Calculate delay of ratio
    delay_ratio = delay(ratio, 2)
    
    # Calculate rank of delay of ratio
    rank_delay_ratio = rank(delay_ratio)
    
    # Calculate rank of volume
    rank_volume = rank(volume)
    
    # Calculate rank of rank of volume
    rank_rank_volume = rank(rank_volume)
    
    # Calculate denominator
    denominator = ratio / (vwap - close)
    
    # Calculate final result
    result = (rank_delay_ratio * rank_rank_volume) / denominator
    
    return pd.Series(result, index=df.index, name='alpha_114')

def alpha114(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_114, code, benchmark, end_date, lookback)
import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_061(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha061 factor.
    Formula: (MAX(RANK(DECAYLINEAR(DELTA(VWAP,1),12)),RANK(DECAYLINEAR(RANK(CORR((LOW),MEAN(VOLUME,80),8)),17)))*-1)
    """
    # Calculate VWAP if not present
    if 'vwap' not in df.columns:
        need_ohlc = False
        if {'amount', 'volume'}.issubset(df.columns):
            vwap_calc = df['amount'] / df['volume'].replace(0, np.nan)
            valid = (
                df['amount'].ne(0) & df['volume'].ne(0) & vwap_calc.notna() & vwap_calc.between(df['low'], df['high'])
            )
            need_ohlc = ~valid.all()
            vwap = vwap_calc
        else:
            need_ohlc = True

        if need_ohlc:
            if not {'open', 'high', 'low', 'close'}.issubset(df.columns):
                raise ValueError(
                    "VWAP column not found and cannot be approximated (missing 'open', 'high', 'low', or 'close' columns)"
                )
            ohlc_avg = (df['open'] + df['high'] + df['low'] + df['close']) / 4
            if 'valid' in locals():
                vwap = vwap.where(valid, ohlc_avg)
            else:
                vwap = ohlc_avg
    else:
        vwap = df['vwap']
    
    # Calculate DELTA(VWAP, 1)
    vwap_delta = delta(vwap, 1)
    
    # Calculate DECAYLINEAR(DELTA(VWAP, 1), 12)
    decay_vwap = decay_linear(vwap_delta, 12)
    
    # Calculate RANK(DECAYLINEAR(DELTA(VWAP, 1), 12))
    rank_decay_vwap = rank(decay_vwap)
    
    # Calculate MEAN(VOLUME, 80)
    mean_volume = ts_mean(df['volume'], 80)
    
    # Calculate CORR(LOW, MEAN(VOLUME, 80), 8)
    correlation = rolling_corr(df['low'], mean_volume, 8)
    
    # Calculate RANK(CORR(LOW, MEAN(VOLUME, 80), 8))
    rank_corr = rank(correlation)
    
    # Calculate DECAYLINEAR(RANK(CORR(LOW, MEAN(VOLUME, 80), 8)), 17)
    decay_rank_corr = decay_linear(rank_corr, 17)
    
    # Calculate RANK(DECAYLINEAR(RANK(CORR(LOW, MEAN(VOLUME, 80), 8)), 17))
    rank_decay_rank_corr = rank(decay_rank_corr)
    
    # Calculate MAX of the two ranks
    max_rank = np.maximum(rank_decay_vwap, rank_decay_rank_corr)
    
    # Apply negative sign
    result = -1 * max_rank
    
    return pd.Series(result, index=df.index, name='alpha_061')

def alpha061(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_061, code, benchmark, end_date, lookback)
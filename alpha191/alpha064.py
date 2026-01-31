import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_064(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha064 factor.
    Formula: (MAX(RANK(DECAYLINEAR(CORR(RANK(VWAP),RANK(VOLUME),4),4)),RANK(DECAYLINEAR(MAX(CORR(RANK(OPEN),RANK(VOLUME),4),RANK(OPEN)),4)))*-1)
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
    
    # Calculate RANK(VWAP)
    ranked_vwap = rank(vwap)
    
    # Calculate RANK(VOLUME)
    ranked_volume = rank(df['volume'])
    
    # Calculate CORR(RANK(VWAP), RANK(VOLUME), 4)
    corr_vwap_volume = rolling_corr(ranked_vwap, ranked_volume, 4)
    
    # Calculate DECAYLINEAR(CORR(RANK(VWAP), RANK(VOLUME), 4), 4)
    decay_corr_vwap_volume = decay_linear(corr_vwap_volume, 4)
    
    # Calculate RANK(DECAYLINEAR(CORR(RANK(VWAP), RANK(VOLUME), 4), 4))
    rank_decay_corr_vwap_volume = rank(decay_corr_vwap_volume)
    
    # Calculate MEAN(VOLUME, 60)
    mean_volume = ts_mean(df['volume'], 60)
    
    # Calculate RANK(CLOSE)
    ranked_close = rank(df['close'])
    
    # Calculate RANK(MEAN(VOLUME, 60))
    ranked_mean_volume = rank(mean_volume)
    
    # Calculate CORR(RANK(CLOSE), RANK(MEAN(VOLUME, 60)), 4)
    corr_close_mean_volume = rolling_corr(ranked_close, ranked_mean_volume, 4)
    
    # Calculate MAX(CORR(RANK(CLOSE), RANK(MEAN(VOLUME, 60)), 4), 13)
    max_corr = np.maximum(corr_close_mean_volume, 13)
    
    # Calculate DECAYLINEAR(MAX(CORR(RANK(CLOSE), RANK(MEAN(VOLUME, 60)), 4), 13), 14)
    decay_max_corr = decay_linear(max_corr, 14)
    
    # Calculate RANK(DECAYLINEAR(MAX(CORR(RANK(CLOSE), RANK(MEAN(VOLUME, 60)), 4), 13), 14))
    rank_decay_max_corr = rank(decay_max_corr)
    
    # Calculate MAX of the two ranks
    max_rank = np.maximum(rank_decay_corr_vwap_volume, rank_decay_max_corr)
    
    # Apply negative sign
    result = -1 * max_rank
    
    return pd.Series(result, index=df.index, name='alpha_064')

def alpha064(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_064, code, benchmark, end_date, lookback)
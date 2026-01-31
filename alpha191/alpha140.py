import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_140(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha140 factor.
    Formula: MIN(RANK(DECAYLINEAR(((RANK(OPEN)+RANK(LOW))-(RANK(HIGH)+RANK(CLOSE))),8)),TSRANK(DECAYLINEAR(CORR(TSRANK(CLOSE,8),TSRANK(MEAN(VOLUME,60),20),8),7),3))
    """
    # Extract values as numpy arrays
    open_price = df['open'].values
    low = df['low'].values
    high = df['high'].values
    close = df['close'].values
    volume = df['volume'].values
    
    # Calculate rank of open
    rank_open = rank(open_price)
    
    # Calculate rank of low
    rank_low = rank(low)
    
    # Calculate rank of high
    rank_high = rank(high)
    
    # Calculate rank of close
    rank_close = rank(close)
    
    # Calculate rank of open plus rank of low minus rank of high minus rank of close
    rank_open_plus_rank_low_minus_rank_high_minus_rank_close = rank_open + rank_low - rank_high - rank_close
    
    # Calculate decay linear of rank
    decay_linear_rank = decay_linear(rank_open_plus_rank_low_minus_rank_high_minus_rank_close, 8)
    
    # Calculate rank of decay linear of rank
    rank_decay_linear_rank = rank(decay_linear_rank)
    
    # Calculate TSRANK of close
    tsrank_close = ts_rank(close, 8)
    
    # Calculate mean volume
    mean_volume = ts_mean(volume, 60)
    
    # Calculate TSRANK of mean volume
    tsrank_mean_volume = ts_rank(mean_volume, 20)
    
    # Calculate correlation between TSRANK of close and TSRANK of mean volume
    corr_tsrank_close_tsrank_mean_volume = rolling_corr(tsrank_close, tsrank_mean_volume, 8)
    
    # Calculate decay linear of correlation
    decay_linear_corr = decay_linear(corr_tsrank_close_tsrank_mean_volume, 7)
    
    # Calculate TSRANK of decay linear of correlation
    tsrank_decay_linear_corr = ts_rank(decay_linear_corr, 3)
    
    # Calculate final result
    result = np.minimum(rank_decay_linear_rank, tsrank_decay_linear_corr)
    
    return pd.Series(result, index=df.index, name='alpha_140')

def alpha140(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_140, code, benchmark, end_date, lookback)
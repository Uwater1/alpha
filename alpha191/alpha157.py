import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_157(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha157 factor.
    Formula: (MIN(PROD(RANK(RANK(LOG(SUM(TSMIN(RANK(RANK((-1*RANK(DELTA((CLOSE-1),5))))),2),1)))),1),5) +TSRANK(DELAY((-1*RET),6),5))
    """
    # Extract values as numpy arrays
    close = df['close'].values
    
    # Calculate DELTA((CLOSE-1),5)
    close_minus_1 = close - 1
    delta_close = delta(close_minus_1, 5)
    
    # Calculate -1*RANK(DELTA((CLOSE-1),5))
    rank_delta = rank(delta_close)
    neg_rank_delta = -1 * rank_delta
    
    # Calculate RANK(RANK((-1*RANK(DELTA((CLOSE-1),5)))))
    rank_rank_neg = rank(neg_rank_delta)
    
    # Calculate TSMIN(RANK(RANK((-1*RANK(DELTA((CLOSE-1),5))))),2)
    min_rank_rank = ts_min(rank_rank_neg, 2)
    
    # Calculate SUM(TSMIN(RANK(RANK((-1*RANK(DELTA((CLOSE-1),5))))),2),1)
    sum_min_rank_rank = ts_sum(min_rank_rank, 1)
    
    # Protect against log of zero/negative values
    log_input = sum_min_rank_rank.copy()
    log_input[log_input <= 0] = np.nan
    
    # Calculate LOG(SUM(...))
    log_sum = np.log(log_input)
    
    # Calculate RANK(LOG(SUM(...)))
    rank_log_sum = rank(log_sum)
    
    # Calculate RANK(RANK(LOG(SUM(...))))
    rank_rank_log_sum = rank(rank_log_sum)
    
    # Calculate PROD(RANK(RANK(LOG(SUM(...)))),1)
    # Since window is 1, product is just the value itself
    prod_result = rank_rank_log_sum
    
    # Calculate MIN(PROD(...),5)
    min_prod = ts_min(prod_result, 5)
    
    # Calculate RET
    ret = compute_ret(close)
    
    # Calculate -1*RET
    neg_ret = -1 * ret
    
    # Calculate DELAY((-1*RET),6)
    delay_neg_ret = delay(neg_ret, 6)
    
    # Calculate TSRANK(DELAY((-1*RET),6),5)
    tsrank_delay = ts_rank(delay_neg_ret, 5)
    
    # Calculate final result: MIN(...) + TSRANK(...)
    result = min_prod + tsrank_delay
    
    return pd.Series(result, index=df.index, name='alpha_157')

def alpha157(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_157, code, benchmark, end_date, lookback)
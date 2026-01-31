import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_186(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha186 factor.
    Formula: (MEAN(ABS(SUM((LD>0&LD>HD)?LD:0,14)*100/SUM(TR,14)-SUM((HD>0&HD>LD)?HD:0,14)*100/SUM(TR,14))/(SUM((LD>0&LD>HD)?LD:0,14)*100/SUM(TR,14)+SUM((HD>0&HD>LD)?HD:0,14)*100/SUM(TR,14))*100,6)+DELAY(MEAN(ABS(SUM((LD>0&LD>HD)?LD:0,14)*100/SUM(TR,14)-SUM((HD>0&HD>LD)?HD:0,14)*100/SUM(TR,14))/(SUM((LD>0&LD>HD)?LD:0,14)*100/SUM(TR,14)+SUM((HD>0&HD>LD)?HD:0,14)*100/SUM(TR,14))*100,6),6))/2
    """
    # Extract values as numpy arrays
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    
    # Calculate HD = (OPEN>=DELAY(OPEN,1)?0:MAX((OPEN-LOW),(OPEN-DELAY(OPEN,1))))
    # Note: We need OPEN data for this calculation
    open_price = df['open'].values
    delay_open = delay(open_price, 1)
    
    # Calculate HD
    open_ge_delay = open_price >= delay_open
    open_low_diff = open_price - low
    open_delay_diff = open_price - delay_open
    hd = np.where(open_ge_delay, 0, np.maximum(open_low_diff, open_delay_diff))
    
    # Calculate LD = (OPEN<=DELAY(OPEN,1)?0:MAX((DELAY(OPEN,1)-LOW),(OPEN-DELAY(OPEN,1))))
    open_le_delay = open_price <= delay_open
    delay_open_low_diff = delay_open - low
    ld = np.where(open_le_delay, 0, np.maximum(delay_open_low_diff, open_delay_diff))
    
    # Calculate TR = MAX(MAX(HIGH-LOW,ABS(HIGH-DELAY(CLOSE,1))),ABS(LOW-DELAY(CLOSE,1)))
    delay_close = delay(close, 1)
    high_low_diff = high - low
    high_delay_close_diff = np.abs(high - delay_close)
    low_delay_close_diff = np.abs(low - delay_close)
    tr = np.maximum(np.maximum(high_low_diff, high_delay_close_diff), low_delay_close_diff)
    
    # Calculate SUM((LD>0&LD>HD)?LD:0,14)
    ld_condition = (ld > 0) & (ld > hd)
    sum_ld = ts_sum(np.where(ld_condition, ld, 0), 14)
    
    # Calculate SUM((HD>0&HD>LD)?HD:0,14)
    hd_condition = (hd > 0) & (hd > ld)
    sum_hd = ts_sum(np.where(hd_condition, hd, 0), 14)
    
    # Calculate SUM(TR,14)
    sum_tr = ts_sum(tr, 14)
    
    # Calculate numerator: sum_ld*100/sum_tr - sum_hd*100/sum_tr
    numerator = sum_ld * 100 / sum_tr - sum_hd * 100 / sum_tr
    
    # Calculate denominator: sum_ld*100/sum_tr + sum_hd*100/sum_tr
    denominator = sum_ld * 100 / sum_tr + sum_hd * 100 / sum_tr
    
    # Calculate ratio
    ratio = np.abs(numerator) / denominator * 100
    
    # Calculate MEAN(...,6)
    mean_ratio = ts_mean(ratio, 6)
    
    # Calculate DELAY(MEAN(...,6),6)
    delay_mean_ratio = delay(mean_ratio, 6)
    
    # Calculate final result: (mean_ratio + delay_mean_ratio) / 2
    result = (mean_ratio + delay_mean_ratio) / 2
    
    return pd.Series(result, index=df.index, name='alpha_186')

def alpha186(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_186, code, benchmark, end_date, lookback)
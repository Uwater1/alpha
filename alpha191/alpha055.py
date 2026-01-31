import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_055(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha055 factor.
    Formula: SUM(16*(CLOSE-DELAY(CLOSE,1)+(CLOSE-OPEN)/2+DELAY(CLOSE,1)-DELAY(OPEN,1))/((ABS(HIGH-DELAY(CLOSE,1))>ABS(LOW-DELAY(CLOSE,1))&ABS(HIGH-DELAY(CLOSE,1))>ABS(HIGH-DELAY(LOW,1))?ABS(HIGH-DELAY(CLOSE,1))+ABS(LOW-DELAY(CLOSE,1))/2+ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4:(ABS(LOW-DELAY(CLOSE,1))>ABS(HIGH-DELAY(LOW,1))&ABS(LOW-DELAY(CLOSE,1))>ABS(HIGH-DELAY(CLOSE,1))?ABS(LOW-DELAY(CLOSE,1))+ABS(HIGH-DELAY(CLOSE,1))/2+ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4:ABS(HIGH-DELAY(LOW,1))+ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4)))*MAX(ABS(HIGH-DELAY(CLOSE,1)),ABS(LOW-DELAY(CLOSE,1))),20)
    """
    # Calculate DELAY values
    delayed_close = delay(df['close'], 1)
    delayed_open = delay(df['open'], 1)
    delayed_low = delay(df['low'], 1)
    
    # Calculate numerator: CLOSE-DELAY(CLOSE,1)+(CLOSE-OPEN)/2+DELAY(CLOSE,1)-DELAY(OPEN,1)
    numerator = (df['close'] - delayed_close + 
                 (df['close'] - df['open']) / 2 + 
                 delayed_close - delayed_open)
    
    # Calculate denominators for conditional logic
    high_close_diff = np.abs(df['high'] - delayed_close)
    low_close_diff = np.abs(df['low'] - delayed_close)
    high_low_diff = np.abs(df['high'] - delayed_low)
    
    # Calculate DELAY(CLOSE,1)-DELAY(OPEN,1)
    delayed_diff = delayed_close - delayed_open
    
    # Build conditional denominator using np.where
    # Condition 1: ABS(HIGH-DELAY(CLOSE,1)) > ABS(LOW-DELAY(CLOSE,1)) & ABS(HIGH-DELAY(CLOSE,1)) > ABS(HIGH-DELAY(LOW,1))
    condition1 = (high_close_diff > low_close_diff) & (high_close_diff > high_low_diff)
    
    # Condition 2: ABS(LOW-DELAY(CLOSE,1)) > ABS(HIGH-DELAY(LOW,1)) & ABS(LOW-DELAY(CLOSE,1)) > ABS(HIGH-DELAY(CLOSE,1))
    condition2 = (low_close_diff > high_low_diff) & (low_close_diff > high_close_diff)
    
    # Denominator calculation
    denominator = np.where(
        condition1,
        high_close_diff + low_close_diff / 2 + np.abs(delayed_diff) / 4,
        np.where(
            condition2,
            low_close_diff + high_close_diff / 2 + np.abs(delayed_diff) / 4,
            high_low_diff + np.abs(delayed_diff) / 4
        )
    )
    
    # Calculate MAX(ABS(HIGH-DELAY(CLOSE,1)), ABS(LOW-DELAY(CLOSE,1)))
    max_abs_diff = np.maximum(high_close_diff, low_close_diff)
    
    # Calculate the main expression
    main_expr = (16 * numerator / denominator) * max_abs_diff
    
    # Calculate 20-period sum
    result = ts_sum(main_expr, 20)
    
    return pd.Series(result, index=df.index, name='alpha_055')

def alpha055(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_055, code, benchmark, end_date, lookback)
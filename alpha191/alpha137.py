import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_137(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha137 factor.
    Formula: 16*(CLOSE-DELAY(CLOSE,1)+(CLOSE-OPEN)/2+DELAY(CLOSE,1)-DELAY(OPEN,1))/((ABS(HIGH-DELAY(CLOSE,1))>ABS(LOW-DELAY(CLOSE,1)) &ABS(HIGH-DELAY(CLOSE,1))>ABS(HIGH-DELAY(LOW,1))?ABS(HIGH-DELAY(CLOSE,1))+ABS(LOW-DELAY(CLOSE,1))/2+ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4:(ABS(LOW-DELAY(CLOSE,1))>ABS(HIGH-DELAY(LOW,1)) & ABS(LOW-DELAY(CLOSE,1))>ABS(HIGH-DELAY(CLOSE,1))?ABS(LOW-DELAY(CLOSE,1))+ABS(HIGH-DELAY(CLOSE,1))/2+ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4:ABS(HIGH-DELAY(LOW,1))+ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4)))*MAX(ABS(HIGH-DELAY(CLOSE,1)),ABS(LOW-DELAY(CLOSE,1)))
    """
    # Extract values as numpy arrays
    close = df['close'].values
    open_price = df['open'].values
    high = df['high'].values
    low = df['low'].values
    
    # Calculate delay of close
    delay_close = delay(close, 1)
    
    # Calculate delay of open
    delay_open = delay(open_price, 1)
    
    # Calculate delay of low
    delay_low = delay(low, 1)
    
    # Calculate close change
    close_change = close - delay_close
    
    # Calculate close minus open
    close_minus_open = close - open_price
    
    # Calculate delay of close minus delay of open
    delay_close_minus_delay_open = delay_close - delay_open
    
    # Calculate numerator
    numerator = 16 * (close_change + close_minus_open / 2 + delay_close_minus_delay_open)
    
    # Calculate absolute high minus delay of close
    abs_high_minus_delay_close = np.abs(high - delay_close)
    
    # Calculate absolute low minus delay of close
    abs_low_minus_delay_close = np.abs(low - delay_close)
    
    # Calculate absolute high minus delay of low
    abs_high_minus_delay_low = np.abs(high - delay_low)
    
    # Calculate denominator
    denominator = np.where(
        (abs_high_minus_delay_close > abs_low_minus_delay_close) & (abs_high_minus_delay_close > abs_high_minus_delay_low),
        abs_high_minus_delay_close + abs_low_minus_delay_close / 2 + np.abs(delay_close - delay_open) / 4,
        np.where(
            (abs_low_minus_delay_close > abs_high_minus_delay_low) & (abs_low_minus_delay_close > abs_high_minus_delay_close),
            abs_low_minus_delay_close + abs_high_minus_delay_close / 2 + np.abs(delay_close - delay_open) / 4,
            abs_high_minus_delay_low + np.abs(delay_close - delay_open) / 4
        )
    )
    
    # Calculate maximum of absolute high minus delay of close and absolute low minus delay of close
    max_abs_high_low_minus_delay_close = np.maximum(abs_high_minus_delay_close, abs_low_minus_delay_close)
    
    # Calculate final result
    result = numerator / denominator * max_abs_high_low_minus_delay_close
    
    return pd.Series(result, index=df.index, name='alpha_137')

def alpha137(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_137, code, benchmark, end_date, lookback)
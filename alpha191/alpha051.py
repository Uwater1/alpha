import numpy as np
import pandas as pd
from .operators import delay, ts_sum
from .utils import run_alpha_factor


def alpha_051(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha051 factor.

    Formula:
        SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)
        /(SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)
        +SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12))

    This factor measures the ratio of upward price movements to total price movements
    over a 12-day window, based on comparisons between current (HIGH+LOW) and
    previous day's (HIGH+LOW).
    """
    # Extract values as numpy arrays
    high = df['high'].values
    low = df['low'].values

    # Calculate DELAY(HIGH,1) and DELAY(LOW,1)
    delay_high = delay(high, 1)
    delay_low = delay(low, 1)

    # Calculate HIGH+LOW and DELAY(HIGH,1)+DELAY(LOW,1)
    high_low_sum = high + low
    delay_high_low_sum = delay_high + delay_low

    # Calculate ABS(HIGH-DELAY(HIGH,1)) and ABS(LOW-DELAY(LOW,1))
    abs_high_diff = np.abs(high - delay_high)
    abs_low_diff = np.abs(low - delay_low)

    # Calculate MAX(ABS(HIGH-DELAY(HIGH,1)), ABS(LOW-DELAY(LOW,1)))
    max_diff = np.maximum(abs_high_diff, abs_low_diff)

    # Condition 1: (HIGH+LOW) <= (DELAY(HIGH,1)+DELAY(LOW,1))
    # If true, use 0; else use max_diff
    condition1 = high_low_sum <= delay_high_low_sum
    numerator_values = np.where(condition1, 0, max_diff)

    # Condition 2: (HIGH+LOW) >= (DELAY(HIGH,1)+DELAY(LOW,1))
    # If true, use 0; else use max_diff
    condition2 = high_low_sum >= delay_high_low_sum
    denominator_add_values = np.where(condition2, 0, max_diff)

    # Calculate SUM over 12 periods
    sum_numerator = ts_sum(numerator_values, 12)
    sum_denominator_add = ts_sum(denominator_add_values, 12)

    # Calculate denominator: sum_numerator + sum_denominator_add
    denominator = sum_numerator + sum_denominator_add

    # Handle division by zero
    result = np.full(len(df), np.nan)
    valid_mask = (denominator != 0) & ~np.isnan(denominator)
    result[valid_mask] = sum_numerator[valid_mask] / denominator[valid_mask]

    return pd.Series(result, index=df.index, name='alpha_051')


def alpha051(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    """
    Alpha051 factor wrapper function.

    Parameters:
    -----------
    code : str
        Stock code to calculate factor for
    benchmark : str, default 'zz800'
        Benchmark index code
    end_date : str, default "2026-01-23"
        End date for calculation
    lookback : int, default 350
        Number of lookback days

    Returns:
    --------
    pd.Series
        Alpha051 factor values
    """
    return run_alpha_factor(alpha_051, code, benchmark, end_date, lookback)

"""
Alpha049 factor implementation.

Formula:
    alpha_049 = SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)/(SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)+SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12))
"""

import numpy as np
import pandas as pd
from .operators import delay, ts_sum
from .utils import run_alpha_factor


def alpha_049(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha049 factor.

    Formula:
        alpha_049 = SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)/(SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)+SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12))
    """
    # Ensure we have required columns
    required_cols = ['high', 'low']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in DataFrame")

    if 'date' in df.columns:
        index = pd.to_datetime(df['date'])
    else:
        index = df.index

    high = df['high'].values
    low = df['low'].values

    # Step 1: Compute DELAY(HIGH, 1)
    delayed_high_1 = delay(high, 1)

    # Step 2: Compute DELAY(LOW, 1)
    delayed_low_1 = delay(low, 1)

    # Step 3: Compute (HIGH + LOW)
    sum_high_low = high + low

    # Step 4: Compute (DELAY(HIGH,1) + DELAY(LOW,1))
    sum_delayed_high_low = delayed_high_1 + delayed_low_1

    # Step 5: Compute ABS(HIGH - DELAY(HIGH,1))
    abs_diff_high = np.abs(high - delayed_high_1)

    # Step 6: Compute ABS(LOW - DELAY(LOW,1))
    abs_diff_low = np.abs(low - delayed_low_1)

    # Step 7: Compute MAX(ABS(HIGH-DELAY(HIGH,1)), ABS(LOW-DELAY(LOW,1)))
    max_abs_diff = np.maximum(abs_diff_high, abs_diff_low)

    # Step 8: Compute the numerator condition
    # ((HIGH+LOW) >= (DELAY(HIGH,1)+DELAY(LOW,1)) ? 0 : MAX(...))
    numerator_component = np.where(sum_high_low >= sum_delayed_high_low, 0.0, max_abs_diff)

    # Step 9: Compute the denominator condition for the second sum
    # ((HIGH+LOW) <= (DELAY(HIGH,1)+DELAY(LOW,1)) ? 0 : MAX(...))
    denominator_component = np.where(sum_high_low <= sum_delayed_high_low, 0.0, max_abs_diff)

    # Step 10: Compute SUM(..., 12) for numerator
    sum_numerator = ts_sum(numerator_component, 12)

    # Step 11: Compute SUM(..., 12) for denominator part 1
    sum_denominator_1 = ts_sum(numerator_component, 12)

    # Step 12: Compute SUM(..., 12) for denominator part 2
    sum_denominator_2 = ts_sum(denominator_component, 12)

    # Step 13: Compute the final result
    # Protect against division by zero
    denom = sum_denominator_1 + sum_denominator_2
    denom[denom == 0] = np.nan
    alpha_values = sum_numerator / denom

    return pd.Series(alpha_values, index=index, name='alpha_049')


def alpha049(
    code: str,
    benchmark: str = 'zz800',
    end_date: str = "2026-01-23",
    lookback: int = 350
) -> float:
    """
    Compute Alpha049 factor value for a stock at a specific date.
    """
    return run_alpha_factor(alpha_049, code, benchmark, end_date, lookback)
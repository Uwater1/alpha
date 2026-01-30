"""
Alpha050 factor implementation.

Formula:
    alpha_050 = SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)/(SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)+SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12))-SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)/(SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)+SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12))
"""

import numpy as np
import pandas as pd
from .operators import delay, ts_sum
from .utils import run_alpha_factor


def alpha_050(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha050 factor.

    Formula:
        alpha_050 = SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)/(SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)+SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12))-SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)/(SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)+SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12))
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

    # Step 8: Compute the first numerator condition
    # ((HIGH+LOW) <= (DELAY(HIGH,1)+DELAY(LOW,1)) ? 0 : MAX(...))
    numerator_1_component = np.where(sum_high_low <= sum_delayed_high_low, 0.0, max_abs_diff)

    # Step 9: Compute the first denominator condition
    # ((HIGH+LOW) >= (DELAY(HIGH,1)+DELAY(LOW,1)) ? 0 : MAX(...))
    denominator_1_component = np.where(sum_high_low >= sum_delayed_high_low, 0.0, max_abs_diff)

    # Step 10: Compute the second numerator condition (same as denominator_1_component)
    numerator_2_component = denominator_1_component

    # Step 11: Compute the second denominator condition (same as numerator_1_component)
    denominator_2_component = numerator_1_component

    # Step 12: Compute SUM(..., 12) for first numerator
    sum_numerator_1 = ts_sum(numerator_1_component, 12)

    # Step 13: Compute SUM(..., 12) for first denominator part 1
    sum_denominator_1_1 = ts_sum(numerator_1_component, 12)

    # Step 14: Compute SUM(..., 12) for first denominator part 2
    sum_denominator_1_2 = ts_sum(denominator_1_component, 12)

    # Step 15: Compute SUM(..., 12) for second numerator
    sum_numerator_2 = ts_sum(numerator_2_component, 12)

    # Step 16: Compute SUM(..., 12) for second denominator part 1
    sum_denominator_2_1 = ts_sum(numerator_2_component, 12)

    # Step 17: Compute SUM(..., 12) for second denominator part 2
    sum_denominator_2_2 = ts_sum(denominator_2_component, 12)

    # Step 18: Compute the first fraction
    fraction_1 = sum_numerator_1 / (sum_denominator_1_1 + sum_denominator_1_2)

    # Step 19: Compute the second fraction
    fraction_2 = sum_numerator_2 / (sum_denominator_2_1 + sum_denominator_2_2)

    # Step 20: Compute the final result
    alpha_values = fraction_1 - fraction_2

    return pd.Series(alpha_values, index=index, name='alpha_050')


def alpha050(
    code: str,
    benchmark: str = 'zz800',
    end_date: str = "2026-01-23",
    lookback: int = 350
) -> float:
    """
    Compute Alpha050 factor value for a stock at a specific date.
    """
    return run_alpha_factor(alpha_050, code, benchmark, end_date, lookback)
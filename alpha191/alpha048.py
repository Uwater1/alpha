"""
Alpha048 factor implementation.

Formula:
    alpha_048 = (-1*((RANK(((SIGN((CLOSE-DELAY(CLOSE,1)))+SIGN((DELAY(CLOSE,1)-DELAY(CLOSE,2))))+SIGN((DELAY(CLOSE,2)-DELAY(CLOSE,3))))))*SUM(VOLUME,5))/SUM(VOLUME,20))
"""

import numpy as np
import pandas as pd
from .operators import delay, ts_sum, sign, rank
from .utils import run_alpha_factor


def alpha_048(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha048 factor.

    Formula:
        alpha_048 = (-1*((RANK(((SIGN((CLOSE-DELAY(CLOSE,1)))+SIGN((DELAY(CLOSE,1)-DELAY(CLOSE,2))))+SIGN((DELAY(CLOSE,2)-DELAY(CLOSE,3))))))*SUM(VOLUME,5))/SUM(VOLUME,20))
    """
    # Ensure we have required columns
    required_cols = ['close', 'volume']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in DataFrame")

    if 'date' in df.columns:
        index = pd.to_datetime(df['date'])
    else:
        index = df.index

    close = df['close'].values
    volume = df['volume'].values

    # Step 1: Compute DELAY(CLOSE, 1)
    delayed_close_1 = delay(close, 1)

    # Step 2: Compute DELAY(CLOSE, 2)
    delayed_close_2 = delay(close, 2)

    # Step 3: Compute DELAY(CLOSE, 3)
    delayed_close_3 = delay(close, 3)

    # Step 4: Compute (CLOSE - DELAY(CLOSE, 1))
    diff_1 = close - delayed_close_1

    # Step 5: Compute (DELAY(CLOSE, 1) - DELAY(CLOSE, 2))
    diff_2 = delayed_close_1 - delayed_close_2

    # Step 6: Compute (DELAY(CLOSE, 2) - DELAY(CLOSE, 3))
    diff_3 = delayed_close_2 - delayed_close_3

    # Step 7: Compute SIGN((CLOSE-DELAY(CLOSE,1)))
    sign_1 = sign(diff_1)

    # Step 8: Compute SIGN((DELAY(CLOSE,1)-DELAY(CLOSE,2)))
    sign_2 = sign(diff_2)

    # Step 9: Compute SIGN((DELAY(CLOSE,2)-DELAY(CLOSE,3)))
    sign_3 = sign(diff_3)

    # Step 10: Compute the sum of signs
    sum_signs = sign_1 + sign_2 + sign_3

    # Step 11: Compute RANK of the sum of signs
    ranked_sum_signs = rank(sum_signs)

    # Step 12: Compute SUM(VOLUME, 5)
    sum_volume_5 = ts_sum(volume, 5)

    # Step 13: Compute SUM(VOLUME, 20)
    sum_volume_20 = ts_sum(volume, 20)

    # Step 14: Compute the numerator
    numerator = ranked_sum_signs * sum_volume_5

    # Step 15: Compute the final result
    alpha_values = (-1 * numerator) / sum_volume_20

    return pd.Series(alpha_values, index=index, name='alpha_048')


def alpha048(
    code: str,
    benchmark: str = 'zz800',
    end_date: str = "2026-01-23",
    lookback: int = 350
) -> float:
    """
    Compute Alpha048 factor value for a stock at a specific date.
    """
    return run_alpha_factor(alpha_048, code, benchmark, end_date, lookback)
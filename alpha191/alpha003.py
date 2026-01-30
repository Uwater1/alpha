import numpy as np
import pandas as pd
from .operators import delay, ts_sum
from .utils import run_alpha_factor


def alpha_003(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha003 factor.
    Formula: SUM((CLOSE=DELAY(CLOSE,1)?0:CLOSE-(CLOSE>DELAY(CLOSE,1)?MIN(LOW,DELAY(CLOSE,1)):MAX(HIGH,DELAY(CLOSE,1)))),6)
    """
    # Ensure we have required columns
    required_cols = ['high', 'low', 'close']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in DataFrame")

    if 'date' in df.columns:
        index = pd.to_datetime(df['date'])
    else:
        index = df.index

    close = df['close'].values
    low = df['low'].values
    high = df['high'].values

    prev_close = delay(close, 1)

    # (CLOSE > DELAY(CLOSE,1) ? MIN(LOW, DELAY(CLOSE,1)) : MAX(HIGH, DELAY(CLOSE,1)))
    cond_inner = np.where(close > prev_close, np.minimum(low, prev_close), np.maximum(high, prev_close))

    # (CLOSE == DELAY(CLOSE,1) ? 0 : CLOSE - cond_inner)
    term = np.where(close == prev_close, 0.0, close - cond_inner)

    result_values = ts_sum(term, 6)

    return pd.Series(result_values, index=index, name='alpha_003')


def alpha003(
    code: str,
    benchmark: str = 'zz800',
    end_date: str = "2026-01-23",
    lookback: int = 350
) -> float:
    """
    Compute Alpha003 factor value for a stock at a specific date.
    """
    return run_alpha_factor(alpha_003, code, benchmark, end_date, lookback)

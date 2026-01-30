"""
Alpha001 factor implementation.

Formula:
    alpha_001 = -1 * CORR(
        RANK(DELTA(LOG(VOLUME), 1)),
        RANK((CLOSE-OPEN)/OPEN),
        6
    )
"""

import numpy as np
import pandas as pd
from .operators import ts_rank, rolling_corr
from .utils import load_stock_csv, run_alpha_factor


def alpha_001(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha001 factor.

    Formula:
        alpha_001 = -1 * CORR(
            RANK(DELTA(LOG(VOLUME), 1)),
            RANK((CLOSE-OPEN)/OPEN),
            6
        )
    """
    # Ensure we have required columns
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in DataFrame")

    # Get date index
    if 'date' in df.columns:
        index = pd.to_datetime(df['date'])
    else:
        index = df.index

    # Step 1: Compute DELTA(LOG(volume), 1)
    volume = df['volume'].replace(0, np.nan).values
    log_volume = np.log(volume)
    delta_log_volume = np.diff(log_volume, prepend=np.nan)

    # Step 2: Compute RANK over window=6 (simulated by ts_rank)
    rank_delta_volume = ts_rank(delta_log_volume, window=6)

    # Step 3: Compute (close - open) / open
    returns_ratio = (df['close'].values - df['open'].values) / df['open'].values

    # Step 4: Compute RANK over window=6 (simulated by ts_rank)
    rank_returns = ts_rank(returns_ratio, window=6)

    # Step 5: Compute CORR with window=6
    correlation = rolling_corr(rank_delta_volume, rank_returns, window=6)

    # Step 6: Final result
    alpha_values = -1 * correlation

    return pd.Series(alpha_values, index=index, name='alpha_001')


def alpha001(
    code: str,
    benchmark: str = 'zz800',
    end_date: str = "2026-01-23",
    lookback: int = 350
) -> float:
    """
    Compute Alpha001 factor value for a stock at a specific date.
    """
    return run_alpha_factor(alpha_001, code, benchmark, end_date, lookback)

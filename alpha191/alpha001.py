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


def alpha_001(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha001 factor.

    Parameters
    ----------
    df : pd.DataFrame
        Single-stock DataFrame with columns: date, open, high, low, close, volume, amount
        Data should be date-sorted.

    Returns
    -------
    pd.Series
        Alpha001 values indexed by date

    Usage:
        >>> import pandas as pd
        >>> from alpha191.alpha001 import alpha_001
        >>> df = pd.read_csv('bao/hs300/sh_600009.csv')
        >>> result = alpha_001(df)
        >>> print(result.tail(10))
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

    # Step 1: Compute DELTA(LOG(volume), 1) = log(volume[t]) - log(volume[t-1])
    # Handle zero volume values by replacing with NaN before log
    volume = df['volume'].replace(0, np.nan).values
    log_volume = np.log(volume)
    delta_log_volume = np.diff(log_volume, prepend=np.nan)

    # Step 2: Compute RANK of delta_log_volume over rolling window (default 6)
    rank_delta_volume = ts_rank(delta_log_volume, window=6)

    # Step 3: Compute (close - open) / open
    returns_ratio = (df['close'].values - df['open'].values) / df['open'].values

    # Step 4: Compute RANK of returns_ratio over rolling window (default 6)
    rank_returns = ts_rank(returns_ratio, window=6)

    # Step 5: Compute CORR of the two ranks with window=6
    correlation = rolling_corr(rank_delta_volume, rank_returns, window=6)

    # Step 6: Multiply by -1
    alpha_values = -1 * correlation

    # Step 7: Return as pd.Series indexed by date
    result = pd.Series(alpha_values, index=index, name='alpha_001')

    return result

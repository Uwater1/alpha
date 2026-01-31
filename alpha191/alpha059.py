import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_059(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha059 factor.
    Formula: SUM((CLOSE=DELAY(CLOSE,1)?0:CLOSE-(CLOSE>DELAY(CLOSE,1)?MIN(LOW,DELAY(CLOSE,1)):MAX(HIGH,DELAY(CLOSE,1)))),20)
    """
    # Calculate DELAY(CLOSE, 1)
    delayed_close = delay(df['close'], 1)
    
    # Calculate condition: CLOSE > DELAY(CLOSE, 1)
    condition = df['close'] > delayed_close
    
    # Calculate the main expression using np.where
    # If CLOSE == DELAY(CLOSE, 1): 0
    # Else if CLOSE > DELAY(CLOSE, 1): MIN(LOW, DELAY(CLOSE, 1))
    # Else: MAX(HIGH, DELAY(CLOSE, 1))
    main_expr = np.where(
        df['close'] == delayed_close,
        0,
        np.where(
            condition,
            np.minimum(df['low'], delayed_close),
            np.maximum(df['high'], delayed_close)
        )
    )
    
    # Calculate 20-period sum
    result = ts_sum(main_expr, 20)
    
    return pd.Series(result, index=df.index, name='alpha_059')

def alpha059(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_059, code, benchmark, end_date, lookback)
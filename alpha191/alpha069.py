import numpy as np
import pandas as pd
from .operators import *
from .utils import run_alpha_factor

def alpha_069(df: pd.DataFrame) -> pd.Series:
    """
    Compute Alpha069 factor.
    Formula: (SUM(DTM,20)>SUM(DBM,20)?(SUM(DTM,20)-SUM(DBM,20))/SUM(DTM,20):(SUM(DTM,20)=SUM(DBM,20)？0:(SUM(DTM,20)-SUM(DBM,20))/SUM(DBM,20)))
    """
    # Calculate DTM and DBM using the utility functions
    dtm = compute_dtm(df['open'], df['high'])
    dbm = compute_dbm(df['open'], df['low'])
    
    # Calculate SUM(DTM, 20) and SUM(DBM, 20)
    sum_dtm = ts_sum(dtm, 20)
    sum_dbm = ts_sum(dbm, 20)
    
    # Calculate the difference
    diff = sum_dtm - sum_dbm
    
    # Build conditional result using np.where
    # If SUM(DTM,20) > SUM(DBM,20): (SUM(DTM,20)-SUM(DBM,20))/SUM(DTM,20)
    # If SUM(DTM,20) == SUM(DBM,20): 0
    # Else: (SUM(DTM,20)-SUM(DBM,20))/SUM(DBM,20)
    result = np.where(
        sum_dtm > sum_dbm,
        diff / sum_dtm,
        np.where(
            sum_dtm == sum_dbm,
            0,
            diff / sum_dbm
        )
    )
    
    return pd.Series(result, index=df.index, name='alpha_069')

def alpha069(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_069, code, benchmark, end_date, lookback)
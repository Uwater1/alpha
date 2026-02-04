import pandas as pd
import numpy as np
from typing import Union, List, Optional

def stack_wide_to_long(wide_df: pd.DataFrame, name: str) -> pd.Series:
    """
    Convert a wide DataFrame (Date x Asset) to a long Series with MultiIndex (Date, Asset).
    """
    long_df = wide_df.stack()
    long_df.index.names = ['date', 'asset']
    long_df.name = name
    return long_df

def get_clean_factor_and_forward_returns(
    factor: pd.DataFrame,
    prices: pd.DataFrame,
    periods: Union[int, List[int]] = (1, 5, 20),
    quantiles: int = 5,
    bins: Optional[int] = None,
    filter_zscore: Optional[float] = None,
    groupby: Optional[pd.DataFrame] = None,
    binning_by_group: bool = False,
    max_loss: float = 0.35,
    zero_aware: bool = False
) -> pd.DataFrame:
    """
    A bridge function similar to Alphalens' get_clean_factor_and_forward_returns.
    Expects wide DataFrames (Date x Asset) for factor and prices.
    Returns a MultiIndex DataFrame (date, asset) formatted for Alphalens performance functions.
    """
    if isinstance(periods, int):
        periods = [periods]
    
    # 1. Calculate forward returns (wide format)
    # return_t = (price_{t+n} / price_t) - 1
    forward_returns = {}
    for p in periods:
        ret = prices.shift(-p) / prices - 1
        forward_returns[f"{p}D"] = stack_wide_to_long(ret, f"{p}D")
    
    # 2. Stack factor
    long_factor = stack_wide_to_long(factor, 'factor')
    
    # 3. Combine
    merged_data = pd.DataFrame(index=long_factor.index)
    merged_data['factor'] = long_factor
    for p_str, ret_series in forward_returns.items():
        merged_data[p_str] = ret_series
        
    # 4. Filter NaNs
    merged_data = merged_data.dropna(subset=['factor'])
    
    # 5. Binning (Quantiles)
    if bins is not None:
        # Use bins
        def binning(x):
            try:
                return pd.cut(x, bins, labels=False) + 1
            except ValueError:
                return np.nan
        merged_data['factor_quantile'] = merged_data.groupby(level='date')['factor'].transform(binning)
    else:
        # Use quantiles
        def quantize(x):
            try:
                if len(x.dropna()) < quantiles:
                    return np.nan
                return pd.qcut(x, quantiles, labels=False, duplicates='drop') + 1
            except ValueError:
                return np.nan
        merged_data['factor_quantile'] = merged_data.groupby(level='date')['factor'].transform(quantize)
        
    # 6. Groupby (if provided)
    if groupby is not None:
        long_groups = stack_wide_to_long(groupby, 'group')
        merged_data['group'] = long_groups
        
    # Drop rows with any NaN returns (Alphalens usually does this to ensure consistency)
    # But we might want to keep them for IC calculation if we have at least one return.
    # For now, we follow Alphalens loosely.
    
    return merged_data.dropna()

def print_table(table: pd.DataFrame, name: Optional[str] = None):
    """Simple table printer."""
    if name:
        print(f"\n{name}")
    print(table)

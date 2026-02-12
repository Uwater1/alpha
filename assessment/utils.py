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
    zero_aware: bool = False,
    return_wide: bool = False
) -> Union[pd.DataFrame, tuple]:
    """
    A bridge function similar to Alphalens' get_clean_factor_and_forward_returns.
    Expects wide DataFrames (Date x Asset) for factor and prices.
    Returns a MultiIndex DataFrame (date, asset) formatted for Alphalens performance functions.
    If return_wide is True, also returns (factor, quantized_factor, forward_returns_dict)
    """
    if isinstance(periods, int):
        periods = [periods]
    
    # 1. Calculate forward returns (wide format) - optimized
    # return_t = (price_{t+n} / price_t) - 1
    forward_returns = {}
    for p in periods:
        # Use more efficient calculation with float32
        ret = prices.shift(-p).astype(np.float32) / prices.astype(np.float32) - 1
        forward_returns[f"{p}D"] = stack_wide_to_long(ret, f"{p}D")
    
    # 2. Stack factor
    long_factor = stack_wide_to_long(factor, 'factor')
    
    # 3. Combine - optimized memory usage
    merged_data = pd.DataFrame(index=long_factor.index, dtype=np.float32)
    merged_data['factor'] = long_factor.astype(np.float32)
    for p_str, ret_series in forward_returns.items():
        merged_data[p_str] = ret_series.astype(np.float32)
        
    # 4. Filter NaNs
    merged_data = merged_data.dropna(subset=['factor'])
    
    # 5. Binning (Quantiles)
    if bins is not None:
        # Use bins - more efficient with apply
        def binning(group):
            try:
                return pd.cut(group, bins, labels=False) + 1
            except ValueError:
                return pd.Series([np.nan] * len(group), index=group.index)
        merged_data['factor_quantile'] = merged_data.groupby(level='date', group_keys=False)['factor'].apply(binning)
        q_wide = merged_data['factor_quantile'].unstack()
    else:
        # Vectorized Quantization using wide-matrix rank - optimized
        ranks = factor.rank(axis=1, pct=True)
        # Convert percentiles to quantile bins - use float32 for efficiency
        long_ranks = stack_wide_to_long(ranks.astype(np.float32), 'rank')
        
        # 1-indexed quantiles: ceil(rank * quantiles) - optimized calculation
        quantized = np.ceil(long_ranks.values * quantiles)
        # Clip just in case of precision issues
        merged_data['factor_quantile'] = pd.Series(quantized, index=long_ranks.index).clip(1, quantiles).astype(np.int8)
        
        # Quantized wide matrix
        q_wide = np.ceil(ranks * quantiles).clip(1, quantiles).fillna(-1).astype(np.int8)
        
    # 6. Groupby (if provided)
    if groupby is not None:
        long_groups = stack_wide_to_long(groupby, 'group')
        merged_data['group'] = long_groups
        
    long_res = merged_data.dropna()
    if return_wide:
        return long_res, factor, q_wide
    return long_res

def print_table(table: pd.DataFrame, name: Optional[str] = None):
    """Simple table printer."""
    if name:
        print(f"\n{name}")
    print(table)

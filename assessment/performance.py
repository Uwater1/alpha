import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Any, List, Optional
try:
    from numba import jit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Fallback decorator that does nothing
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if not args else decorator(args[0])


@jit(nopython=True, cache=True)
def _fast_pearson_corr(x, y):
    """
    Fast Pearson correlation using Numba.
    Assumes no NaN values in inputs.
    """
    n = len(x)
    if n == 0:
        return np.nan
    
    sum_x = 0.0
    sum_y = 0.0
    sum_xx = 0.0
    sum_yy = 0.0
    sum_xy = 0.0
    
    for i in range(n):
        sum_x += x[i]
        sum_y += y[i]
        sum_xx += x[i] * x[i]
        sum_yy += y[i] * y[i]
        sum_xy += x[i] * y[i]
    
    mean_x = sum_x / n
    mean_y = sum_y / n
    
    numerator = sum_xy - n * mean_x * mean_y
    denominator = np.sqrt((sum_xx - n * mean_x * mean_x) * (sum_yy - n * mean_y * mean_y))
    
    if denominator == 0:
        return np.nan
    
    return numerator / denominator


def fast_spearman_corr(x, y):
    """
    Fast Spearman correlation by ranking then computing Pearson correlation.
    Handles NaN values by removing them pairwise.
    """
    # Remove NaN values pairwise
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]
    
    if len(x_clean) < 2:
        return np.nan
    
    # Rank the data (using scipy's rankdata for ties handling)
    from scipy.stats import rankdata
    x_ranked = rankdata(x_clean).astype(np.float64)
    y_ranked = rankdata(y_clean).astype(np.float64)
    
    # Use fast Pearson correlation on ranked data
    if HAS_NUMBA:
        return _fast_pearson_corr(x_ranked, y_ranked)
    else:
        # Fallback to numpy's corrcoef
        return np.corrcoef(x_ranked, y_ranked)[0, 1]

def factor_information_coefficient(factor_data: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the Spearman Rank Correlation based Information Coefficient (IC)
    between factor values and forward returns.
    Uses fast Numba-accelerated correlation when available.
    """
    def spearman_ic(group):
        f = group['factor'].values
        ic_cols = {}
        for col in group.columns:
            if col.endswith('D'):
                ret_values = group[col].values
                ic_cols[col] = fast_spearman_corr(f, ret_values)
        return pd.Series(ic_cols)

    ic = factor_data.groupby(level='date').apply(spearman_ic)
    return ic

def mean_information_coefficient(
    factor_data: pd.DataFrame,
    by_time: Optional[str] = None
) -> pd.DataFrame:
    """
    Get the mean information coefficient.
    """
    ic = factor_information_coefficient(factor_data)
    if by_time is not None:
        return ic.resample(by_time).mean()
    return ic.mean().to_frame().T

def quantile_turnover(quantile_factor: pd.Series, quantile: int, period: int = 1) -> pd.Series:
    """
    Computes the proportion of names in a factor quantile that were
    not in that quantile in the previous period.
    """
    quant_factor = quantile_factor[quantile_factor == quantile]
    
    # We need to compute turnover for each date
    dates = quant_factor.index.get_level_values('date').unique()
    turnover = pd.Series(index=dates, dtype=float)
    
    for i in range(period, len(dates)):
        current_date = dates[i]
        prev_date = dates[i-period]
        
        current_assets = set(quant_factor.xs(current_date, level='date').index)
        prev_assets = set(quant_factor.xs(prev_date, level='date').index)
        
        if not prev_assets:
            turnover[current_date] = np.nan
        else:
            # Fraction of current assets that were NOT in the previous assets
            turnover[current_date] = 1.0 - len(current_assets & prev_assets) / len(current_assets)
            
    return turnover.dropna()

def mean_return_by_quantile(factor_data: pd.DataFrame) -> pd.DataFrame:
    """
    Computes mean returns for factor quantiles.
    """
    # Columns that end with 'D' are return columns
    return_cols = [c for c in factor_data.columns if c.endswith('D')]
    
    mean_ret = factor_data.groupby('factor_quantile')[return_cols].mean()
    return mean_ret

def factor_rank_autocorrelation(factor_data: pd.DataFrame, period: int = 1) -> pd.Series:
    """
    Computes the rank autocorrelation of the factor. 
    This is a measure of factor stability (turnover).
    """
    # Unstack to get wide format (Date x Asset)
    factor_wide = factor_data['factor'].unstack()
    
    # Calculate rank autocorrelation
    # We rank cross-sectionally for each day
    ranks = factor_wide.rank(axis=1)
    
    autocorr = ranks.corrwith(ranks.shift(period), axis=1)
    return autocorr.dropna()

def factor_alpha_beta(factor_data: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the alpha (excess returns) and beta (market exposure) of a factor.
    """
    try:
        from statsmodels.regression.linear_model import OLS
        from statsmodels.tools.tools import add_constant
    except ImportError:
        # If statsmodels not available, return empty result
        return pd.DataFrame()

    return_cols = [c for c in factor_data.columns if c.endswith('D')]
    
    # We use the mean return of all stocks in the universe as the market return
    # This is a simplification but common in factor analysis
    results = {}
    for col in return_cols:
        # Long-Short returns based on factor weights (demeaned factor)
        # weight = factor / sum(abs(factor))
        # Simplest approach: use the factor itself as weight
        daily_returns = factor_data.groupby(level='date').apply(
            lambda x: (x['factor'] * x[col]).sum() / (x['factor'].abs().sum() + 1e-9)
        )
        
        market_returns = factor_data.groupby(level='date')[col].mean()
        
        # Check alignment
        common_idx = daily_returns.index.intersection(market_returns.index)
        daily_returns = daily_returns.loc[common_idx]
        market_returns = market_returns.loc[common_idx]
        
        if len(common_idx) < 5:
            continue
            
        # Regression: factor_return ~ alpha + beta * market_return
        X = add_constant(market_returns.values)
        model = OLS(daily_returns.values, X).fit()
        
        results[col] = pd.Series({
            'Alpha': model.params[0],
            'Beta': model.params[1],
            'Alpha T-Stat': model.tvalues[0],
            'Beta T-Stat': model.tvalues[1]
        })
        
    return pd.DataFrame(results)

def cumulative_returns(returns: pd.Series) -> pd.Series:
    """
    Computes cumulative returns from simple daily returns.
    """
    return (1 + returns).cumprod() - 1

def compute_performance_metrics(factor_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Main entry point to compute all performance metrics.
    """
    ic = factor_information_coefficient(factor_data)
    mean_ret = mean_return_by_quantile(factor_data)
    alpha_beta = factor_alpha_beta(factor_data)
    
    # Summary IC stats
    ic_summary = pd.DataFrame({
        'IC Mean': ic.mean(),
        'IC Std.': ic.std(),
        'Risk-Adjusted IC (IR)': ic.mean() / ic.std(),
        't-stat(IC)': (ic.mean() / ic.std()) * np.sqrt(len(ic)),
        'IC P-value': [stats.t.sf(np.abs(t), len(ic)-1)*2 for t in (ic.mean() / ic.std()) * np.sqrt(len(ic))],
    }).T
    
    # Turnover (1D stability)
    autocorr = factor_rank_autocorrelation(factor_data, period=1)
    
    return {
        'ic': ic,
        'ic_summary': ic_summary,
        'mean_ret': mean_ret,
        'alpha_beta': alpha_beta,
        'autocorr': autocorr,
        'autocorr_mean': autocorr.mean()
    }

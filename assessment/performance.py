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
    """
    # Unstack factor to Date x Asset
    f_wide = factor_data['factor'].unstack()
    f_ranked = f_wide.rank(axis=1)
    
    ic_results = {}
    return_cols = [c for c in factor_data.columns if c.endswith('D')]
    
    for col in return_cols:
        # Unstack return to Date x Asset
        r_wide = factor_data[col].unstack()
        r_ranked = r_wide.rank(axis=1)
        
        # Vectorized correlation between rows for each day
        ic_results[col] = f_ranked.corrwith(r_ranked, axis=1)
        
    return pd.DataFrame(ic_results)

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
    not in that quantile in the previous period using wide matrix operations.
    """
    # Unstack to get wide format (Date x Asset)
    # This is much faster if we do it once outside, but for compatibility we do it here
    q_wide = quantile_factor.unstack()
    
    # Boolean mask for the specific quantile
    mask = (q_wide == quantile)
    
    # count (mask & mask.shift(period)) / count(mask)
    # This avoids the explicit set operations and looping over dates
    count_current = mask.sum(axis=1)
    
    # Overlap between current and previous
    overlap = (mask & mask.shift(period)).sum(axis=1)
    
    # Turnover = 1 - overlap / count
    turnover = 1.0 - overlap / count_current
    
    return turnover.dropna()

def mean_return_by_quantile(factor_data: pd.DataFrame) -> pd.DataFrame:
    """
    Computes mean returns for factor quantiles.
    """
    # Columns that end with 'D' are return columns
    return_cols = [c for c in factor_data.columns if c.endswith('D')]
    
    mean_ret = factor_data.groupby('factor_quantile')[return_cols].mean()
    return mean_ret

def monotonicity_score(mean_ret: pd.DataFrame) -> pd.Series:
    """
    Computes the monotonicity score for each return period.
    Spearman rank correlation between quantile rank and mean return.
    """
    scores = {}
    for col in mean_ret.columns:
        # Quantile indices (1, 2, 3...) vs Mean Returns
        # We use Spearman correlation
        corr, _ = stats.spearmanr(mean_ret.index, mean_ret[col])
        scores[col] = corr
    return pd.Series(scores)

def average_quantile_turnover(factor_data: pd.DataFrame, period: int = 1) -> pd.DataFrame:
    """
    Computes average turnover for each quantile.
    """
    quantiles = factor_data['factor_quantile'].unique()
    quantiles = sorted([q for q in quantiles if not np.isnan(q)])
    
    turnover_results = {}
    for q in quantiles:
        turnover_series = quantile_turnover(factor_data['factor_quantile'], q, period=period)
        turnover_results[int(q)] = turnover_series.mean()
        
    return pd.Series(turnover_results, name='Average Turnover')

def factor_rank_autocorrelation(factor_data: pd.DataFrame, period: int = 1) -> pd.Series:
    """
    Computes the rank autocorrelation of the factor. 
    """
    # Unstack to get wide format (Date x Asset)
    factor_wide = factor_data['factor'].unstack()
    
    # Calculate rank autocorrelation using vectorized corrwith
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

def calculate_portfolio_performance(returns: pd.Series, periods_per_year: int = 252) -> Dict[str, float]:
    """
    Computes annualized return, volatility, Sharpe ratio, max drawdown, and Calmar ratio.
    """
    if returns.empty:
        return {
            'ann_ret': np.nan, 'ann_vol': np.nan, 'sharpe': np.nan, 
            'max_dd': np.nan, 'calmar': np.nan
        }
    
    # Use log returns for easier compounding if needed, but simple returns are standard for daily
    total_ret = (1 + returns).prod() - 1
    n_years = len(returns) / periods_per_year
    
    ann_ret = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else np.nan
    ann_vol = returns.std() * np.sqrt(periods_per_year)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    
    cum_ret = (1 + returns).cumprod()
    running_max = cum_ret.expanding().max()
    drawdown = (cum_ret - running_max) / running_max
    max_dd = drawdown.min()
    
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else np.nan
    
    return {
        'ann_ret': ann_ret,
        'ann_vol': ann_vol,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'calmar': calmar
    }

def quantile_performance_stats(factor_data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Computes detailed stats for each quantile: Mean, SE, T-Stat, P-Value.
    Also computes stats for the Long-Short spread.
    """
    return_cols = [c for c in factor_data.columns if c.endswith('D')]
    all_stats = {}
    
    for col in return_cols:
        # Daily mean returns per quantile
        date_col = [level_name for level_name in factor_data.index.names if 'date' in str(level_name).lower()][0]
        daily_q_ret = factor_data.groupby([date_col, 'factor_quantile'])[col].mean().unstack()
        
        # Stats per quantile
        q_mean = daily_q_ret.mean()
        q_std = daily_q_ret.std()
        q_count = daily_q_ret.count()
        q_se = q_std / np.sqrt(q_count)
        q_tstat = q_mean / q_se
        q_pvalue = stats.t.sf(np.abs(q_tstat), q_count - 1) * 2
        
        q_stats = pd.DataFrame(index=daily_q_ret.columns, columns=['Mean', 'Std. Error', 't-stat', 'p-value', 'ann_ret', 'ann_vol', 'sharpe', 'max_dd', 'calmar'])
        q_stats.update(pd.DataFrame({
            'Mean': q_mean,
            'Std. Error': q_se,
            't-stat': q_tstat,
            'p-value': q_pvalue
        }))
        
        # Long-Short stats (Top - Bottom)
        max_q = daily_q_ret.columns.max()
        min_q = daily_q_ret.columns.min()
        if pd.notna(max_q) and pd.notna(min_q):
            ls_daily = daily_q_ret[max_q] - daily_q_ret[min_q]
            ls_mean = ls_daily.mean()
            ls_std = ls_daily.std()
            ls_count = ls_daily.count()
            ls_se = ls_std / np.sqrt(ls_count)
            ls_tstat = ls_mean / ls_se
            ls_pvalue = stats.t.sf(np.abs(ls_tstat), ls_count - 1) * 2
            
            # Additional Portfolio Metrics for LS
            ls_perf = calculate_portfolio_performance(ls_daily)
            
            ls_data = {
                'Mean': ls_mean,
                'Std. Error': ls_se,
                't-stat': ls_tstat,
                'p-value': ls_pvalue,
                'ann_ret': ls_perf['ann_ret'],
                'ann_vol': ls_perf['ann_vol'],
                'sharpe': ls_perf['sharpe'],
                'max_dd': ls_perf['max_dd'],
                'calmar': ls_perf['calmar']
            }
            
            q_stats.loc['Long-Short'] = pd.Series(ls_data)
            
        all_stats[col] = q_stats
        
    return all_stats

def factor_rre(factor_data: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the Reciprocal Rank Evaluation (RRE) score, which measures the stability
    of factor rankings over time.
    
    RRE = 1 / (1 + KL_Divergence(Rank_Prob || Rank_Prob_Prev))
    where Rank_Prob = Rank / Sum(Rank)
    
    Returns a DataFrame with RRE scores for each horizon (though RRE is typically horizon-agnostic 
    and calculated on the factor values themselves, we return it as a single column 'RRE').
    """
    # Unstack factor to Date x Asset
    f_wide = factor_data['factor'].unstack()
    
    # Calculate Ranks
    ranks = f_wide.rank(axis=1)
    
    # Calculate Probabilities: P = Rank / Sum(Rank)
    # Sum of ranks for N assets = N*(N+1)/2
    row_sums = ranks.sum(axis=1)
    probs = ranks.div(row_sums, axis=0)
    
    # Previous Probabilities
    probs_prev = probs.shift(1)
    
    # KL Divergence: Sum( P * log(P / Q) )
    # Add epsilon to avoid division by zero or log of zero
    eps = 1e-8
    
    # We only compute for days where we have both current and prev
    valid_idx = probs.index.intersection(probs_prev.index)
    
    p = probs.loc[valid_idx]
    q = probs_prev.loc[valid_idx]
    
    # Compute KL divergence per day
    kl_div = (p * np.log((p + eps) / (q + eps))).sum(axis=1, min_count=1)
    
    # RRE = 1 / (1 + KL)
    rre_series = 1 / (1 + kl_div)
    
    return rre_series

def compute_rolling_ic_stats(ic: pd.DataFrame, windows: List[int] = [252, 504, 756, 1260]) -> pd.DataFrame:
    """
    Computes rolling IC statistics for different window sizes.
    
    Args:
        ic: DataFrame of daily IC values (columns are horizons like '1D', '5D', etc.)
        windows: List of rolling window sizes in trading days (default: 1Y, 2Y, 3Y, 5Y)
    
    Returns:
        DataFrame with rolling IC mean, std, and IR for each window size
        Rows = metrics (IC Mean, IC Std, IR), Columns = windows (1Y, 2Y, etc.)
    """
    window_names = {252: '1Y', 504: '2Y', 756: '3Y', 1260: '5Y'}
    
    # We'll create a simplified table: show only the primary horizon (20D or first available)
    # for each window, to keep output clean
    primary_horizon = '20D' if '20D' in ic.columns else ic.columns[0]
    
    results = []
    for window in windows:
        if len(ic) < window:
            continue
        window_name = window_names.get(window, f'{window}D')
        
        rolling_mean = ic[primary_horizon].rolling(window=window).mean()
        rolling_std = ic[primary_horizon].rolling(window=window).std()
        
        # Get the latest (most recent) rolling values
        latest_mean = rolling_mean.iloc[-1]
        latest_std = rolling_std.iloc[-1]
        latest_ir = latest_mean / latest_std if latest_std > 0 else np.nan
        
        # Also compute the min/max of rolling IC to show range
        min_rolling = rolling_mean.min()
        max_rolling = rolling_mean.max()
        
        results.append({
            'Window': window_name,
            'IC Mean': latest_mean,
            'IC Std': latest_std,
            'IR': latest_ir,
            'Min IC': min_rolling,
            'Max IC': max_rolling
        })
    
    if not results:
        return pd.DataFrame()
    
    result_df = pd.DataFrame(results).set_index('Window').T
    return result_df


def compute_ic_trend(ic: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes the trend of IC over time using linear regression.
    A negative slope indicates factor decay, positive indicates improvement.
    
    Returns:
        Dict with slope, interpretation, and R-squared for each horizon
    """
    results = {}
    
    for col in ic.columns:
        ic_series = ic[col].dropna()
        if len(ic_series) < 20:
            results[col] = {'slope': np.nan, 'r_squared': np.nan, 'interpretation': 'Insufficient data'}
            continue
        
        # Use day index as X
        x = np.arange(len(ic_series))
        y = ic_series.values
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        # Annualized slope (slope per 252 trading days)
        annual_slope = slope * 252
        
        # Interpretation
        if abs(annual_slope) < 0.005:
            interpretation = "Stable"
        elif annual_slope > 0.02:
            interpretation = "Strong Improvement"
        elif annual_slope > 0.005:
            interpretation = "Mild Improvement"
        elif annual_slope < -0.02:
            interpretation = "Strong Decay"
        else:
            interpretation = "Mild Decay"
        
        results[col] = {
            'slope': slope,
            'annual_slope': annual_slope,
            'r_squared': r_value ** 2,
            'p_value': p_value,
            'interpretation': interpretation
        }
    
    return results


def compute_yearly_ic_breakdown(ic: pd.DataFrame) -> pd.DataFrame:
    """
    Computes IC statistics broken down by year.
    
    Returns:
        DataFrame with IC Mean, IR, and Winrate for each year and horizon
    """
    ic_with_year = ic.copy()
    ic_with_year['year'] = ic_with_year.index.year
    
    yearly_stats = []
    for year, year_data in ic_with_year.groupby('year'):
        year_ic = year_data.drop(columns=['year'])
        
        ic_mean = year_ic.mean()
        ic_std = year_ic.std()
        ic_ir = ic_mean / ic_std
        ic_winrate = (year_ic > 0).sum() / len(year_ic)
        
        for col in year_ic.columns:
            yearly_stats.append({
                'Year': int(year),
                'Horizon': col,
                'IC Mean': ic_mean[col],
                'IC Std': ic_std[col],
                'IR': ic_ir[col],
                'Winrate': ic_winrate[col],
                'N Days': len(year_ic)
            })
    
    return pd.DataFrame(yearly_stats)


def compute_regime_analysis(ic: pd.DataFrame, market_returns: Optional[pd.Series] = None) -> Dict[str, pd.DataFrame]:
    """
    Computes IC statistics in different market regimes (bull vs bear).
    
    If market_returns is not provided, uses the IC index to infer regimes
    based on recent performance.
    
    Args:
        ic: DataFrame of daily IC values
        market_returns: Optional Series of market returns to define regimes
    
    Returns:
        Dict with 'bull' and 'bear' DataFrames containing IC stats for each regime
    """
    if market_returns is not None:
        # Use provided market returns to define regimes
        aligned_returns = market_returns.reindex(ic.index).dropna()
        # 60-day rolling return to define regime
        rolling_ret = aligned_returns.rolling(60).sum()
        bull_mask = rolling_ret > 0
        bear_mask = rolling_ret <= 0
    else:
        # Simple regime: split by median IC or use time-based split
        # Use rolling 60-day IC as regime indicator (high IC period vs low IC period)
        rolling_ic = ic.iloc[:, 0].rolling(60).mean()  # Use first horizon
        median_rolling = rolling_ic.median()
        bull_mask = rolling_ic >= median_rolling
        bear_mask = rolling_ic < median_rolling
    
    # Reindex masks to IC index
    bull_mask = bull_mask.reindex(ic.index).fillna(False)
    bear_mask = bear_mask.reindex(ic.index).fillna(False)
    
    results = {}
    
    for regime_name, mask in [('Up Market', bull_mask), ('Down Market', bear_mask)]:
        regime_ic = ic[mask]
        if len(regime_ic) < 5:
            continue
        
        regime_stats = pd.DataFrame({
            'IC Mean': regime_ic.mean(),
            'IC Std': regime_ic.std(),
            'IR': regime_ic.mean() / regime_ic.std(),
            'Winrate': (regime_ic > 0).sum() / len(regime_ic),
            'N Days': len(regime_ic)
        })
        results[regime_name] = regime_stats
    
    return results


def compute_stability_metrics(factor_data: pd.DataFrame, ic: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """
    Main entry point for in-depth stability analysis.
    
    Computes:
    - Multi-window rolling IC stats (1Y, 2Y, 3Y, 5Y)
    - Year-by-year IC breakdown
    - IC trend analysis (linear regression)
    - Regime analysis (bull vs bear market)
    
    Args:
        factor_data: The factor data DataFrame with MultiIndex (date, asset)
        ic: Optional pre-computed IC DataFrame. If not provided, will compute it.
    
    Returns:
        Dict containing all stability metrics
    """
    if ic is None:
        ic = factor_information_coefficient(factor_data)
    
    # 1. Multi-window rolling IC
    rolling_stats = compute_rolling_ic_stats(ic)
    
    # 2. Year-by-year breakdown
    yearly_breakdown = compute_yearly_ic_breakdown(ic)
    
    # 3. IC Trend
    ic_trend = compute_ic_trend(ic)
    
    # 4. Regime analysis (using IC itself as proxy for market regime)
    regime_stats = compute_regime_analysis(ic)
    
    # 5. Calculate overall stability score
    # Stability score = ratio of min rolling IC mean to max rolling IC mean
    # Higher = more stable
    stability_scores = {}
    for col in ic.columns:
        rolling_mean = ic[col].rolling(252).mean().dropna()
        if len(rolling_mean) > 0:
            min_val = rolling_mean.min()
            max_val = rolling_mean.max()
            # Use absolute values for consistency assessment
            if max_val != 0:
                consistency = min_val / max_val if min_val * max_val > 0 else 0
            else:
                consistency = 0
            stability_scores[col] = consistency
        else:
            stability_scores[col] = np.nan
    
    return {
        'rolling_stats': rolling_stats,
        'yearly_breakdown': yearly_breakdown,
        'ic_trend': ic_trend,
        'regime_stats': regime_stats,
        'stability_scores': pd.Series(stability_scores)
    }


def compute_performance_metrics(factor_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Main entry point to compute all performance metrics.
    """
    ic = factor_information_coefficient(factor_data)
    mean_ret = mean_return_by_quantile(factor_data)
    alpha_beta = factor_alpha_beta(factor_data)
    
    # Monotonicity
    mono_score = monotonicity_score(mean_ret)
    
    # Quantile Turnover (Average across all days)
    q_turnover = average_quantile_turnover(factor_data, period=1)
    
    # RRE (Rank Stability)
    rre_series = factor_rre(factor_data)
    rre_mean = rre_series.mean()
    
    # Calculate additional IC metrics
    ic_winrate = {}
    ic_median = {}
    ic_skew = {}
    ic_max_drawdown = {}
    
    for col in ic.columns:
        ic_series = ic[col].dropna()
        
        # IC Winrate: percentage of positive IC values
        ic_winrate[col] = (ic_series > 0).sum() / len(ic_series) if len(ic_series) > 0 else np.nan
        
        # IC Median
        ic_median[col] = ic_series.median()
        
        # IC Skewness
        ic_skew[col] = ic_series.skew()
        
        # IC Max Drawdown: max drawdown of cumulative IC
        cumulative_ic = ic_series.cumsum()
        running_max = cumulative_ic.expanding().max()
        drawdown = cumulative_ic - running_max
        ic_max_drawdown[col] = drawdown.min()
    
    # Summary IC stats
    ic_summary = pd.DataFrame({
        'IC Mean': ic.mean(),
        'IC Std.': ic.std(),
        'Risk-Adjusted IC (IR)': ic.mean() / ic.std(),
        't-stat(IC)': (ic.mean() / ic.std()) * np.sqrt(len(ic)),
        'IC P-value': [stats.t.sf(np.abs(t), len(ic)-1)*2 for t in (ic.mean() / ic.std()) * np.sqrt(len(ic))],
        'IC Winrate': pd.Series(ic_winrate),
        'IC Median': pd.Series(ic_median),
        'IC Skew': pd.Series(ic_skew),
        'IC Max Drawdown': pd.Series(ic_max_drawdown),
    }).T
    
    # Turnover (1D stability)
    autocorr = factor_rank_autocorrelation(factor_data, period=1)
    
    # Detailed Quantile Stats
    q_stats = quantile_performance_stats(factor_data)
    
    # Portfolio returns for Top, Bottom, and LS (for plotting)
    date_col = [level_name for level_name in factor_data.index.names if 'date' in str(level_name).lower()][0]
    daily_q_ret = factor_data.groupby([date_col, 'factor_quantile'])[ic.columns[0]].mean().unstack()
    max_q = daily_q_ret.columns.max()
    min_q = daily_q_ret.columns.min()
    
    port_returns = pd.DataFrame({
        'Top': daily_q_ret[max_q],
        'Bottom': daily_q_ret[min_q],
        'Long-Short': daily_q_ret[max_q] - daily_q_ret[min_q]
    })

    return {
        'ic': ic,
        'ic_summary': ic_summary,
        'mean_ret': mean_ret,
        'mono_score': mono_score,
        'quantile_turnover': q_turnover,
        'rre': rre_mean,
        'alpha_beta': alpha_beta,
        'autocorr': autocorr,
        'autocorr_mean': autocorr.mean(),
        'q_stats': q_stats,
        'port_returns': port_returns
    }

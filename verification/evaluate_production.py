import pandas as pd
import numpy as np
import importlib
import random
import os
import sys

# Ensure parent directory is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from alpha191.utils import (
    load_benchmark_csv,
    get_benchmark_members,
    parallel_load_stocks_with_alpha
)
from assessment import (
    get_clean_factor_and_forward_returns,
    compute_performance_metrics,
    compute_stability_metrics
)
from verification.alpha_combination_model import IncrementalCombinationModel

def get_alpha_func(alpha_name: str):
    """Get the alpha function by name."""
    try:
        mod_name = alpha_name.replace("_", "")
        alpha_module = importlib.import_module(f"alpha191.{mod_name}")
        func_name = alpha_name
        alpha_func = getattr(alpha_module, func_name)
        return alpha_func
    except (ImportError, AttributeError) as e:
        print(f"Failed to load {alpha_name}: {e}")
        return None

def main():
    benchmark = "hs300"
    print(f"Loading IncrementalCombinationModel...")

    ic_path = "alpha_ic.csv"
    corr_path = "alpha_correlation.csv"
    if not os.path.exists(ic_path):
        ic_path = os.path.join("verification", "alpha_ic.csv")
        corr_path = os.path.join("verification", "alpha_correlation.csv")

    model = IncrementalCombinationModel(ic_file=ic_path, corr_file=corr_path)

    # 1. Get the pool and weights
    random.seed(0)
    all_alphas = list(model.ic_dict.keys())
    random.shuffle(all_alphas)

    print("Running incremental optimization...")
    pool, weights, combined_ic = model.incremental_optimization(all_alphas, max_pool_size=10)
    print(f"Optimal Pool: {pool}")
    print(f"Weights: {weights}")
    print(f"Theoretical Combined IC: {combined_ic:.4f}")

    # Let's compute individual ICs for the pool for reference
    for a in pool:
        print(f"{a}: {model.ic_dict[a]:.4f}")

    # Since the alpha combination IC assumes a linear model with *cross-sectional* standardization
    # and our target was forward 20-day returns, the theoretical combined IC was over 0.6.
    # However, in practice across the time-series, this combination is very volatile and the Long-Short return is negative.

    # 2. Build combined factor (trying cross-sectional ranking per day)
    # The assessment module handles cross-sectional grouping later, but here we just pass raw signals.
    # Let's just output this result to see.
    funcs = []
    for name in pool:
        f = get_alpha_func(name)
        if f:
            funcs.append((f, name))

    def combined_alpha_func(df):
        combined_series = None
        for i, (func, name) in enumerate(funcs):
            w = weights[i]
            res = func(df)

            # Since df is time-series per stock, cross-sectional ranking here is impossible.
            # We must just sum the weighted raw values, but alpha scales differ wildly.
            # Time-series standardization (z-score) is the best proxy per-stock.
            if isinstance(res, pd.Series):
                # Apply a rolling window Z-score to avoid look-ahead bias
                roll = res.rolling(window=252, min_periods=20)
                mean_val = roll.mean()
                std_val = roll.std()
                # To prevent division by zero, replace 0 std with NaN
                std_val = std_val.replace(0, np.nan)
                res = (res - mean_val) / std_val

            if combined_series is None:
                combined_series = w * res
            else:
                combined_series = combined_series.add(w * res, fill_value=0)

        return combined_series

    # 3. Evaluate using assessment module
    horizons = [1, 5, 10, 20]
    m_quantiles = 10

    print(f"\nLoading stock data for benchmark {benchmark}...")
    codes = get_benchmark_members(benchmark)
    benchmark_df = load_benchmark_csv(benchmark)
    timeline = benchmark_df.index

    factor_results, price_results = parallel_load_stocks_with_alpha(
        codes, combined_alpha_func, benchmark, n_jobs=-1, show_progress=True
    )

    if not factor_results:
        print("No data loaded successfully.")
        return

    print("Formatting data for assessment...")
    factor_matrix = pd.DataFrame(factor_results, dtype=np.float32).reindex(timeline)
    price_matrix = pd.DataFrame(price_results, dtype=np.float32).reindex(timeline)

    print("Computing clean factor and forward returns...")
    factor_data_tuple = get_clean_factor_and_forward_returns(
        factor_matrix,
        price_matrix,
        periods=horizons,
        quantiles=m_quantiles,
        max_loss=0.35,
        filter_zscore=None,
        return_wide=True
    )
    if isinstance(factor_data_tuple, tuple) and len(factor_data_tuple) == 3:
        factor_data, f_wide, q_wide = factor_data_tuple
    else:
        factor_data = factor_data_tuple
        f_wide, q_wide = None, None

    print("Computing performance metrics...")
    metrics = compute_performance_metrics(factor_data, f_wide=f_wide, q_wide=q_wide)

    print("\n" + "="*80)
    print(" PRODUCTION COMBINED ALPHA EVALUATION ".center(80, "="))
    print("="*80)

    pd.options.display.float_format = '{:.6f}'.format

    # IC Summary
    print("\n[1] Information Coefficient (IC) Summary")
    print(metrics['ic_summary'].to_string())

    # Quantile metrics (focus on 20D)
    print("\n[2] Quantile Return Stats & Long-Short Performance (20D)")
    if '20D' in metrics['q_stats']:
        q_stats_20d = metrics['q_stats']['20D']
        print(q_stats_20d.to_string())

        ls_row = q_stats_20d.loc['Long-Short'] if 'Long-Short' in q_stats_20d.index else None
        if ls_row is not None:
            print("\n[ L-S PORTFOLIO PERFORMANCE (20D) ]")
            print(f"Annualized Return: {ls_row['ann_ret']*100:.2f}%")
            print(f"Annualized Vol:    {ls_row['ann_vol']*100:.2f}%")
            print(f"Sharpe Ratio:      {ls_row['sharpe']:.2f}")
            print(f"Max Drawdown:      {ls_row['max_dd']*100:.2f}%")
            print(f"Calmar Ratio:      {ls_row['calmar']:.2f}")
    else:
        print("20D stats not available.")

    # Turnover
    print("\n[3] Turnover By Quantile")
    if 'quantile_turnover' in metrics:
        print(metrics['quantile_turnover'].to_frame().T.to_string())

    print("\n" + "="*80)

if __name__ == "__main__":
    main()

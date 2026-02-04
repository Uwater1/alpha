import pandas as pd
import numpy as np
import sys
import importlib
from pathlib import Path
from typing import List, Dict, Any
from alpha191.utils import (
    load_benchmark_csv, 
    get_benchmark_members, 
    format_alpha_name,
    parallel_load_stocks_with_alpha
)
from assessment import get_clean_factor_and_forward_returns, compute_performance_metrics

def run_group_test(alpha_name: str, horizon: int = 20, benchmark: str = "hs300", m_quantiles: int = 10, plot: bool = False, n_jobs: int = -1):
    """
    Perform Group/Quantile Return Test on an alpha using the new assessment module.
    """
    print(f"Running Group Test for {alpha_name} ...")
    print(f"  Benchmark: {benchmark} || Horizon:   {horizon} days || Quantiles: {m_quantiles}")
    print("-" * 40)
    
    # Import alpha function
    try:
        alpha_module = importlib.import_module(f"alpha191.{alpha_name}")
        func_name = alpha_name[:5] + "_" + alpha_name[5:]
        alpha_func = getattr(alpha_module, func_name)
    except (ImportError, AttributeError) as e:
        try:
            alpha_func = getattr(alpha_module, alpha_name)
        except (AttributeError, NameError):
            print(f"Error importing {alpha_name}: {e}")
            return

    codes = get_benchmark_members(benchmark)
    
    # Load benchmark data to get the full timeline
    benchmark_df = load_benchmark_csv(benchmark)
    timeline = benchmark_df.index
    
    # Use parallel loading for significant speedup
    factor_results, price_results = parallel_load_stocks_with_alpha(
        codes, alpha_func, benchmark, n_jobs=n_jobs, show_progress=True
    )

    if not factor_results:
        print("No data available for testing.")
        return

    factor_matrix = pd.DataFrame(factor_results).reindex(timeline)
    price_matrix = pd.DataFrame(price_results).reindex(timeline)
    
    # Use assessment module
    factor_data = get_clean_factor_and_forward_returns(
        factor_matrix,
        price_matrix,
        periods=[horizon],
        quantiles=m_quantiles
    )
    
    results = compute_performance_metrics(factor_data)
    mean_returns = results['mean_ret'][f"{horizon}D"]
    
    # Summary Table
    print("\n" + "="*60)
    print(f"{'Group Test Results for ' + alpha_name.upper():^60}")
    print("="*60)
    print(f"{'Group':<10} | {'Mean Ret (%)':<15}")
    print("-" * 60)
    for q in range(1, m_quantiles + 1):
        m_ret = mean_returns.get(q, np.nan) * 100
        print(f"{q:<10} | {m_ret:>13.4f}%")
    
    # Portfolio Returns
    # Long-Short spread calculation (Top vs Bottom quantile)
    ls_spread = mean_returns.get(m_quantiles, np.nan) - mean_returns.get(1, np.nan)
    
    print("-" * 60)
    print(f"{'Long-Short (T-B)':<15} Mean Spread: {ls_spread*100:>.4f}%")
    print("="*60)

    if plot:
        from assessment import plot_quantile_returns_bar
        import matplotlib.pyplot as plt
        plot_file = f"{alpha_name}_group_returns.png"
        plot_quantile_returns_bar(results['mean_ret'], title=f"Mean Return by Quantile ({horizon}D)")
        plt.savefig(plot_file)
        print(f"Group returns plot saved to {plot_file}")
        plt.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Group/Quantile Return Test")
    parser.add_argument("alpha", help="Alpha name (e.g., 1 or alpha001)")
    parser.add_argument("--horizon", type=int, default=20, help="Forward return horizon (default: 20)")
    parser.add_argument("--benchmark", default="hs300", help="Benchmark (hs300, zz500, zz800)")
    parser.add_argument("--quantiles", type=int, default=10, help="Number of groups (default: 10)")
    parser.add_argument("--plot", action="store_true", help="Generate group returns plot")
    parser.add_argument("--jobs", type=int, default=-1, help="Number of parallel workers (default: -1 = all CPUs)")
    
    args = parser.parse_args()
    
    alpha = format_alpha_name(args.alpha)
    run_group_test(alpha, args.horizon, args.benchmark, args.quantiles, args.plot, args.jobs)

import pandas as pd
import numpy as np
import sys
import importlib
from pathlib import Path
from typing import List, Dict, Any, Union

from alpha191.utils import (
    load_benchmark_csv, 
    get_benchmark_members, 
    format_alpha_name,
    parallel_load_stocks_with_alpha
)
from assessment import (
    get_clean_factor_and_forward_returns, 
    compute_performance_metrics,
    plot_quantile_returns_bar,
    plot_cumulative_returns_comparison
)

def run_group_test(alpha_name: str, horizons: List[int] = [20], benchmark: str = "hs300", m_quantiles: int = 10, plot: bool = False, n_jobs: int = -1):
    """
    Perform Group/Quantile Return Test on an alpha using the new assessment module.
    Supports multiple horizons and professional standard terminal output.
    """
    header = f" ALPHA ASSESSMENT: {alpha_name.upper()} "
    print("\n" + "="*80)
    print(f"{header:^80}")
    print("="*80)
    print(f"Benchmark: {benchmark:<10} | Quantiles: {m_quantiles:<5} | Horizons: {str(horizons)}")
    print("-" * 80)
    
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
    
    # Use parallel loading with optimized memory usage
    print("Loading stock data and calculating alpha...")
    factor_results, price_results = parallel_load_stocks_with_alpha(
        codes, alpha_func, benchmark, n_jobs=n_jobs, show_progress=True
    )

    if not factor_results:
        print("No data available for testing.")
        return

    # Optimize memory by using float32 and avoiding unnecessary copies
    factor_matrix = pd.DataFrame(factor_results, dtype=np.float32).reindex(timeline)
    price_matrix = pd.DataFrame(price_results, dtype=np.float32).reindex(timeline)
    
    # Use assessment module with optimized parameters
    factor_data = get_clean_factor_and_forward_returns(
        factor_matrix,
        price_matrix,
        periods=horizons,
        quantiles=m_quantiles,
        max_loss=0.35  # Allow more data loss for speed
    )
    
    results = compute_performance_metrics(factor_data)
    
    # Display Results for each horizon
    for h in horizons:
        h_str = f"{h}D"
        mean_returns = results['mean_ret'][h_str]
        q_stats = results['q_stats'][h_str]
        turnover = results['quantile_turnover']
        
        print(f"\n[ QUANTILE RETURNS & STATS ({h_str}) ]")
        print("-" * 85)
        print(f"{'Group':^8} | {'Mean Ret (%)':^15} | {'Std Error (%)':^15} | {'t-stat':^10} | {'p-value':^10} | {'Turnover':^10}")
        print("-" * 85)
        
        for q in range(1, m_quantiles + 1):
            m_ret = q_stats.loc[q, 'Mean'] * 100
            m_se = q_stats.loc[q, 'Std. Error'] * 100
            t_stat = q_stats.loc[q, 't-stat']
            p_val = q_stats.loc[q, 'p-value']
            q_turnover = turnover.get(q, np.nan)
            
            print(f"{q:^8} | {m_ret:15.4f}% | {m_se:15.4f}% | {t_stat:10.2f} | {p_val:10.4f} | {q_turnover:10.4f}")
        
        # Long-Short Row
        ls_row = q_stats.loc['Long-Short']
        print("-" * 85)
        print(f"{'L-S (T-B)':^8} | {ls_row['Mean']*100:14.4f}% | {ls_row['Std. Error']*100:15.4f}% | {ls_row['t-stat']:10.2f} | {ls_row['p-value']:10.4f} | {'-':^10}")
        print("-" * 85)
        
        # Portfolio Metrics Summary
        print(f"\n[ L-S PORTFOLIO PERFORMANCE ({h_str}) ]")
        metrics = [
            ("Annualized Return", f"{ls_row['ann_ret']*100:.2f}%"),
            ("Annualized Vol", f"{ls_row['ann_vol']*100:.2f}%"),
            ("Sharpe Ratio", f"{ls_row['sharpe']:.2f}"),
            ("Max Drawdown", f"{ls_row['max_dd']*100:.2f}%"),
            ("Calmar Ratio", f"{ls_row['calmar']:.2f}"),
            ("Monotonicity", f"{results['mono_score'][h_str]:.4f}"),
            ("RRE (Stability)", f"{results.get('rre', np.nan):.4f}")
        ]
        
        for name, val in metrics:
            print(f"{name:<20}: {val:>10}")
        
        print("\n" + "-" * 40)

        if plot:
            import matplotlib.pyplot as plt
            # 1. Bar Chart with Error Bars
            fig, ax = plt.subplots(figsize=(10, 6))
            plot_quantile_returns_bar(
                pd.DataFrame(mean_returns), 
                errors=pd.DataFrame(q_stats['Std. Error'].drop('Long-Short', errors='ignore')),
                ax=ax,
                title=f"Mean Return by Quantile ({h_str}) with SE bars"
            )
            bar_plot_file = f"{alpha_name}_{h_str}_group_returns.png"
            plt.savefig(bar_plot_file)
            print(f"Bar plot saved to {bar_plot_file}")
            plt.close()
            
    # Comparison plot for the first horizon
    if plot:
        import matplotlib.pyplot as plt
        plot_cumulative_returns_comparison(results['port_returns'], title=f"Cumulative Returns: Top vs Bottom vs LS ({horizons[0]}D)")
        cum_plot_file = f"{alpha_name}_cumulative_returns.png"
        plt.savefig(cum_plot_file)
        print(f"Cumulative returns plot saved to {cum_plot_file}")
        plt.close()
    
    print("\n" + "="*80)
    print(f"{'ASSESSMENT COMPLETE':^80}")
    print("="*80 + "\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Professional Alpha Assessment - Group/Quantile Return Test")
    parser.add_argument("alpha", help="Alpha name (e.g., 1 or alpha001)")
    parser.add_argument("--horizon", type=str, default="20", help="Forward return horizon(s), comma-separated (default: 20)")
    parser.add_argument("--benchmark", default="hs300", help="Benchmark (hs300, zz500, zz800)")
    parser.add_argument("--quantiles", type=int, default=10, help="Number of groups (default: 10)")
    parser.add_argument("--plot", action="store_true", help="Generate enhanced plots")
    parser.add_argument("--jobs", type=int, default=-1, help="Number of parallel workers (default: -1 = all CPUs)")
    
    args = parser.parse_args()
    
    alpha = format_alpha_name(args.alpha)
    # Parse horizons
    try:
        horizons = [int(h.strip()) for h in args.horizon.split(',')]
    except ValueError:
        raise ValueError("Error: horizon must be a comma-separated list of integers.")
        
    run_group_test(alpha, horizons, args.benchmark, args.quantiles, args.plot, args.jobs)

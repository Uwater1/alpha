import pandas as pd
import numpy as np
import sys
import importlib
from pathlib import Path
from typing import List, Dict, Any
from alpha191.utils import load_stock_csv, load_benchmark_csv, get_benchmark_members, format_alpha_name
from assessment import get_clean_factor_and_forward_returns, compute_performance_metrics

def assess_alpha(alpha_name: str, benchmark: str = "zz800", horizon: int = 20, plot: bool = False):
    """Assess an alpha using the new assessment module."""
    print(f"Assessing {alpha_name} on {benchmark} with horizon {horizon} days ...")
    
    # Import alpha function
    try:
        alpha_module = importlib.import_module(f"alpha191.{alpha_name}")
        func_name = alpha_name[:5] + "_" + alpha_name[5:]
        alpha_func = getattr(alpha_module, func_name)
    except (ImportError, AttributeError) as e:
        print(f"Error importing {alpha_name}: {e}")
        try:
            alpha_func = getattr(alpha_module, alpha_name)
        except AttributeError:
            return
    
    codes = get_benchmark_members(benchmark)
    
    factor_results = {}
    price_results = {}
    
    # Load benchmark data to get the full timeline
    benchmark_df = load_benchmark_csv(benchmark)
    timeline = benchmark_df.index
    
    total = len(codes)
    failed_codes = []
    for i, code in enumerate(codes):
        try:
            df = load_stock_csv(code, benchmark=benchmark)
            df['benchmark_close'] = benchmark_df['close'].reindex(df.index)
            df['benchmark_open'] = benchmark_df['open'].reindex(df.index)
            
            alpha_series = alpha_func(df)
            
            factor_results[code] = alpha_series.astype(np.float32)
            price_results[code] = df['close'].astype(np.float32)
            
        except Exception as e:
            failed_codes.append((code, str(e)))
            continue

    if not factor_results:
        print("No data loaded successfully.")
        return

    factor_matrix = pd.DataFrame(factor_results).reindex(timeline)
    price_matrix = pd.DataFrame(price_results).reindex(timeline)
    
    # Use assessment module
    factor_data = get_clean_factor_and_forward_returns(
        factor_matrix,
        price_matrix,
        periods=[horizon],
        quantiles=10
    )
    
    results = compute_performance_metrics(factor_data)
    
    # Print Results
    print("\n" + "="*40)
    print(f"  Assessment Results for {alpha_name.upper()}")
    print("="*40)
    
    ic_summary = results['ic_summary']
    print(ic_summary.to_string())
    
    print("-" * 40)
    print(f"IC Mean:         {results['ic'].mean().iloc[0]:.6f}")
    print(f"IC Std:          {results['ic'].std().iloc[0]:.6f}")
    print(f"ICIR:            {results['ic_summary'].loc['Risk-Adjusted IC (IR)'].iloc[0]:.6f}")
    print(f"Rank Autocorr:   {results['autocorr_mean']:.6f}")
    
    # Detailed Mean Returns by Quantile
    print("\nMean Return by Quantile:")
    print(results['mean_ret'].to_string())
    print("="*40)
    
    if plot:
        from assessment import create_full_tear_sheet
        plot_file = f"{alpha_name}_tear_sheet.png"
        create_full_tear_sheet(factor_data, output_path=plot_file)
    
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Assess Alpha Factor Performance")
    parser.add_argument("alpha", help="Alpha name (e.g., 1 or alpha001)")
    parser.add_argument("--horizon", type=int, default=20, help="Forward return horizon (default: 20)")
    parser.add_argument("--benchmark", default="zz800", help="Benchmark (hs300, zz500, zz800)")
    parser.add_argument("--plot", action="store_true", help="Generate tear sheet plot")
    
    args = parser.parse_args()
    
    alpha = format_alpha_name(args.alpha)
    assess_alpha(alpha, args.benchmark, args.horizon, args.plot)

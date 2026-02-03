import pandas as pd
import numpy as np
import sys
import importlib
from pathlib import Path
from typing import List, Dict, Any
from alpha191.utils import load_stock_csv, load_benchmark_csv, get_benchmark_members, format_alpha_name

def run_group_test(alpha_name: str, horizon: int = 20, benchmark: str = "hs300", m_quantiles: int = 10):
    """
    Perform Group/Quantile Return Test on an alpha.
    Divide stocks into m quantiles and calculate group returns.
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
    
    factor_results = {}
    return_results = {}
    
    print(f"Loading data for {len(codes)} stocks...")
    for i, code in enumerate(codes):
        try:
            df = load_stock_csv(code, benchmark=benchmark)
            
            # Align with benchmark
            df['benchmark_close'] = benchmark_df['close'].reindex(df.index)
            df['benchmark_open'] = benchmark_df['open'].reindex(df.index)
            
            # Calculate alpha and returns
            alpha_series = alpha_func(df)
            forward_returns = (df['close'].shift(-horizon) / df['close']) - 1
            
            factor_results[code] = alpha_series.astype(np.float32)
            return_results[code] = forward_returns.astype(np.float32)
        except Exception:
            continue

    if not factor_results:
        print("No data available for testing.")
        return

    factor_matrix = pd.DataFrame(factor_results).reindex(timeline).astype(np.float32)
    return_matrix = pd.DataFrame(return_results).reindex(timeline).astype(np.float32)
    
    # Group results
    group_returns_list = []
    
    for date in timeline:
        alphas = factor_matrix.loc[date].dropna()
        returns = return_matrix.loc[date].reindex(alphas.index).dropna()
        
        if len(alphas) < m_quantiles:
            continue
            
        # Re-align alphas to only includes stocks with returns
        alphas = alphas.reindex(returns.index)
        
        try:
            # Use rank for grouping to avoid issues with duplicates/distribution
            # Divide into m quantiles
            labels = range(1, m_quantiles + 1)
            groups = pd.qcut(alphas, q=m_quantiles, labels=labels, duplicates='drop')
            
            # Calculate mean return for each group
            daily_group_returns = returns.groupby(groups).mean()
            group_returns_list.append(daily_group_returns)
        except Exception:
            continue

    if not group_returns_list:
        print("Grouping failed (likely insufficient data across time).")
        return

    # Combine into a single DataFrame
    group_returns_df = pd.concat(group_returns_list, axis=1, keys=range(len(group_returns_list))).T
    
    # Summary Statistics
    # To avoid overlapping returns in cumulative return calculation, 
    # we should sample the returns every 'horizon' days.
    sampled_returns = group_returns_df.iloc[::horizon]
    
    mean_returns = group_returns_df.mean()
    win_rates = (group_returns_df > 0).mean()
    
    # Cumulative returns: (1+r1)*(1+r2)... - 1 for sampled non-overlapping periods
    cum_returns = (1 + sampled_returns).prod() - 1
    
    long_short_returns = group_returns_df[m_quantiles] - group_returns_df[1]
    ls_mean = long_short_returns.mean()
    ls_ir = ls_mean / long_short_returns.std() if long_short_returns.std() != 0 else 0
    
    # Output Table
    print("\n" + "="*60)
    print(f"{'Group Test Results for ' + alpha_name.upper():^60}")
    print("="*60)
    print(f"{'Group':<10} | {'Mean Ret (%)':<15} | {'Win Rate (%)':<15} | {'Cum Ret (%)':<15}")
    print("-" * 60)
    for i in range(1, m_quantiles + 1):
        m_ret = mean_returns.get(i, np.nan) * 100
        wr = win_rates.get(i, np.nan) * 100
        cr = cum_returns.get(i, np.nan) * 100
        print(f"{i:<10} | {m_ret:>13.4f}% | {wr:>13.2f}% | {cr:>13.2f}%")
    
    print("-" * 60)
    print(f"{'Long-Short (T-B)':<15} Mean: {ls_mean*100:>.4f}% | IR: {ls_ir:>.4f}")
    print("="*60)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python grouptest.py <alpha_name> [period] [range] [quantile]")
        print("  alpha_name:  Number (1-191) or format 'alpha001'")
        print("  period:      Forward return horizon (default: 20)")
        print("  range:       Benchmark (hs300, zz500, zz800) (default: hs300)")
        print("  quantile:    Number of groups (default: 10)")
        sys.exit(1)

    alpha = format_alpha_name(sys.argv[1])
    period = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    benchmark = sys.argv[3] if len(sys.argv) > 3 else "hs300"
    quantiles = int(sys.argv[4]) if len(sys.argv) > 4 else 10

    run_group_test(alpha, period, benchmark, quantiles)

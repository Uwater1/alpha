#!/usr/bin/env python3
"""
Alpha Performance Assessment Script
===================================
Scans all 191 alphas and computes comprehensive performance metrics.
Optimized for speed by pre-loading all stock data into memory.

Output:
    alpha_performance.csv: A wide CSV containing detailed metrics for each alpha.
"""

import pandas as pd
import numpy as np
import importlib
import sys
import os
from pathlib import Path
from tqdm import tqdm
import warnings
from concurrent.futures import ThreadPoolExecutor

# Add parent directory to path to allow imports
sys.path.append(str(Path(__file__).parent))

from alpha191.utils import (
    load_benchmark_csv, 
    get_benchmark_members, 
    load_stock_csv
)
from assessment import get_clean_factor_and_forward_returns, compute_performance_metrics

# Configuration
BENCHMARK = 'zz800'
HORIZONS = [1, 5, 10, 20]
OUTPUT_FILE = 'alpha_performance.csv'

def get_alpha_function(alpha_name):
    """Dynamically import alpha function."""
    try:
        module = importlib.import_module(f"alpha191.{alpha_name}")
        # Try different naming conventions
        func_name1 = alpha_name[:5] + "_" + alpha_name[5:] # alpha_001
        func_name2 = alpha_name # alpha001
        
        if hasattr(module, func_name1):
            return getattr(module, func_name1)
        elif hasattr(module, func_name2):
            return getattr(module, func_name2)
    except (ImportError, ModuleNotFoundError):
        pass
    return None

def preload_data(benchmark):
    """Load all stock data into memory using float32."""
    print("Loading benchmark data...")
    benchmark_df = load_benchmark_csv(benchmark)
    codes = get_benchmark_members(benchmark)
    
    print(f"Pre-loading {len(codes)} stocks into memory...")
    
    stock_cache = {}
    
    def load_one(code):
        try:
            df = load_stock_csv(code, benchmark)
            # Add benchmark columns which might be used by some alphas
            df['benchmark_close'] = benchmark_df['close'].reindex(df.index)
            df['benchmark_open'] = benchmark_df['open'].reindex(df.index)
            
            # Ensure float32 for memory efficiency
            float_cols = df.select_dtypes(include=['float64']).columns
            df[float_cols] = df[float_cols].astype(np.float32)
            
            # Ensure volume is float32
            if 'volume' in df.columns:
                 df['volume'] = df['volume'].astype(np.float32)
                 
            return code, df
        except Exception:
            return code, None

    # Use ThreadPool for I/O bound loading
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(tqdm(executor.map(load_one, codes), total=len(codes)))
        
    for code, df in results:
        if df is not None:
            stock_cache[code] = df
            
    print(f"Loaded {len(stock_cache)} stocks successfully.")
    return stock_cache, benchmark_df.index

def compute_alpha_for_all_stocks(alpha_func, stock_cache):
    """Compute alpha values using cached data."""
    results = {}
    prices = {}
    
    for code, df in stock_cache.items():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                val = alpha_func(df)
            
            if val is not None:
                 # Downcast to float32
                results[code] = val.replace([np.inf, -np.inf], np.nan).astype(np.float32)
                prices[code] = df['close']
        except Exception:
            pass
            
    return results, prices

def append_results(filepath, results):
    """Append new results to CSV."""
    df = pd.DataFrame(results)
    if not os.path.exists(filepath):
        df.to_csv(filepath, index=False)
    else:
        df.to_csv(filepath, mode='a', header=False, index=False)

def extract_detailed_metrics(alpha_name, metrics):
    """Flatten complex metrics dictionary into a single row."""
    row = {'Alpha': alpha_name}
    
    # 1. IC Summary Metrics (for each horizon)
    ic_summary = metrics.get('ic_summary')
    if ic_summary is not None:
        for horizon in ic_summary.columns: # 1D, 5D, etc.
            h_str = str(horizon)
            row[f'IC_Mean_{h_str}'] = ic_summary.loc['IC Mean', horizon]
            row[f'IC_Std_{h_str}'] = ic_summary.loc['IC Std.', horizon]
            row[f'IC_IR_{h_str}'] = ic_summary.loc['Risk-Adjusted IC (IR)', horizon]
            row[f'IC_Winrate_{h_str}'] = ic_summary.loc['IC Winrate', horizon]
            if 'IC Skew' in ic_summary.index:
                row[f'IC_Skew_{h_str}'] = ic_summary.loc['IC Skew', horizon]
            if 'IC Max Drawdown' in ic_summary.index:
                row[f'IC_MaxDD_{h_str}'] = ic_summary.loc['IC Max Drawdown', horizon]
            if 't-stat(IC)' in ic_summary.index:
                 row[f'IC_tStat_{h_str}'] = ic_summary.loc['t-stat(IC)', horizon]

    # 2. Long-Short Portfolio Metrics (for each horizon)
    q_stats = metrics.get('q_stats')
    if q_stats:
        for horizon, stats_df in q_stats.items():
            if 'Long-Short' in stats_df.index:
                ls = stats_df.loc['Long-Short']
                row[f'LS_Sharpe_{horizon}'] = ls.get('sharpe', np.nan)
                row[f'LS_AnnRet_{horizon}'] = ls.get('ann_ret', np.nan)
                row[f'LS_MaxDD_{horizon}'] = ls.get('max_dd', np.nan)
                row[f'LS_Calmar_{horizon}'] = ls.get('calmar', np.nan)
                row[f'LS_MeanRet_{horizon}'] = ls.get('Mean', np.nan)

    # 3. Monotonicity (for each horizon)
    mono = metrics.get('mono_score')
    if mono is not None:
        for horizon, score in mono.items():
            row[f'Monotonicity_{horizon}'] = score

    # 4. Global Metrics
    row['Turnover'] = metrics.get('quantile_turnover', pd.Series()).mean()
    row['RRE'] = metrics.get('rre', np.nan)
    
    # 5. Quantile Spread (Q10 - Q1)
    mean_ret = metrics.get('mean_ret')
    if mean_ret is not None:
         # Assuming quantile indices are 1-10
        max_q = mean_ret.index.max()
        min_q = mean_ret.index.min()
        for horizon in mean_ret.columns:
            spread = mean_ret.loc[max_q, horizon] - mean_ret.loc[min_q, horizon]
            row[f'Q_Spread_{horizon}'] = spread

    return row

def main():
    print(f"Starting Alpha Performance Assessment on {BENCHMARK}...")
    
    # Load All Data
    stock_cache, timeline = preload_data(BENCHMARK)
    
    existing_alphas = set()
    if os.path.exists(OUTPUT_FILE):
        try:
            existing_df = pd.read_csv(OUTPUT_FILE)
            if not existing_df.empty:
                existing_alphas = set(existing_df['Alpha'].astype(str).unique())
                print(f"Resuming: Found {len(existing_alphas)} existing records.")
        except Exception: 
            pass

    alpha_range = [f"alpha{i:03d}" for i in range(1, 192)]
    alphas_to_process = [a for a in alpha_range if a not in existing_alphas]
    
    print(f"Processing {len(alphas_to_process)} alphas...")
    
    batch_results = []
    
    for i, alpha_name in enumerate(tqdm(alphas_to_process)):
        alpha_func = get_alpha_function(alpha_name)
        if not alpha_func:
            continue
            
        try:
            factor_results, price_results = compute_alpha_for_all_stocks(alpha_func, stock_cache)
            
            if not factor_results:
                continue

            factor_matrix = pd.DataFrame(factor_results).reindex(timeline)
            price_matrix = pd.DataFrame(price_results).reindex(timeline)
            
            # Compute Metrics
            # using optimized assessment module
            factor_data = get_clean_factor_and_forward_returns(
                factor_matrix,
                price_matrix,
                periods=HORIZONS,
                max_loss=0.40
            )
            
            metrics = compute_performance_metrics(factor_data)
            
            # Extract flattened metrics
            res_row = extract_detailed_metrics(alpha_name, metrics)
            batch_results.append(res_row)
            
            # Save every 5 alphas or at end
            if len(batch_results) >= 5 or i == len(alphas_to_process) - 1:
                append_results(OUTPUT_FILE, batch_results)
                batch_results = []
                
        except Exception as e:
            # print(f"Error {alpha_name}: {e}")
            pass

    print(f"\nAssessment completed. Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

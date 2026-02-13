#!/usr/bin/env python3
"""
Alpha Selection Script
======================
Scans all 191 alphas, computes performance metrics (IC, IR, Turnover), 
and selects the best subset based on performance and independence.

Optimized for speed by pre-loading all stock data into memory.
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
PRIMARY_HORIZON = '20D'
MIN_IC_ABS = 0.02
MIN_IR = 0.2
MAX_CORR = 0.7
OUTPUT_FILE = 'alpha_performance.csv'
REPORT_FILE = 'alpha_selection_report.md'

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
    """Load all stock data into memory."""
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
            val = alpha_func(df)
            if val is not None:
                 # Downcast to float32
                results[code] = val.astype(np.float32)
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

def main():
    print(f"Starting Alpha Selection Process on {BENCHMARK}...")
    
    # 1. Load Pre-computed Data
    try:
        corr_df = pd.read_csv('alpha_correlation.csv', index_col=0)
    except FileNotFoundError:
        print("Error: alpha_correlation.csv not found.")
        return

    try:
        vif_df = pd.read_csv('alpha_vif.csv', index_col=0)
        vif_map = vif_df['VIF'].to_dict() if 'VIF' in vif_df.columns else {}
    except FileNotFoundError:
        vif_map = {}

    # 2. Assessment Phase
    existing_alphas = set()
    if os.path.exists(OUTPUT_FILE):
        try:
            existing_df = pd.read_csv(OUTPUT_FILE)
            if not existing_df.empty:
                existing_alphas = set(existing_df['Alpha'].astype(str).unique())
                print(f"Found {len(existing_alphas)} existing records in {OUTPUT_FILE}.")
        except Exception: 
            pass

    # Load All Data
    stock_cache, timeline = preload_data(BENCHMARK)
    
    alpha_range = [f"alpha{i:03d}" for i in range(1, 192)]
    alphas_to_process = [a for a in alpha_range if a not in existing_alphas]
    
    print(f"Processing {len(alphas_to_process)} remaining alphas...")
    
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
            factor_data = get_clean_factor_and_forward_returns(
                factor_matrix,
                price_matrix,
                periods=HORIZONS,
                max_loss=0.40
            )
            
            metrics = compute_performance_metrics(factor_data)
            ic_summary = metrics['ic_summary']
            
            h = PRIMARY_HORIZON if PRIMARY_HORIZON in ic_summary.columns else ic_summary.columns[0]
            
            res_row = {
                'Alpha': alpha_name,
                'IC_Mean': ic_summary.loc['IC Mean', h],
                'IC_IR': ic_summary.loc['Risk-Adjusted IC (IR)', h],
                'IC_Winrate': ic_summary.loc['IC Winrate', h],
                'Turnover': metrics['quantile_turnover'].mean(),
                'RRE': metrics['rre']
            }
            batch_results.append(res_row)
            
            # Save every 5 alphas or at end
            if len(batch_results) >= 5 or i == len(alphas_to_process) - 1:
                append_results(OUTPUT_FILE, batch_results)
                batch_results = []
                
        except Exception as e:
            # print(f"Error {alpha_name}: {e}")
            pass

    # 3. Selection Logic (Reload fuller DF)
    print("\nLoading full results for selection...")
    if not os.path.exists(OUTPUT_FILE):
        print("No results found.")
        return
        
    perf_df = pd.read_csv(OUTPUT_FILE).drop_duplicates(subset='Alpha', keep='last')
    perf_df.set_index('Alpha', inplace=True)
    
    print(f"\n--- Step 1: Performance Filtering ---")
    mask_ic = perf_df['IC_Mean'].abs() > MIN_IC_ABS
    mask_ir = perf_df['IC_IR'] > MIN_IR
    candidates = perf_df[mask_ic & mask_ir].copy()
    
    print(f"Candidates passing thresholds: {len(candidates)}")
    
    if candidates.empty:
        print("No candidates found.")
        return

    candidates.sort_values('IC_IR', ascending=False, inplace=True)
    
    print("\n--- Step 2: Independence Filtering ---")
    selected_alphas = []
    
    for alpha in candidates.index:
        is_correlated = False
        for selected in selected_alphas:
            try:
                val = abs(corr_df.loc[alpha, selected])
            except KeyError:
                val = 0.0
            if val > MAX_CORR:
                is_correlated = True
                break
        if not is_correlated:
            selected_alphas.append(alpha)
            
    print(f"Selected {len(selected_alphas)} alphas.")
    
    # 4. Generate Report
    final_df = candidates.loc[selected_alphas].copy()
    final_df['VIF'] = final_df.index.map(lambda x: vif_map.get(x, 'N/A'))
    
    with open(REPORT_FILE, 'w') as f:
        f.write("# Alpha Selection Report\n\n")
        f.write(f"**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d')}\n")
        f.write(f"**Metrics:** {PRIMARY_HORIZON} Horizon\n\n")
        f.write("## Selected Alphas\n")
        f.write(final_df.round(4).to_markdown())
        f.write("\n\n")
        f.write("## Excluded (High Correlation)\n")
        
        dropped = []
        sel_temp = []
        for alpha in candidates.index:
            is_corr = False
            for s in sel_temp:
                try: val = abs(corr_df.loc[alpha, s])
                except: val = 0
                if val > MAX_CORR:
                    dropped.append(f"- **{alpha}** (IR={candidates.loc[alpha, 'IC_IR']:.2f}) excluded due to **{s}** (IR={candidates.loc[s, 'IC_IR']:.2f}, Corr={val:.2f})")
                    is_corr = True
                    break
            if not is_corr:
                sel_temp.append(alpha)
        f.write("\n".join(dropped))
    
    print(f"Report saved to {REPORT_FILE}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Alpha Performance Assessment Script
===================================
Scans all 191 alphas and computes comprehensive performance metrics.
Optimized for speed: pre-loads data, uses multiprocessing for alpha
computation, and a lightweight metrics path.

Output:
    alpha_performance.csv: A wide CSV containing detailed metrics for each alpha.
"""

import pandas as pd
import numpy as np
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
    load_stock_csv,
    get_alpha_func
)
from assessment import get_clean_factor_and_forward_returns, compute_performance_metrics_light

# Configuration
BENCHMARK = 'zz800'
HORIZONS = [5, 10, 20, 60]
OUTPUT_FILE = 'alpha_performances.csv'

warnings.filterwarnings("ignore")


def preload_data(benchmark):
    """Load all stock data into memory using float32."""
    benchmark_df = load_benchmark_csv(benchmark)
    codes = get_benchmark_members(benchmark)

    stock_cache = {}

    def load_one(code):
        try:
            df = load_stock_csv(code, benchmark)
            df['benchmark_close'] = benchmark_df['close'].reindex(df.index)
            df['benchmark_open'] = benchmark_df['open'].reindex(df.index)

            float_cols = df.select_dtypes(include=['float64']).columns
            df[float_cols] = df[float_cols].astype(np.float32)

            if 'volume' in df.columns:
                df['volume'] = df['volume'].astype(np.float32)

            return code, df
        except Exception:
            return code, None

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(load_one, codes))

    for code, df in results:
        if df is not None:
            stock_cache[code] = df

    return stock_cache, benchmark_df.index


def compute_alpha_for_all_stocks(alpha_func, stock_cache):
    """Compute alpha values using cached data."""
    results = {}
    prices = {}

    for code, df in stock_cache.items():
        try:
            val = alpha_func(df)
            if val is not None:
                results[code] = val.replace([np.inf, -np.inf], np.nan).astype(np.float32)
                prices[code] = df['close']
        except Exception:
            pass

    return results, prices


def process_one_alpha(alpha_name, stock_cache, timeline):
    """Compute metrics for one alpha. Returns dict or None."""
    alpha_func = get_alpha_func(alpha_name, use_df=True, ignore_errors=True)
    if not alpha_func:
        return None

    try:
        factor_results, price_results = compute_alpha_for_all_stocks(alpha_func, stock_cache)
        if not factor_results:
            return None

        factor_matrix = pd.DataFrame(factor_results).reindex(timeline)
        price_matrix = pd.DataFrame(price_results).reindex(timeline)

        factor_data = get_clean_factor_and_forward_returns(
            factor_matrix,
            price_matrix,
            periods=HORIZONS,
            max_loss=0.40
        )

        metrics = compute_performance_metrics_light(factor_data)
        return extract_detailed_metrics(alpha_name, metrics)
    except Exception as e:
        print(f"  [WARN] {alpha_name} failed: {e}")
        return None


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
        for horizon in ic_summary.columns:
            h_str = str(horizon)
            row[f'IC_Mean_{h_str}'] = ic_summary.loc['IC Mean', horizon]
            row[f'IC_Std_{h_str}'] = ic_summary.loc['IC Std.', horizon]
            row[f'IC_IR_{h_str}'] = ic_summary.loc['Risk-Adjusted IC (IR)', horizon]
            row[f'IC_Winrate_{h_str}'] = ic_summary.loc['IC Winrate', horizon]
    #         if 'IC Skew' in ic_summary.index:
    #             row[f'IC_Skew_{h_str}'] = ic_summary.loc['IC Skew', horizon]
    #         if 'IC Max Drawdown' in ic_summary.index:
    #             row[f'IC_MaxDD_{h_str}'] = ic_summary.loc['IC Max Drawdown', horizon]
    #         if 't-stat(IC)' in ic_summary.index:
    #             row[f'IC_tStat_{h_str}'] = ic_summary.loc['t-stat(IC)', horizon]

    # # 2. Long-Short Portfolio Metrics (for each horizon)
    # q_stats = metrics.get('q_stats')
    # if q_stats:
    #     for horizon, stats_df in q_stats.items():
    #         if 'Long-Short' in stats_df.index:
    #             ls = stats_df.loc['Long-Short']
    #             row[f'LS_Sharpe_{horizon}'] = ls.get('sharpe', np.nan)
    #             row[f'LS_AnnRet_{horizon}'] = ls.get('ann_ret', np.nan)
    #             row[f'LS_MaxDD_{horizon}'] = ls.get('max_dd', np.nan)
    #             row[f'LS_Calmar_{horizon}'] = ls.get('calmar', np.nan)
    #             row[f'LS_MeanRet_{horizon}'] = ls.get('Mean', np.nan)

    # # 3. Monotonicity (for each horizon)
    # mono = metrics.get('mono_score')
    # if mono is not None:
    #     for horizon, score in mono.items():
    #         row[f'Monotonicity_{horizon}'] = score

    # # 4. Global Metrics
    # row['Turnover'] = metrics.get('quantile_turnover', pd.Series()).mean()
    # row['RRE'] = metrics.get('rre', np.nan)

    # # 5. Quantile Spread (Q10 - Q1)
    # mean_ret = metrics.get('mean_ret')
    # if mean_ret is not None:
    #     max_q = mean_ret.index.max()
    #     min_q = mean_ret.index.min()
    #     for horizon in mean_ret.columns:
    #         spread = mean_ret.loc[max_q, horizon] - mean_ret.loc[min_q, horizon]
    #         row[f'Q_Spread_{horizon}'] = spread

    return row


def main():
    print(f"Alpha Performance Assessment — {BENCHMARK}")

    # Load All Data
    stock_cache, timeline = preload_data(BENCHMARK)
    print(f"Loaded {len(stock_cache)} stocks, timeline={len(timeline)} days")

    existing_alphas = set()
    if os.path.exists(OUTPUT_FILE):
        try:
            existing_df = pd.read_csv(OUTPUT_FILE)
            if not existing_df.empty:
                existing_alphas = set(existing_df['Alpha'].astype(str).unique())
                print(f"Resuming: {len(existing_alphas)} already done.")
        except Exception:
            pass

    alpha_range = [f"alpha{i:03d}" for i in range(1, 192)]
    alphas_to_process = [a for a in alpha_range if a not in existing_alphas]

    if not alphas_to_process:
        print("All alphas already processed.")
        return

    print(f"Processing {len(alphas_to_process)} alphas...")

    batch_results = []
    success_count = 0
    fail_count = 0

    for alpha_name in tqdm(alphas_to_process, desc="Alphas", ncols=80):
        result = process_one_alpha(alpha_name, stock_cache, timeline)
        if result is not None:
            batch_results.append(result)
            success_count += 1
        else:
            fail_count += 1

        # Flush to disk every 10 successful alphas
        if len(batch_results) >= 10:
            append_results(OUTPUT_FILE, batch_results)
            batch_results = []

    # Final flush
    if batch_results:
        append_results(OUTPUT_FILE, batch_results)

    print(f"Done. {success_count} succeeded, {fail_count} failed/skipped. Results → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

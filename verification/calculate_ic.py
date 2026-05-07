import pandas as pd
import numpy as np
from pathlib import Path
import os
import sys

# Ensure alpha191 is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from alpha191 import *
from alpha191.utils import get_benchmark_members, load_stock_csv, load_benchmark_csv
import time

def get_alpha_func(alpha_num: int):
    func_name = f"alpha_{alpha_num:03d}"
    import alpha191
    if hasattr(alpha191, func_name):
        return getattr(alpha191, func_name)
    return None

def main():
    benchmark = "hs300"
    print(f"Loading stock codes for {benchmark}...")
    stock_codes = get_benchmark_members(benchmark)
    benchmark_df = load_benchmark_csv(benchmark)

    # Dictionary to store IC for each alpha
    alpha_ic_scores = {}
    missing_alphas = {143}

    # Pre-allocate for faster concatenation later or compute rolling IC
    print("Computing ICs for 184 alphas against 20-day returns...")

    # We will accumulate the target returns and factor values
    # To compute pooled IC
    all_targets = []
    all_factor_vals = {f"alpha_{i:03d}": [] for i in range(1, 192) if i not in missing_alphas}

    processed_stocks = 0
    start_time = time.time()
    for code in stock_codes:
        try:
            df = load_stock_csv(code, benchmark=benchmark)
            # We want to match paper's target: 20-day return, selling/buying at closing price
            # (Ref(close, -20) / close - 1)
            # In pandas: shift(-20) / close - 1
            if len(df) < 250:
                continue

            # Cap at 250 rows for memory/speed
            df = df.iloc[-400:]

            target = (df['close'].shift(-20) / df['close']) - 1

            df['benchmark_close'] = benchmark_df['close'].reindex(df.index)
            df['benchmark_open'] = benchmark_df['open'].reindex(df.index)
            df['benchmark_index_close'] = df['benchmark_close']
            df['benchmark_index_open'] = df['benchmark_open']

            # Since target contains NaNs at the end, we drop those indices
            valid_idx = target.dropna().index
            target_vals = target.loc[valid_idx].values
            all_targets.append(target_vals)

            for i in range(1, 192):
                if i in missing_alphas:
                    continue
                alpha_name = f"alpha_{i:03d}"
                alpha_func = get_alpha_func(i)
                if alpha_func:
                    try:
                        res = alpha_func(df)
                        if isinstance(res, pd.Series):
                            res = res.replace([np.inf, -np.inf], np.nan)
                            # Align with valid target indices
                            res_vals = res.loc[valid_idx].values
                            all_factor_vals[alpha_name].append(res_vals)
                        else:
                            all_factor_vals[alpha_name].append(np.full(len(valid_idx), np.nan))
                    except Exception as e:
                        all_factor_vals[alpha_name].append(np.full(len(valid_idx), np.nan))

            processed_stocks += 1
            if processed_stocks % 10 == 0:
                print(f"Processed {processed_stocks} stocks...")
        except Exception as e:
            print(f"Error on {code}: {e}")

    print(f"Calculation completed in {time.time() - start_time:.2f} seconds.")

    # Flatten the target
    flattened_targets = np.concatenate(all_targets)

    print("Calculating overall pooled IC for each factor...")
    ic_results = {}
    for alpha_name, vals_list in all_factor_vals.items():
        if not vals_list:
            continue
        flattened_vals = np.concatenate(vals_list)

        # Calculate Pearson correlation, handling NaNs
        mask = ~np.isnan(flattened_vals) & ~np.isnan(flattened_targets)
        if np.sum(mask) > 10:
            corr = np.corrcoef(flattened_vals[mask], flattened_targets[mask])[0, 1]
            ic_results[alpha_name] = corr
        else:
            ic_results[alpha_name] = np.nan

    ic_df = pd.Series(ic_results, name="IC").to_frame()
    ic_df.to_csv("alpha_ic.csv")
    print("IC Results saved to alpha_ic.csv")
    print(ic_df.dropna().sort_values(by="IC", key=abs, ascending=False).head(20))

if __name__ == "__main__":
    main()

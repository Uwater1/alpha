
import os
import sys
import pandas as pd
import numpy as np
import time
from pathlib import Path
from alpha191 import *
from alpha191.utils import get_benchmark_members, load_stock_csv, load_benchmark_csv

def get_alpha_func(alpha_num: int):
    """Get the alpha function (the one that takes a DataFrame) by number."""
    func_name = f"alpha_{alpha_num:03d}"
    # Check in individual modules first or from alpha191 package
    import alpha191
    if hasattr(alpha191, func_name):
        return getattr(alpha191, func_name)
    return None

def main():
    benchmark = "hs300"
    print(f"Loading stock codes for {benchmark}...")
    try:
        stock_codes = get_benchmark_members(benchmark)
    except Exception as e:
        print(f"Error loading benchmark members: {e}")
        # Fallback to listing directory if needed
        benchmark_dir = Path('bao') / benchmark
        stock_codes = [f.stem for f in benchmark_dir.glob('*.csv')]

    # Limit to a subset of stocks for speed if necessary, 
    # but let's try 50 stocks first to get a good representative sample
    sample_size = 50
    stock_codes = stock_codes[:sample_size]
    print(f"Using {len(stock_codes)} stocks for covariance calculation.")

    benchmark_df = load_benchmark_csv(benchmark)

    # Dictionary to store concatenated alpha signals
    # Key: alpha_name, Value: list of series
    alpha_data = {f"alpha_{i:03d}": [] for i in range(1, 192)}
    
    # Existing alphas (based on fulltest.py notes)
    missing_alphas = {30, 143, 165, 183}

    start_time = time.time()
    
    processed_stocks = 0
    for code in stock_codes:
        try:
            df = load_stock_csv(code, benchmark=benchmark)
            # Add benchmark data as required by some alphas
            df['benchmark_close'] = benchmark_df['close'].reindex(df.index)
            df['benchmark_open'] = benchmark_df['open'].reindex(df.index)
            df['benchmark_index_close'] = df['benchmark_close']
            df['benchmark_index_open'] = df['benchmark_open']
            
            # Use a fixed time window for consistency across stocks
            # For example, last 250 days
            if len(df) > 250:
                df = df.iloc[-250:]
            
            for i in range(1, 192):
                if i in missing_alphas:
                    continue
                
                alpha_func = get_alpha_func(i)
                if alpha_func:
                    try:
                        res = alpha_func(df)
                        # Ensure we have a Series and it has the same length as df
                        if isinstance(res, pd.Series):
                            # Replace Infs with NaN
                            res = res.replace([np.inf, -np.inf], np.nan)
                            alpha_data[f"alpha_{i:03d}"].append(res.values)
                    except Exception as e:
                        # Some alphas might fail on some stocks
                        pass
            
            processed_stocks += 1
            if processed_stocks % 10 == 0:
                print(f"Processed {processed_stocks}/{len(stock_codes)} stocks...")
                
        except Exception as e:
            print(f"Error processing stock {code}: {e}")

    print(f"Calculation completed in {time.time() - start_time:.2f} seconds.")

    # Combine data for each alpha
    final_signals = {}
    for alpha_name, values_list in alpha_data.items():
        if not values_list:
            continue
        # Concatenate all stock signals for this alpha into one long vector
        combined = np.concatenate(values_list)
        # Only keep if not all NaN
        if not np.all(np.isnan(combined)):
            final_signals[alpha_name] = combined

    # Create DataFrame of signals
    df_signals = pd.DataFrame(final_signals)
    
    print(f"Calculating covariance matrix for {len(df_signals.columns)} alphas...")
    # Calculate correlation matrix too, as it's easier to interpret
    cov_matrix = df_signals.cov()
    corr_matrix = df_signals.corr()

    # Save to files
    cov_matrix.to_csv("alpha_covariance.csv")
    corr_matrix.to_csv("alpha_correlation.csv")
    print("Results saved to alpha_covariance.csv and alpha_correlation.csv")

    # Present human readable results (Top 20 highly correlated pairs)
    print("\nTop 20 most correlated alpha pairs (absolute value):")
    print("-" * 60)
    
    # Unstack correlation matrix and get top pairs
    corr_pairs = corr_matrix.unstack()
    # Remove self-correlation
    corr_pairs = corr_pairs[corr_pairs.index.get_level_values(0) != corr_pairs.index.get_level_values(1)]
    # Take absolute value for ranking
    top_corr = corr_pairs.abs().sort_values(ascending=False).head(40) # 40 because each pair is twice
    
    seen = set()
    count = 0
    for (a1, a2), val in top_corr.items():
        pair = tuple(sorted((a1, a2)))
        if pair not in seen:
            seen.add(pair)
            actual_corr = corr_pairs[(a1, a2)]
            print(f"{a1} <-> {a2}: {actual_corr:.4f}")
            count += 1
            if count >= 20:
                break

if __name__ == "__main__":
    main()

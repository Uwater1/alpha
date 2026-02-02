import pandas as pd
import numpy as np
import scipy.stats as stats
import sys
import importlib
from pathlib import Path
from typing import List, Dict, Any
from alpha191.utils import load_stock_csv, load_benchmark_csv

def get_benchmark_members(benchmark: str) -> List[str]:
    """Get stock codes for the specified benchmark."""
    if benchmark == "hs300":
        df = pd.read_csv("bao/hs300_l.csv")
    elif benchmark == "zz500":
        df = pd.read_csv("bao/zz500-l.csv")
    elif benchmark == "zz800":
        df1 = pd.read_csv("bao/hs300_l.csv")
        df2 = pd.read_csv("bao/zz500-l.csv")
        df = pd.concat([df1, df2])
    else:
        raise ValueError(f"Invalid benchmark: {benchmark}")
    
    # Standardize codes from sh.600000 to sh_600000
    codes = df['code'].str.replace('.', '_', regex=False).tolist()
    return codes

def assess_alpha(alpha_name: str, benchmark: str = "zz800", horizon: int = 20):
    """Assess an alpha using Spearman Rank IC."""
    print(f"Assessing {alpha_name} on {benchmark} with horizon {horizon} days...")
    
    # Import alpha function
    try:
        # Import module, e.g., alpha191.alpha001
        alpha_module = importlib.import_module(f"alpha191.{alpha_name}")
        # The function inside alpha001.py is alpha_001 (with underscore)
        # Convert "alpha001" to "alpha_001"
        func_name = alpha_name[:5] + "_" + alpha_name[5:]
        alpha_func = getattr(alpha_module, func_name)
    except (ImportError, AttributeError) as e:
        print(f"Error importing {alpha_name}: {e}")
        # Try fallback to exact name
        try:
            alpha_func = getattr(alpha_module, alpha_name)
        except AttributeError:
            return
    
    codes = get_benchmark_members(benchmark)
    
    factor_results = {}
    return_results = {}
    
    # Load benchmark data to get the full timeline
    benchmark_df = load_benchmark_csv(benchmark)
    timeline = benchmark_df.index
    
    total = len(codes)
    failed_codes = []
    for i, code in enumerate(codes):
        if i % 100 == 0:
            print(f"Processing {i}/{total} stocks...")
        
        try:
            # Load stock data
            df = load_stock_csv(code, benchmark=benchmark)
            
            # Add benchmark data for alignment (needed by some alphas)
            df['benchmark_close'] = benchmark_df['close'].reindex(df.index)
            df['benchmark_open'] = benchmark_df['open'].reindex(df.index)
            
            # Calculate alpha
            alpha_series = alpha_func(df)
            
            # Calculate forward returns
            # return(t, t+horizon) = (close(t+horizon) / close(t)) - 1
            # Shift close prices backwards to get close(t+horizon) at position t
            forward_returns = (df['close'].shift(-horizon) / df['close']) - 1
            
            factor_results[code] = alpha_series.astype(np.float32)
            return_results[code] = forward_returns.astype(np.float32)
            
        except Exception as e:
            # Skip stocks with no data or other issues
            failed_codes.append((code, str(e)))
            continue

    if failed_codes:
        print(f"Failed to load {len(failed_codes)} stocks.")
        print(f"First 5 errors: {failed_codes[:5]}")

    print("Aggregating results...")
    factor_matrix = pd.DataFrame(factor_results).reindex(timeline)
    return_matrix = pd.DataFrame(return_results).reindex(timeline)
    
    # Debug info
    print(f"Factor Matrix Shape: {factor_matrix.shape}")
    print(f"Return Matrix Shape: {return_matrix.shape}")
    print(f"Factor Matrix Non-NaNs: {factor_matrix.notna().sum().sum()}")
    print(f"Return Matrix Non-NaNs: {return_matrix.notna().sum().sum()}")
    
    # Check for any overlapping non-NaN pairs
    total_valid_pairs = (factor_matrix.notna() & return_matrix.notna()).sum().sum()
    print(f"Total overlapping valid pairs (Factor & Return): {total_valid_pairs}")
    
    if total_valid_pairs == 0:
        print("WARNING: No overlapping valid pairs found between Factor and Return matrices.")
        # Print first few factor and return non-NaN counts per stock to see where the data is
        print("\nFirst 10 stocks non-NaN counts:")
        for code in list(factor_results.keys())[:10]:
            f_nn = factor_matrix[code].notna().sum()
            r_nn = return_matrix[code].notna().sum()
            overlap = (factor_matrix[code].notna() & return_matrix[code].notna()).sum()
            print(f"  {code}: Factor={f_nn}, Return={r_nn}, Overlap={overlap}")

    ic_series = pd.Series(index=timeline, dtype=float)
    
    for date in timeline:
        f_vals = factor_matrix.loc[date]
        r_vals = return_matrix.loc[date]
        
        # Keep stocks that have both factor and return data
        mask = f_vals.notna() & r_vals.notna()
        if mask.sum() > 30: # Minimum sample size for meaningful IC
            ic, _ = stats.spearmanr(f_vals[mask], r_vals[mask])
            ic_series.loc[date] = ic
            
    # Compute stats
    ic_series = ic_series.dropna()
    ic_mean = ic_series.mean()
    ic_std = ic_series.std()
    ic_ir = ic_mean / ic_std if ic_std != 0 else np.nan
    n_obs = len(ic_series)
    t_stat = ic_ir * np.sqrt(n_obs) if not np.isnan(ic_ir) else np.nan
    
    output = {
        "alpha": alpha_name,
        "benchmark": benchmark,
        "horizon": horizon,
        "IC_mean": ic_mean,
        "IC_std": ic_std,
        "ICIR": ic_ir,
        "t_stat": t_stat,
        "n_obs": n_obs,
        "IC_series": ic_series
    }
    
    return output
def format_alpha_name(alpha_name: str) -> str:
    """Convert input like '1' or '42' to 'alpha001' or 'alpha042' format.
    If input already starts with 'alpha', return it as-is."""
    if alpha_name.lower().startswith("alpha"):
        return alpha_name.lower()
    else:
        # Convert number to zero-padded format
        try:
            num = int(alpha_name)
            return f"alpha{num:03d}"
        except ValueError:
            raise ValueError(f"Invalid alpha name: {alpha_name}. Expected format: '1' or 'alpha001'")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python assess_alpha.py <alpha_name> [benchmark] [horizon]")
        print("  alpha_name: Number (1-191) or format 'alpha001'")
        print("  benchmark:  hs300, zz500, or zz800 (default: zz800)")
        print("  horizon:    Forward return horizon in days (default: 20)")
        print("\nExamples:")
        print("  python assess_alpha.py 1")
        print("  python assess_alpha.py 42 zz800")
        print("  python assess_alpha.py 1 zz500 5")
        sys.exit(1)

    alpha = format_alpha_name(sys.argv[1])
    benchmark = sys.argv[2] if len(sys.argv) > 2 else "zz800"
    horizon = int(sys.argv[3]) if len(sys.argv) > 3 else 20

    result = assess_alpha(alpha, benchmark, horizon)
    
    if result:
        print("\nAssessment Results:")
        for k, v in result.items():
            if k == "IC_series":
                print(f'"{k}": pd.Series (length {len(v)})')
                print(v.tail())
            else:
                print(f'"{k}": {v},')

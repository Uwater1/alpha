import pandas as pd
import numpy as np
import scipy.stats as stats
import sys
import importlib
import numba
from pathlib import Path
from typing import List, Dict, Any
from alpha191.utils import load_stock_csv, load_benchmark_csv

@numba.jit(nopython=True)
def fast_pearson(x, y):
    """Numba-accelerated Pearson correlation for a single pair of vectors."""
    mask = ~np.isnan(x) & ~np.isnan(y)
    x_valid = x[mask]
    y_valid = y[mask]
    
    if len(x_valid) < 30:
        return np.nan
        
    mu_x = np.mean(x_valid)
    mu_y = np.mean(y_valid)
    
    std_x = np.std(x_valid)
    std_y = np.std(y_valid)
    
    if std_x == 0 or std_y == 0:
        return 0.0
        
    return np.mean((x_valid - mu_x) * (y_valid - mu_y)) / (std_x * std_y)

@numba.jit(nopython=True)
def compute_ic_series_numba(factor_rank_mat, return_rank_mat):
    """Compute IC series for each row (date) using Numba."""
    n_days = factor_rank_mat.shape[0]
    ic_values = np.empty(n_days)
    
    for i in range(n_days):
        ic_values[i] = fast_pearson(factor_rank_mat[i], return_rank_mat[i])
        
    return ic_values

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
    
    factor_matrix = pd.DataFrame(factor_results).reindex(timeline).astype(np.float32)
    return_matrix = pd.DataFrame(return_results).reindex(timeline).astype(np.float32)
    
    # Compute IC using Numba-accelerated Spearman (Pearson on daily Ranks)
    # Cross-sectional ranking for each date
    f_rank = factor_matrix.rank(axis=1, method='min').values
    r_rank = return_matrix.rank(axis=1, method='min').values
    
    ic_vals = compute_ic_series_numba(f_rank, r_rank)
    ic_series = pd.Series(ic_vals, index=timeline).dropna()
            
    # Compute stats
    ic_mean = ic_series.mean()
    ic_std = ic_series.std()
    ic_ir = ic_mean / ic_std if ic_std != 0 else np.nan
    ic_winrate = (ic_series > 0).mean() # Percentage of days with positive IC
    n_obs = len(ic_series)
    t_stat = ic_ir * np.sqrt(n_obs) if not np.isnan(ic_ir) else np.nan
    
    output = {
        "alpha": alpha_name,
        "benchmark": benchmark,
        "horizon": horizon,
        "IC_mean": ic_mean,
        "IC_std": ic_std,
        "IC_winrate": ic_winrate,
        "ICIR": ic_ir, #  mean / std
        "t_stat": t_stat, # mean / std * sqrt(n_obs)
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
        print("Usage: python assess_alpha.py <alpha_name> [horizon] [benchmark]")
        print("  alpha_name: Number (1-191) or format 'alpha001'")
        print("  horizon:    Forward return horizon in days (default: 20)")
        print("  benchmark:  hs300, zz500, or zz800 (default: zz800)")
        print("\nExamples:")
        print("  python assess_alpha.py 1")
        print("  python assess_alpha.py 42 5")
        print("  python assess_alpha.py 1 20 zz500")
        sys.exit(1)

    alpha = format_alpha_name(sys.argv[1])
    horizon = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    benchmark = sys.argv[3] if len(sys.argv) > 3 else "zz800"

    result = assess_alpha(alpha, benchmark, horizon)
    
    if result:
        print("\nAssessment Results:")
        for k, v in result.items():
            if k == "IC_series":
                print(f'"{k}": pd.Series (length {len(v)})')
                print(v.tail())
            else:
                print(f'"{k}": {v},')

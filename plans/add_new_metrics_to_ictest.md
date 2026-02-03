# Implementation Plan: Add New Metrics to ICtest.py

## Overview
Add 9 new metrics to the ICtest.py output to provide deeper analysis of alpha factor performance.

## New Metrics to Add

### 1. Distribution Metrics
- **IC_median**: Median of IC series
- **IC_skew**: Skewness of IC series distribution

### 2. Stability Metrics
- **rolling_IC_min**: Minimum value of rolling IC (60-day window)
- **rolling_IC_max**: Maximum value of rolling IC (60-day window)
- **IC_max_drawdown**: Maximum drawdown of IC series

### 3. Coverage Metrics
- **avg_cs_size**: Average cross-sectional size (number of valid stocks per date)
- **min_cs_size**: Minimum cross-sectional size

### 4. Directional Asymmetry Metrics
- **IC_mean_pos**: Mean of positive IC values
- **IC_mean_neg**: Mean of negative IC values

## Implementation Details

### Changes Required

#### 1. Update `assess_alpha()` Function Signature
**Location:** Line 60
**Change:** Add `rolling_window` parameter with default value of 60

```python
def assess_alpha(alpha_name: str, benchmark: str = "zz800", horizon: int = 20, rolling_window: int = 60):
```

#### 2. Add New Metric Calculations
**Location:** After line 136 (after existing stats computation)

Add the following calculations:

```python
# Distribution metrics
ic_median = ic_series.median()
ic_skew = stats.skew(ic_series.dropna())

# Stability metrics - rolling IC
rolling_ic = ic_series.rolling(window=rolling_window, min_periods=1)
rolling_ic_min = rolling_ic.min().min()
rolling_ic_max = rolling_ic.max().max()

# Stability metrics - max drawdown
# Calculate cumulative maximum and drawdown
ic_cummax = ic_series.cummax()
ic_drawdown = (ic_series - ic_cummax) / ic_cummax
ic_max_drawdown = ic_drawdown.min()

# Coverage metrics - cross-sectional size
# Count non-NaN values per row (date) in factor_matrix
cs_sizes = factor_matrix.notna().sum(axis=1)
avg_cs_size = cs_sizes.mean()
min_cs_size = cs_sizes.min()

# Directional asymmetry metrics
ic_pos = ic_series[ic_series > 0]
ic_neg = ic_series[ic_series < 0]
ic_mean_pos = ic_pos.mean() if len(ic_pos) > 0 else np.nan
ic_mean_neg = ic_neg.mean() if len(ic_neg) > 0 else np.nan
```

#### 3. Update Output Dictionary
**Location:** Lines 139-150
**Change:** Add new metrics to the output dictionary

```python
output = {
    "alpha": alpha_name,
    "benchmark": benchmark,
    "horizon": horizon,
    "rolling_window": rolling_window,
    # Existing metrics
    "IC_mean": ic_mean,
    "IC_std": ic_std,
    "IC_winrate": ic_winrate,
    "ICIR": ic_ir,
    "t_stat": t_stat,
    "n_obs": n_obs,
    "IC_series": ic_series,
    # Distribution
    "IC_median": ic_median,
    "IC_skew": ic_skew,
    # Stability
    "rolling_IC_min": rolling_ic_min,
    "rolling_IC_max": rolling_ic_max,
    "IC_max_drawdown": ic_max_drawdown,
    # Coverage
    "avg_cs_size": avg_cs_size,
    "min_cs_size": min_cs_size,
    # Directional asymmetry
    "IC_mean_pos": ic_mean_pos,
    "IC_mean_neg": ic_mean_neg,
}
```

#### 4. Update Command-Line Argument Parsing
**Location:** Lines 166-180
**Change:** Add support for 4th argument (rolling_window)

```python
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ICtest.py <alpha_name> [horizon] [benchmark] [rolling_window]")
        print("  alpha_name:     Number (1-191) or format 'alpha001'")
        print("  horizon:        Forward return horizon in days (default: 20)")
        print("  benchmark:      hs300, zz500, or zz800 (default: zz800)")
        print("  rolling_window: Rolling window for IC stability metrics (default: 60)")
        print("\nExamples:")
        print("  python ICtest.py 1")
        print("  python ICtest.py 42 5")
        print("  python ICtest.py 1 20 zz500")
        print("  python ICtest.py 1 20 zz800 60")
        sys.exit(1)

    alpha = format_alpha_name(sys.argv[1])
    horizon = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    benchmark = sys.argv[3] if len(sys.argv) > 3 else "zz800"
    rolling_window = int(sys.argv[4]) if len(sys.argv) > 4 else 60

    result = assess_alpha(alpha, benchmark, horizon, rolling_window)
```

## Calculation Logic Summary

| Metric | Calculation | Data Source |
|--------|-------------|-------------|
| IC_median | `ic_series.median()` | ic_series |
| IC_skew | `scipy.stats.skew(ic_series.dropna())` | ic_series |
| rolling_IC_min | `ic_series.rolling(window).min().min()` | ic_series |
| rolling_IC_max | `ic_series.rolling(window).max().max()` | ic_series |
| IC_max_drawdown | `min((ic_series - cummax) / cummax)` | ic_series |
| avg_cs_size | `factor_matrix.notna().sum(axis=1).mean()` | factor_matrix |
| min_cs_size | `factor_matrix.notna().sum(axis=1).min()` | factor_matrix |
| IC_mean_pos | `ic_series[ic_series > 0].mean()` | ic_series |
| IC_mean_neg | `ic_series[ic_series < 0].mean()` | ic_series |

## Dependencies
All required libraries are already imported:
- `pandas` (pd)
- `numpy` (np)
- `scipy.stats` (stats)

## Testing Considerations
1. Verify calculations with known test data
2. Check edge cases (empty series, all positive/negative IC)
3. Ensure rolling window doesn't exceed series length
4. Validate NaN handling in all calculations

## Backward Compatibility
- New parameter `rolling_window` has default value (60)
- Existing code calling `assess_alpha()` without the new parameter will work unchanged
- Output dictionary includes all existing fields plus new ones

# Alpha191

A Python implementation of WorldQuant's 191 Alpha factors for CSI 800 stocks.

## Quick Start

```python
from alpha191 import alpha_001

# Compute factor value for a single stock on a specific date
result = alpha001(
    code="sh_600016",
    benchmark="zz800",
    end_date="2026-01-23",
    lookback=350
)
# Returns: float
```

## Installation

```bash
pip install numpy pandas scipy numba
```

## Usage

### Method 1: Convenience Function (Recommended)

Import directly from the package:

```python
from alpha191 import alpha_001, alpha_002

# Compute factor value for a stock
val = alpha001(code="sh_600009", benchmark="hs300")
val = alpha002(code="sz_000001", benchmark="zz500")

# With custom date and lookback
val = alpha001(code="sh_600016", benchmark="zz800", 
               end_date="2026-01-01", lookback=350)
```

Or import all factors at once:

```python
from alpha191 import *

# All alpha modules and functions are now available
val = alpha001(code="sh_600009", benchmark="hs300")
val = alpha101(code="sz_000001")
```

**Parameters:**
- `code` (str): Stock code (e.g., `sh_600016`, `sz_000001`)
- `benchmark` (str): Index pool - `hs300`, `zz500`, or `zz800` (default: `zz800`)
- `end_date` (str): Computation date in `YYYY-MM-DD` format (default: `2026-01-23`)
- `lookback` (int): Historical days to load (default: 350)

**Returns:** `float` (factor value, or `np.nan` if not available)

### Method 2: DataFrame API

Use this when you have your own DataFrame:

```python
from alpha191 import alpha001  # Import module
import pandas as pd

# Load stock data yourself
df = pd.read_csv("stock.csv", parse_dates=["date"], index_col="date")

# Compute full factor series using the module's DataFrame function
factor_series = alpha001.alpha_001(df)  # Returns pd.Series with same index
```

Or import the DataFrame function directly (for use with DataFrames):

```python
# Import the raw DataFrame function from the module file
from alpha191.alpha001 import alpha_001
import pandas as pd

df = pd.read_csv("stock.csv", parse_dates=["date"], index_col="date")
factor_series = alpha_001(df)  # Returns pd.Series with same index
```

### Method 3: Using Utils for Data Loading

```python
from alpha191.utils import load_stock_csv
from alpha191 import alpha001  # Import module

# Load data manually
df = load_stock_csv("sh_600016", benchmark="zz800")
df = df.loc[:"2026-01-23"].iloc[-350:]

# Compute factor using module.function
result = alpha001.alpha_001(df).iloc[-1]  # Get last value
```

## Project Structure

```
alpha191/
├── alphaXXX.py       # Factor implementations (191 files)
├── operators.py      # Math operators (RANK, CORR, DELTA, etc.)
├── utils.py          # Data loading utilities
└── __init__.py       # Exports alphaXXX (modules) and alpha_XXX (functions)

bao/
├── hs300/            # HS300 stock CSV files
├── zz500/            # ZZ500 stock CSV files
├── hs300.csv         # HS300 index CSV files
├── zz500.csv         # ZZ500 index CSV files
└── zz800.csv         # ZZ800 index CSV files

tests/
└── test_alphas.py    # Unit tests
```

## Available Factors

All 191 factors are available:

```python
# Import modules (recommended)
from alpha191 import alpha_001, alpha_002, ..., alpha191

# Or import everything at once
from alpha191 import *

# Access convenience functions (return float)
val = alpha001(code="sh_600009", benchmark="hs300")

# Or access DataFrame functions directly (return pd.Series)
from alpha191 import alpha_001, alpha_002, ..., alpha_191
factor_series = alpha_001(df)
```

See [`alpha191.md`](alpha191.md) for formula details.

## Assessing Alphas

Use `ICtest.py` to evaluate alpha factors using Spearman Rank IC (Information Coefficient) analysis. It supports multi-horizon analysis and parallel processing for fast execution.

```bash
# Basic usage - assess alpha 1 with default settings
python ICtest.py 1

# Assess with custom horizons and benchmark
python ICtest.py 1 --horizons "1,5,10,20,30,60" --benchmark zz800

# Run with parallel processing (8 workers) and generate plots
python ICtest.py 1 --jobs 8 --plot
```

**Arguments:**
- `alpha`: Alpha number (1-191) or format like `alpha001`.
- `--horizons`: Comma-separated list of forward return horizons (default: `1,5,10,20,30,60`).
- `--benchmark`: Index pool - `hs300`, `zz500`, or `zz800` (default: `zz800`).
- `--plot`: Generate a comprehensive `alphaXXX_tear_sheet.png` visual report.
- `--jobs`: Number of parallel workers (default: `-1` to use all CPUs).

**Output Metrics:**
- **IC Summary**: Mean IC, IC Std, ICIR (Information Ratio), and T-stat for significance.
- **IC Decay**: Visual representation of IC across different horizons.
- **Robustness**: Performance comparison between the full period and the recent 3 years.
- **Rank Stability (RRE)**: Measures the day-to-day stability of stock rankings (Higher is better).
- **In-Depth Stability Analysis**:
    - **Year-by-Year Breakdown**: Yearly IC performance for consistency check.
    - **IC Trend Analysis**: Detects if the alpha factor is improving or decaying over time.
    - **Regime Analysis**: Performance during different market regimes (High vs Low IC periods).
    - **IC Consistency Score**: An overall rating of how stable the rolling IC stays.

## Group Return Test

Divide stocks into quantiles based on alpha values and calculate group returns over time to test for monotonicity and spread.

```bash
# Basic usage - divide into 10 groups, 20-day horizon
python grouptest.py 1

# Custom parameters: 5 quantiles, multiple horizons, zz800 benchmark
python grouptest.py 1 --quantiles 5 --horizon "5,10,20" --benchmark zz800 --plot
```

**Arguments:**
- `alpha`: Alpha number (1-191) or format like `alpha001`.
- `--horizon`: Forward return horizon(s), comma-separated (default: `20`).
- `--benchmark`: Index pool - `hs300`, `zz500`, or `zz800` (default: `hs300`).
- `--quantiles`: Number of groups/quantiles (default: `10`).
- `--plot`: Generate `alphaXXX_group_returns.png` and `alphaXXX_cumulative_returns.png`.

**Performance Metrics:**
- **Quantile Stats**: Mean Return, Std Error, t-stat, p-value, and Turnover for each group.
- **Long-Short Portfolio**:
    - **Annualized Return & Volatility**
    - **Sharpe & Calmar Ratios**
    - **Max Drawdown**
- **Monotonicity**: Score indicating how well returns follow the quantile order.

## Assessment Module

The `assessment` package provides professional-grade performance metrics and visualizations for alpha factors.

### Features
- **IC Analysis**: Spearman Rank IC, ICIR, t-stats, and p-values.
- **Quantile Returns**: Mean returns, cumulative returns, and monotonicity analysis.
- **Stability Analysis**: Factor stability via rank autocorrelation, quantile turnover, and **Rank Stability (RRE)**.
- **Advanced Stability (New)**: Multi-window rolling IC, trend analysis, and year-over-year consistency metrics.
- **Visualizations**: Comprehensive tear sheets, IC decay plots, and quantile return bar charts.

### Programmatic Access

```python
from assessment import get_clean_factor_and_forward_returns, compute_performance_metrics, create_full_tear_sheet

# factor_matrix and price_matrix are Date x Stock DataFrames (wide format)
factor_data = get_clean_factor_and_forward_returns(
    factor_matrix, 
    price_matrix, 
    periods=[1, 5, 20], 
    quantiles=10
)

# Compute statistics
metrics = compute_performance_metrics(factor_data)
print(metrics['ic_summary'])

# Generate visual report
create_full_tear_sheet(factor_data, output_path="alpha_report.png")
```

## Expression Alpha Parser

The `alpha191.expression` module allows you to define alpha factors using string expressions. This is based on the logic extracted from `alphatools` and adapted for this project.

### Usage

```python
from alpha191 import ExpressionAlpha

# Define an alpha expression
expr = "rank(delta(log(close), 1))"
ea = ExpressionAlpha(expr)

# Option 1: Generate Python code
print(ea.to_python(func_name='my_alpha'))

# Option 2: Get a function object directly
alpha_func = ea.get_func()

# Use with a DataFrame
import pandas as pd
from alpha191.utils import load_stock_csv

df = load_stock_csv("sh_600016")
factor_series = alpha_func(df)
```

### Supported Operators
- **Basic Data**: `close`, `opens`, `high`, `low`, `volume`, `vwap`, `returns`
- **Arithmetic**: `+`, `-`, `*`, `/`, `^`, `neg`, `abs`, `log`, `sign`
- **Rolling Windows**: `ts_rank`, `ts_sum`, `ts_max`, `ts_min`, `stddev`, `correlation`, `covariance`, `delay`, `delta`
- **Cross-sectional**: `rank`, `ind_neutralize(x, groups)`
- **Conditional**: `condition ? then : else`, `>`, `<`, `==`, `||`

> [!NOTE]
> `ind_neutralize` (or `indneutralize`) requires a group identifier array (e.g., industry categories) as the second argument.

## Testing

```bash
pytest tests/
```

```bash
# Test specific factor
python test_factor.py alpha001

# Or using number
python test_factor.py 1
```

```bash
# Speed test
python speedtest.py

# With specific repeat count
python speedtest.py 100
```

```bash
# Full test suite
python fulltest.py
```

```bash
# Assess factor
python ICtest.py 1 zz800
```

## Data Requirements

CSV files should contain columns:
- `date`, `open`, `high`, `low`, `close`, `volume`
- Optional: `amount`, `vwap`

Data is automatically loaded from:
- `bao/hs300/{code}.csv` for HS300 stocks
- `bao/zz500/{code}.csv` for ZZ500 stocks

## Notes:
In this project we treat zz800 (中证800) as the combination of hs300 (沪深300) and zz500 (中证500)

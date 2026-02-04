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

Use `ICtest.py` to evaluate alpha factors using Spearman Rank IC (Information Coefficient) analysis:

```bash
# Basic usage - assess alpha 1 on default benchmark (zz800) with default horizon (20 days)
python ICtest.py 1

# Assess alpha 42 with 5-day horizon
python ICtest.py 42 5

# Assess alpha 1 with 5-day horizon on hs300
python ICtest.py 1 5 hs300
```

**Arguments:**
- `alpha_name` (required): Alpha number (1-191) or format like `alpha001`
- `benchmark`: Index pool - `hs300`, `zz500`, or `zz800` (default: `zz800`)
- `horizon`: Forward return horizon in days (default: `20`)

**Output Metrics:**
- `IC_mean`: Mean Information Coefficient
- `IC_std`: Standard deviation of IC
- `ICIR`: Information Coefficient Information Ratio (IC_mean / IC_std)
- `t_stat`: T-statistic for significance testing
- `n_obs`: Number of observations

## Group Return Test

Divide stocks into $m$ quantiles based on alpha values and calculate group returns over time:

```bash
# Basic usage - divide into 10 groups, 20-day horizon, hs300 benchmark
python grouptest.py 1

# Custom parameters: alpha=1, period=20, range=zz800, quantile=5
python grouptest.py 1 20 zz800 5
```

**Arguments:**
- `alpha_name` (required): Alpha number (1-191) or format like `alpha001`
- `period`: Forward return horizon in days (default: `20`)
- `range`: Index pool - `hs300`, `zz500`, or `zz800` (default: `hs300`)
- `quantile`: Number of groups/quantiles (default: `10`)

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

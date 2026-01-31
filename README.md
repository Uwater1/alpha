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

### Method 1: Convenience Function (Recommended)\n
Import directly from the package:

```python
from alpha191 import alpha_001, alpha_002

# Compute factor value for a stock
val = alpha_001(code="sh_600009", benchmark="hs300")
val = alpha_002(code="sz_000001", benchmark="zz500")

# With custom date and lookback
val = alpha_001(code="sh_600016", benchmark="zz800", 
               end_date="2026-01-01", lookback=350)
```

Or import all factors at once:

```python
from alpha191 import *

# All alpha modules and functions are now available
val = alpha_001(code="sh_600009", benchmark="hs300")
val = alpha_101(code="sz_000001", benchmark="zz500")
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
└── zz500/            # ZZ500 stock CSV files

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

## Testing

```bash
pytest tests/
```

## Data Requirements

CSV files should contain columns:
- `date`, `open`, `high`, `low`, `close`, `volume`
- Optional: `amount`, `vwap`

Data is automatically loaded from:
- `bao/hs300/{code}.csv` for HS300 stocks
- `bao/zz500/{code}.csv` for ZZ500 stocks

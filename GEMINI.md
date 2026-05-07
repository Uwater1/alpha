# Alpha191 Project Overview

This project is a Python implementation of WorldQuant's 191 Alpha factors, specifically optimized for CSI 800 stocks. It includes a complete suite of performance assessment tools, such as IC (Information Coefficient) analysis, group return tests, and stability metrics.

## Architecture

- **`alpha191/`**: Core package containing the 191 alpha factor implementations.
  - `alphaXXX.py`: Individual factor implementations.
  - `operators.py`: Specialized math operators (e.g., `rank`, `ts_sum`, `delay`) used across factors.
  - `utils.py`: Data loading utilities and execution wrappers.
  - `expression/`: Parser for string-based alpha expressions.
- **`assessment/`**: Evaluation module for calculating financial metrics (Sharpe, ICIR, Turnover, etc.) and generating tear sheets.
- **`bao/`**: Local data store containing stock CSV files for HS300 and ZZ500 (combined to form ZZ800).
- **Scripts**:
  - `ICtest.py`: Evaluates a factor's Information Coefficient.
  - `grouptest.py`: Performs quantile return analysis and monotonicity tests.
  - `run_alpha_tests.py`: Batch processing script for multiple alphas.
  - `calculate_covariance.py` / `calculate_vif.py`: Statistical analysis of factor relationships.

## Building and Running

### Environment Setup
The project uses `venv` and `uv` for dependency management.
```bash
uv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Running Tests
```bash
# Run all unit tests
pytest tests/

# Test a specific factor implementation
python test_factor.py 1
```

### Alpha Assessment
```bash
# Run IC test for Alpha 1
python ICtest.py 1 --benchmark zz800 --plot

# Run Group Return test for Alpha 1
python grouptest.py 1 --quantiles 10 --plot

# Run batch tests for alphas listed in 20D_top_cleaned.csv
python run_alpha_tests.py
```

## Development Conventions

### Alpha Implementation Template
All new alphas must follow the template in `instructions.md`:
1.  **Vectorization**: Use `numpy` arrays extracted from `df` for computation.
2.  **Operators**: Prefer functions from `alpha191.operators` to ensure consistency.
3.  **Division Safety**: Always protect against division by zero (set denom to `np.nan`).
4.  **VWAP/RET**: Use the standardized calculation blocks provided in `instructions.md` for these variables.
5.  **Registration**: New alphas must be imported in `alpha191/__init__.py`.

### Code Style
- Follow PEP 8.
- Use type hints for function signatures.
- Document formulas in the docstrings of `alpha_XXX` functions.

### Data Handling
- Stocks are identified by codes like `sh_600016`.
- Benchmarks supported: `hs300`, `zz500`, `zz800`.
- Data is expected in `bao/` with standard OHLCV columns.

## Testing Strategy
- **Unit Tests**: `tests/test_alphas.py` should contain sanity checks for factor output (shape, NaNs, Series type).
- **Speed Tests**: Use `speedtest.py` to ensure implementations are computationally efficient.
- **Validation**: Compare implementation results against `alpha191.md` formulas.

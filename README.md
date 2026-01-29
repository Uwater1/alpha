# Alpha191 CSI 800

## Current Files
- `alpha191_cleaned.txt` : Main essay
- `alpha191.md` : Main Alpha (to be implemented)
- `bao/` : All the data (.csv)

## Alpha191 Implementation
- `alpha191/` : Factor implementations with operators and tests
- `tests/` : Unit tests for all factors
- `scripts/` : Validation scripts for running factors on real data

### Implemented Factors
- **alpha_001**: Correlation between ranked volume changes and ranked price returns (window=6)

### API Usage

```python
from alpha191.alpha001 import alpha001

# Compute alpha_001 for a single stock on a specific date
result = alpha001(code="sh_600016", end_date="2026-01-23", lookback=350)

# Returns: float (or np.nan if result is NaN)
```

**Parameters:**
- `code` (str): Stock code (e.g., "sh_600016", "sz_000001")
- `end_date` (str): End date for computation (default: "2026-01-23")
- `lookback` (int): Number of historical rows to use (default: 350)

**Data Loading:**
- Automatically searches for `{code}.csv` in `bao/hs300/` and `bao/zz500/`
- Raises `FileNotFoundError` if file not found
- Raises `ValueError` if file exists in both directories or insufficient history

### Usage
```bash
# Run tests
pytest tests/

# Validate alpha_001 on HS300 data
python scripts/run_alpha001.py
```

# Alpha191 Factory: Implementation Instructions

Your goal is to implement a specific batch of Alpha factors. Each factor must follow a strict template to ensure compatibility with our library.

## Project Structure
- `alpha191/alphaXXX.py`: Your implementation.
- `alpha191/operators.py`: Library of pre-built math operators.
- `alpha191/utils.py`: Data loading and execution helper.
- `tests/test_alphas.py`: Group tests for factors. (PUT YOUR TEST HERE, DO NOT CREATE NEW FILE)
- `alpha191.md`: list of all alpha, you are to find your alpha from this document, if it said 尚未实现, ingnore this alpha. 

## Implementation Rules

### 1. Template
Every file must look like this:

```python
import numpy as np
import pandas as pd
from .operators import *  # Import only what you need
from .utils import run_alpha_factor

def alpha_XXX(df: pd.DataFrame) -> pd.Series:
    """
    Compute AlphaXXX factor.
    Formula: [Copy formula from alpha191.md]
    """
    # 1. Identify columns needed (open, high, low, close, volume, etc.)
    # 2. Extract values as numpy arrays: val = df['col'].values
    # 3. Translate formula symbols to operator.py functions
    # 4. Handle cross-sectional RANK as ts_rank(X, window=N) if inside windowed op
    # 5. Return pd.Series(result, index=df.index, name='alpha_XXX')
    pass

def alphaXXX(code, benchmark='zz800', end_date="2026-01-23", lookback=350):
    return run_alpha_factor(alpha_XXX, code, benchmark, end_date, lookback)
```

### 2. Formula Translation Guide
| Formula Symbol | Operator Function | Notes |
| :--- | :--- | :--- |
| `DELTA(A, n)` | `delta(a, n)` | |
| `DELAY(A, n)` | `delay(a, n)` | |
| `RANK(A)` | `rank(a)` | Cross-sectional. Use `ts_rank` if in window. |
| `TSRANK(A, n)` | `ts_rank(a, n)` | Time-series rank. |
| `CORR(A, B, n)` | `rolling_corr(a, b, n)` | |
| `SUM(A, n)` | `ts_sum(a, n)` | |
| `MEAN(A, n)` | `ts_mean(a, n)` | |
| `STD(A, n)` | `ts_std(a, n)` | |
| `TSMAX(A, n)` | `ts_max(a, n)` | |
| `TSMIN(A, n)` | `ts_min(a, n)` | |
| `A?B:C` | `np.where(A, B, C)` | Vectorized conditional. |
| `LOG(A)` | `np.log(A)` | Use `np.log`. Handle 0 with `replace(0, np.nan)`. |
| `ABS(A)` | `np.abs(A)` | |
| `SIGN(A)` | `sign(a)` | |

### 3. Step-by-Step Workflow
1. **Read Formula**: Extract formula from `alpha191.md`.
2. **Implement**: Create `alpha191/alphaXXX.py`.
3. **Register**: Add `from .alphaXXX import alpha_XXX` to `alpha191/__init__.py`.
4. **Test**: Add a test case to `tests/test_alphas.py`. Check that the output is a Series and has correct NaNs at start.

### 4. Special Handling: Division by Zero
When the formula is `A / B`, always protect against division by zero in numpy:
```python
denom = B
denom[denom == 0] = np.nan
result = A / denom
```

### 5. Special Variable: RET
If the formula uses `RET`, utilize the `compute_ret` operator or calculate it as `delta(close, 1) / delay(close, 1)`.

### 6. Special Variable: VWAP
If the formula uses `VWAP`  Use the following code to calculate it:
```python
    if 'vwap' in df.columns:
        vwap = df['vwap'].values
    else:
        need_ohlc = True
        if {'amount', 'volume'}.issubset(df.columns):
            vwap_s = df['amount'] / df['volume'].replace(0, np.nan)
            valid = df['amount'].ne(0) & df['volume'].ne(0) & vwap_s.notna() & vwap_s.between(df['low'], df['high'])
            need_ohlc = ~valid.all()
            if not need_ohlc:
                vwap = vwap_s.values

        if need_ohlc:
            ohlc_avg = (df['open'] + df['high'] + df['low'] + df['close']) / 4
            if 'valid' in locals():
                vwap = vwap_s.where(valid, ohlc_avg).values
            else:
                vwap = ohlc_avg.values
```

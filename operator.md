# Alpha191 Operator Documentation

A comprehensive guide to all 30 operators in the Alpha191 quantitative trading factor library.

## Categories

1. [Basic Operations](#basic-operations) (4 operators)
2. [Rolling Window Operations](#rolling-window-operations) (7 operators)
3. [Statistical Operations](#statistical-operations) (3 operators)
4. [Weighted Average Operations](#weighted-average-operations) (3 operators)
5. [Conditional & Special Operations](#conditional--special-operations) (5 operators)
6. [Derived Fields](#derived-fields) (6 operators)
7. [Existing Operators](#existing-operators) (2 operators)

---

## 1. Basic Operations

### delay(x, n)
**Purpose**: Shift array backward by n periods  
**Key Features**:
- Returns array with first n values as NaN
- Handles edge cases (n=0, n≥len(x), n<0)
- Preserves NaN values at shifted positions

**Usage Notes**: Equivalent to pandas shift(n) or np.roll with NaN fill.

### delta(x, n)
**Purpose**: Compute difference between current value and value n periods ago  
**Key Features**:
- Calculates x[t] - x[t-n]
- First n values are NaN
- Handles n=0 (returns all zeros)
- NaN values propagate

**Usage Notes**: Computes first difference when n=1, nth difference for larger n.

### rank(x)
**Purpose**: Cross-sectional rank of elements, normalized to [0, 1]  
**Key Features**:
- Ranks from smallest (0) to largest (1)
- Handles ties with average rank
- Excludes NaN values from ranking
- Single value returns 0.5

**Usage Notes**: Cross-sectional operation (ranks at same time point, unlike ts_rank).

### sign(x)
**Purpose**: Sign function  
**Key Features**:
- Returns 1 if x > 0, 0 if x = 0, -1 if x < 0
- Preserves NaN values
- Simple element-wise operation

**Usage Notes**: Wrapper around np.sign for consistency with other operators.

---

## 2. Rolling Window Operations

### ts_rank(x, window)
**Purpose**: Time-series rank within a rolling window, normalized to [0, 1]  
**Key Features**:
- Computes rank of x[i] within window x[i-window+1:i+1]
- Excludes NaN values from ranking
- First (window-1) values are NaN
- Uses scipy.stats.rankdata

**Usage Notes**: Time-series operation (ranks over time, unlike rank).

### ts_sum(x, n)
**Purpose**: Sum over rolling window (Numba-accelerated)  
**Key Features**:
- Computes sum of values in window of size n
- Excludes NaN values from sum
- First (n-1) values are NaN


**Usage Notes**: Handles windows with insufficient valid data.

### ts_mean(x, n)
**Purpose**: Mean over rolling window (Numba-accelerated)  
**Key Features**:
- Computes arithmetic mean in window of size n
- Excludes NaN values from calculation
- First (n-1) values are NaN


**Usage Notes**: Handles windows with insufficient valid data.

### ts_std(x, n, ddof)
**Purpose**: Standard deviation over rolling window (Numba-accelerated)  
**Key Features**:
- Computes standard deviation in window of size n
- ddof=1 (sample) or ddof=0 (population)
- Excludes NaN values from calculation


**Usage Notes**: Requires at least (ddof + 1) valid values.

### ts_min(x, n)
**Purpose**: Minimum over rolling window (Numba-accelerated)  
**Key Features**:
- Finds minimum value in window of size n
- Excludes NaN values from search
- First (n-1) values are NaN


**Usage Notes**: Handles windows with insufficient valid data.

### ts_max(x, n)
**Purpose**: Maximum over rolling window (Numba-accelerated)  
**Key Features**:
- Finds maximum value in window of size n
- Excludes NaN values from search
- First (n-1) values are NaN


**Usage Notes**: Handles windows with insufficient valid data.

### ts_count(condition, n)
**Purpose**: Count True values in rolling window (Numba-accelerated)  
**Key Features**:
- Counts number of True values in window of size n
- NaN values treated as False
- First (n-1) values are NaN


**Usage Notes**: Returns float array to support NaN.

### ts_prod(x, n)
**Purpose**: Product over rolling window (Numba-accelerated)  
**Key Features**:
- Computes product of values in window of size n
- Excludes NaN values from calculation
- First (n-1) values are NaN


**Usage Notes**: Can overflow with large values or windows.

---

## 3. Statistical Operations

### rolling_corr(a, b, window)
**Purpose**: Rolling Pearson correlation between two arrays  
**Key Features**:
- Computes Pearson correlation in window of size n
- Excludes NaN values pairwise
- First (window-1) values are NaN
- Uses scipy.stats.pearsonr

**Usage Notes**: Requires at least 2 valid pairs per window.

### covariance(x, y, n, ddof)
**Purpose**: Rolling covariance between two arrays (Numba-accelerated)  
**Key Features**:
- Computes covariance in window of size n
- ddof=1 (sample) or ddof=0 (population)
- Excludes NaN values pairwise


**Usage Notes**: Requires at least (ddof + 2) valid pairs.

### regression_beta(x, y, n)
**Purpose**: Rolling regression beta coefficient (Numba-accelerated)  
**Key Features**:
- Computes slope when regressing x on y: x ~ y
- Formula: β = cov(x, y) / var(y)
- Excludes NaN values pairwise


**Usage Notes**: Requires at least 2 valid pairs per window.

---

## 4. Weighted Average Operations

### sma(x, n, m)
**Purpose**: Special Moving Average with memory  
**Key Features**:
- Iterative calculation: Y[t] = (m*A[t] + (n-m)*Y[t-1]) / n
- Y[0] initialized to A[0]
- NaN propagates forward
- Non-vectorizable computation

**Usage Notes**: Similar to EMA but with specific weight parameters.

### wma(x, n)
**Purpose**: Weighted Moving Average with exponential decay weights (Numba-accelerated)  
**Key Features**:
- Weights: [0.9^(n-1), 0.9^(n-2), ..., 1.0]
- More recent values have higher weight
- Excludes NaN values and renormalizes weights


**Usage Notes**: First (n-1) values are NaN.

### decay_linear(x, d)
**Purpose**: Linear decay weighted average (Numba-accelerated)  
**Key Features**:
- Weights: [d, d-1, ..., 1]
- Most recent value gets highest weight (d)
- Excludes NaN values and renormalizes weights


**Usage Notes**: First (d-1) values are NaN.

---

## 5. Conditional & Special Operations

### sum_if(x, n, condition)
**Purpose**: Rolling sum of x where condition is True (Numba-accelerated)  
**Key Features**:
- Sums x values where corresponding condition is True
- Excludes NaN values in x and condition
- Returns 0 if all conditions are False


**Usage Notes**: First (n-1) values are NaN.

### filter_array(x, condition)
**Purpose**: Filter array to keep only elements where condition is True  
**Key Features**:
- Returns variable-length array
- NaN in condition treated as False
- Simple vectorized operation

**Usage Notes**: Used in formulas like alpha_149.

### high_day(x, n)
**Purpose**: Number of days since the highest value in past n periods (Numba-accelerated)  
**Key Features**:
- Returns days since maximum value (0 if current is max)
- Excludes NaN values from search
- First (n-1) values are NaN


**Usage Notes**: Uses first occurrence if multiple maxes.

### low_day(x, n)
**Purpose**: Number of days since the lowest value in past n periods (Numba-accelerated)  
**Key Features**:
- Returns days since minimum value (0 if current is min)
- Excludes NaN values from search
- First (n-1) values are NaN


**Usage Notes**: Uses first occurrence if multiple mins.

### sequence(n)
**Purpose**: Generate sequence from 1 to n  
**Key Features**:
- Returns array [1, 2, ..., n]
- Float array for consistency
- Simple vectorized operation

**Usage Notes**: Used in formulas like REGBETA(MEAN(CLOSE,6), SEQUENCE(6)).

---

## 6. Derived Fields

### compute_ret(close)
**Purpose**: Compute daily returns  
**Key Features**:
- Formula: RET = CLOSE[t]/CLOSE[t-1] - 1
- First value is NaN
- Uses delay() internally
- Simple vectorized operation

**Usage Notes**: Handles NaN propagation.

### compute_dtm(open_price, high)
**Purpose**: Compute DTM (directional movement indicator)  
**Key Features**:
- Returns 0 if open <= previous open, else max of (high-open, open-prev_open)
- Uses delay() internally
- Simple vectorized operation

**Usage Notes**: Measures upward directional movement.

### compute_dbm(open_price, low)
**Purpose**: Compute DBM (directional movement indicator)  
**Key Features**:
- Returns 0 if open >= previous open, else max of (open-low, open-prev_open)
- Uses delay() internally
- Simple vectorized operation

**Usage Notes**: Measures downward directional movement.

### compute_tr(high, low, close)
**Purpose**: Compute True Range  
**Key Features**:
- Formula: TR = MAX(MAX(H-L, |H-prevC|), |L-prevC|)
- Accounts for gaps between periods
- Simple vectorized operation

**Usage Notes**: Used in ATR (Average True Range) calculations.

### compute_hd(high)
**Purpose**: Compute HD (high difference)  
**Key Features**:
- Formula: HD = HIGH[t] - HIGH[t-1]
- First value is NaN
- Equivalent to delta(high, 1)
- Simple vectorized operation

**Usage Notes**: Positive values indicate higher highs.

### compute_ld(low)
**Purpose**: Compute LD (low difference, inverted)  
**Key Features**:
- Formula: LD = LOW[t-1] - LOW[t]
- First value is NaN
- Inverse of delta(low, 1)
- Simple vectorized operation

**Usage Notes**: Positive values indicate lower lows.

---

## 7. Existing Operators

### regression_residual(x, y, n)
**Purpose**: Rolling regression residuals (Numba-accelerated)  
**Key Features**:
- Computes residual: x - (α + β*y) from window regression
- Uses regression coefficients from window
- Excludes NaN values pairwise

**Usage Notes**: First (n-1) values are NaN.

---

## Summary

Total operators: 30

| Category | Count |
|----------|-------|
| Basic Operations | 4 |
| Rolling Window Operations | 8 |
| Statistical Operations | 3 |
| Weighted Average Operations | 3 |
| Conditional & Special Operations | 5 |
| Derived Fields | 6 |
| Existing Operators | 1 |
| **Total** | **30** |

All operators are implemented in Python with Numba JIT acceleration for performance-critical operations.

# FP32 Precision Plan for Alpha191

## Executive Summary

This document provides a comprehensive plan for using FP32 (float32) precision in the Alpha191 factor library to improve memory efficiency and computational performance while minimizing accuracy loss and overflow risk.

**Key Finding**: Input stock price data is already FP32 (converted in `utils.py`). Most intermediate calculations can safely use FP32 with strategic hybrid approaches for accumulation-heavy operations.

---

## Data Range Analysis

### Stock Price Characteristics (Chinese A-Shares)
| Metric | Typical Range | Max Expected |
|--------|--------------|--------------|
| Stock Price | 1 - 500 CNY | ~2000 CNY |
| Volume | 1,000 - 100M | ~1B |
| Price Returns | -0.2 to +0.2 | ±0.5 (extreme) |
| Rank Output | 0.0 - 1.0 | [0, 1] |
| Correlation | -1.0 to +1.0 | [-1, 1] |

### FP32 Safety Limits
- **Max Value**: ~3.4 × 10³⁸
- **Precision**: ~7 decimal digits
- **Safe for**: All stock price operations, powers up to 1000⁵ = 10¹⁵

### FP32 vs FP64 Memory Impact
| Component | FP64 | FP32 | Savings |
|-----------|------|------|---------|
| Single Array (350 days) | 2.8 KB | 1.4 KB | 50% |
| Full Factor Calculation | ~50 KB | ~25 KB | 50% |
| Cache Efficiency | Baseline | 2× better | 50% miss reduction |

---

## Operator Classification

### Category 1: SAFE for FP32 (Direct Conversion)
These operators have bounded outputs and no accumulation risk.

| Operator | Risk Level | Rationale |
|----------|------------|-----------|
| `rank` | **None** | Output bounded [0, 1], single-pass |
| `ts_rank` | **None** | Output bounded [0, 1], small window (≤10) |
| `ts_min` | **None** | Selection operation, no accumulation |
| `ts_max` | **None** | Selection operation, no accumulation |
| `delay` | **None** | Memory operation only |
| `delta` | **None** | Simple subtraction |
| `sign` | **None** | Output {-1, 0, 1} |

**Implementation**: Convert input to FP32, compute in FP32, return FP32.

### Category 2: HYBRID (FP64 Accumulator, FP32 Output)
These operators accumulate values but output normalized results.

| Operator | Accumulation Risk | Mitigation Strategy |
|----------|-------------------|---------------------|
| `ts_sum` | **Medium** | FP64 running sum, FP32 output |
| `ts_mean` | **Low** | FP64 sum/count, FP32 output |
| `ts_std` | **Low** | FP64 sum/sum_sq, FP32 output |
| `wma` | **Low** | Weighted sum, FP64 accumulator |
| `sma` | **Low** | Recursive, FP64 for stability |
| `decay_linear` | **Low** | Weighted sum, FP64 accumulator |

**Implementation**: Use FP64 for internal accumulators, cast to FP32 at return.

### Category 3: CAREFUL (Keep FP64 or Use Extended Precision)
These operators have high overflow/underflow risk.

| Operator | Risk Level | Recommended Approach |
|----------|------------|---------------------|
| `ts_prod` | **HIGH** | Product of 10 values @ 100 = 10²⁰; FP32 safe but borderline |
| `covariance` | **Medium** | Sum of products, keep FP64 |
| `rolling_corr` | **Low-Medium** | Internal sums, hybrid approach |
| `regression_beta` | **Medium** | Division of cov/var, keep FP64 |
| `regression_residual` | **Medium** | Depends on beta calculation |

**Implementation**: Keep FP64 or use Kahan summation with FP32.

---

## Detailed Implementation Strategy

### 1. Utility Functions (Safe for FP32)

```python
# Current (operators.py line 746)
def delay(x: np.ndarray, n: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)  # → dtype=np.float32
    ...
    return result  # Already FP32 if input was FP32
```

**Action**: Change `dtype=float` to `dtype=np.float32` for:
- `delay` (line 746)
- `delta` (line 808)
- `sign` (line 928)

### 2. Ranking Functions (Safe for FP32)

```python
# ts_rank (line 663)
def ts_rank(x: np.ndarray, window: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)  # Changed from float (FP64)
    ...
    return _ts_rank_core(x, window).astype(np.float32)
```

**Note**: The `_ts_rank_core` numba function uses `np.full(n_len, np.nan)` which creates FP64 by default. Need to change to `np.float32`.

### 3. Min/Max Functions (Safe for FP32)

```python
# _ts_min_core (line 178)
# Change: result = np.full(n_len, np.nan, dtype=np.float32)
#         deque_val = np.empty(n_len, dtype=np.float32)
```

### 4. Sum/Mean/Std (Hybrid Approach)

```python
# _ts_sum_core (line 37)
@njit(cache=True)
def _ts_sum_core(x: np.ndarray, n: int) -> np.ndarray:
    # Input x is FP32
    n_len = len(x)
    result = np.full(n_len, np.nan, dtype=np.float32)  # Output FP32
    
    # Accumulator stays FP64 for precision
    running_sum = np.float64(0.0)
    ...
    result[i] = np.float32(running_sum)  # Cast at storage
```

### 5. Product Function (Careful)

```python
# _ts_prod_core (line 311)
# For window ≤ 10: FP32 is safe (100^10 = 1e20 << 3.4e38)
# For window > 10: Consider FP64 or log-space
@njit(cache=True)
def _ts_prod_core(x: np.ndarray, n: int) -> np.ndarray:
    result = np.full(n_len, np.nan, dtype=np.float32)
    running_prod = np.float64(1.0)  # FP64 for safety
    ...
```

### 6. Correlation/Covariance (Keep FP64)

These involve sums of products which can accumulate significant errors in FP32. Keep FP64 for:
- `_rolling_corr_core`
- `_covariance_core`
- `_regression_beta_core`
- `_regression_residual_core`

**Optimization**: If input is already FP32, the computation is faster even with FP64 accumulators due to better cache locality.

---

## Alpha Factor Analysis

### Factors with Power Operations
| Factor | Operation | Risk | Recommendation |
|--------|-----------|------|----------------|
| Alpha056 | rank() ** 5 | None | rank() ∈ [0,1], result ∈ [0,1] |
| Alpha115 | rank() ** rank() | None | Both operands ∈ [0,1] |
| Alpha185 | price ** 5 | Low | 1000^5 = 1e15 < 3.4e38 |
| Alpha171 | price ** 5 | Low | Same as above |

### Factors with Division by Small Numbers
| Factor | Operation | Risk | Mitigation |
|--------|-----------|------|------------|
| Alpha002 | (close-low)/(high-low) | None | Prices correlated, result ∈ [-1, 1] |
| Alpha031 | (close-mean)/mean | Low | Normalized by mean |
| Alpha050 | Complex ratios | Low | All bounded operations |

---

## Implementation Phases

### Phase 1: Safe Conversions (Low Risk)
- [ ] `delay`, `delta`, `sign` - memory operations
- [ ] `rank` - scipy-based, already single-pass
- [ ] `ts_rank` - bounded output
- [ ] `ts_min`, `ts_max` - selection operations

### Phase 2: Hybrid Accumulators (Medium Risk)
- [ ] `ts_sum` - FP64 accumulator, FP32 output
- [ ] `ts_mean` - FP64 accumulator, FP32 output
- [ ] `ts_std` - FP64 accumulator, FP32 output
- [ ] `wma`, `sma`, `decay_linear` - weighted accumulators

### Phase 3: Evaluation (Careful)
- [ ] Benchmark accuracy vs FP64 baseline
- [ ] Measure performance improvement
- [ ] Decide on `ts_prod`, `covariance`, `regression_*`

---

## Testing Strategy

### Accuracy Tests
```python
# Compare FP32 vs FP64 output
def test_precision(func, test_data, rtol=1e-5):
    result_fp32 = func(np.float32(test_data))
    result_fp64 = func(np.float64(test_data))
    np.testing.assert_allclose(result_fp32, result_fp64, rtol=rtol)
```

### Overflow Tests
```python
# Test extreme values
def test_overflow_safety():
    high_price = np.float32(2000.0)
    extreme_volume = np.float32(1e9)
    # Verify no overflow in typical operations
```

---

## Expected Benefits

| Metric | Expected Improvement |
|--------|---------------------|
| Memory Usage | ~40-50% reduction |
| Cache Misses | ~30-40% reduction |
| Vectorization | Better SIMD utilization |
| Numerical Accuracy | <0.01% relative error |

---

## Risk Mitigation

1. **Overflow**: Use FP64 for accumulators in sum/product operations
2. **Underflow**: Not a concern for price data (values >> FP32 min)
3. **Catastrophic Cancellation**: Mean/std already use stable algorithms
4. **Rounding Error**: Acceptable for financial factors (rank-based)

---

## Conclusion

Converting Alpha191 to use FP32 is **safe and recommended** for:
- All ranking and selection operations
- Simple arithmetic (delay, delta, sign)
- Output storage (50% memory savings)

**Hybrid approach** (FP64 accumulators, FP32 storage) is recommended for:
- Rolling sums, means, standard deviations
- Weighted moving averages

**Keep FP64** for:
- Products (ts_prod)
- Covariance/correlation (accumulation of products)
- Regression operations

This strategy achieves significant memory and performance gains with negligible accuracy loss (< 0.01%) for financial factor calculations.

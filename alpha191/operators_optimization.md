### Key Speed Optimizations

The code is already highly optimized: Numba JIT on the expensive rolling loops, small effective array sizes (≤350 rows due to `run_alpha_factor` slicing), and vectorized NumPy for simple ops. Quadratic-time loops (O(n × window)) are acceptable here—350² ≈ 120k operations per call is negligible with Numba (~millions–billions ops/sec).

However, several rolling operators can be upgraded from O(n × window) loops to **O(n) total** using **sliding-window running aggregates**. This is a classic optimization for rolling stats ignoring NaNs: maintain running sums/counts and update them in O(1) when sliding the window (add new value, subtract outgoing value, but only if non-NaN).

Gains:
- **5–300× theoretical speedup** per call (depending on window size).
- Biggest wins on frequently used ops (ts_mean, ts_sum, rolling_corr, rolling_cov, regression_beta).
- Still correct NaN-skipping semantics.
- Practical impact modest at n=350 but significant if lookback/window increased or many nested calls.

#### 1. **Primary Targets: Convert to Running Aggregates (Biggest Wins)**
Rewrite these with running variables instead of per-window loops:

- **ts_sum / rolling_sum**
- **ts_mean / rolling_mean**
- **ts_count** (count non-zero non-NaN)
- **ts_std** (if used; maintain sum + sum_sq + count)
- **rolling_corr**
- **rolling_cov / covariance**
- **regression_beta**
- **regression_residual** (uses same aggregates as beta)

**Pattern (Numba @njit)**:
```python
@njit(cache=True)
def _new_ts_sum_core(x: np.ndarray, window: int) -> np.ndarray:
    n = len(x)
    result = np.full(n, np.nan)
    if n < window:
        return result
    
    # Initial window (O(window) once)
    running_sum = 0.0
    running_count = 0
    for j in range(window):
        if not np.isnan(x[j]):
            running_sum += x[j]
            running_count += 1
    
    if running_count > 0:
        result[window-1] = running_sum
    
    # Slide window (O(1) per step)
    for i in range(window, n):
        new_val = x[i]
        old_val = x[i - window]
        
        if not np.isnan(new_val):
            running_sum += new_val
            running_count += 1
        if not np.isnan(old_val):
            running_sum -= old_val
            running_count -= 1
        
        if running_count > 0:
            result[i] = running_sum
        # else NaN (already filled)
    
    return result
```

Adapt similarly for others:
- **Mean**: add `running_sum` + `running_count` → `result[i] = running_sum / running_count if running_count > 0`.
- **Count non-zero**: `running_nonzero` (increment only if `not np.isnan(val) and val != 0`).
  - If `running_count == 0`: NaN, else `running_nonzero`.
- **Std** (biased for simplicity, or adjust for ddof):
  - Maintain `sum_sq` alongside `sum` + `count`.
  - `var = sum_sq / count - (sum / count)**2` (or unbiased formula).
- **Corr / Cov / Beta**:
  - Maintain 5 running sums (`sum_a`, `sum_b`, `sum_ab`, `sum_a2`, `sum_b2`) + `pair_count`.
  - Add/subtract only when **both** values non-NaN.
  - Compute final stats exactly as current one-pass formula.
- **Regression residual**: same aggregates, but only compute residual if current `x[i]` and `y[i]` non-NaN.

**Why faster**: Eliminates inner loop. Initial O(window) + (n - window) × O(1).

#### 2. **Secondary Targets (Nice-to-Have, Smaller Gains)**
- **decay_linear** (and similar weighted like wma if present):
  - Weights fixed → maintain `running_weighted_sum` + `running_weight_sum`.
  - Incoming always highest weight (d), outgoing always lowest (1).
  - Add: if new non-NaN, `weighted_sum += new * d`, `weight_sum += d`.
  - Subtract: if old non-NaN, `weighted_sum -= old * 1`, `weight_sum -= 1`.
  - But wait: after slide, previous weights shift down by 1 → all middle weights change!
  - Cannot O(1) without extra running sums (e.g., maintain multiple weighted levels).
  - **Conclusion**: Not easily O(n) → leave loop (small n anyway).

- **ts_min / ts_max / high_day / low_day**:
  - Possible with monotonic deque (store indices, skip NaNs by not adding them).
  - Numba-compatible but ~50–100 lines extra code.
  - Gains minor (current already fast).

- **ts_prod**:
  - Multiplicative → hard to slide (division on remove risky with 0s/negatives).
  - Leave loop.

- **ts_rank**:
  - Requires order stats → no simple O(n).
  - Could extract windows + np.sort + search, but slower constants.
  - Leave loop (acceptable at n≤350).

#### 3. **Minor Cleanups (Negligible Speed but Good)**
- Precompute constants outside loops (e.g., in decay_exp, precompute full weight array once).
- Remove unused `weight_sum_total` in decay_linear.
- In public wrappers: ensure `np.asarray(x, float)` once, validate early.
- For sma/decay_exp: already O(n) recursive → excellent, no change.

#### Summary of Impact
| Operator              | Current Complexity | Proposed | Expected Speedup (n=350, window=100) |
|-----------------------|--------------------|---------|-------------------------------------|
| ts_sum/mean/count     | O(n×w)             | O(n)    | ~100× theoretical                   |
| rolling_corr/cov/beta | O(n×w)             | O(n)    | ~100× theoretical                   |
| ts_std                | O(n×w)             | O(n)    | ~100× theoretical                   |
| ts_min/max/rank/prod  | O(n×w)             | unchanged | Minor/none                         |
| high/low_day          | O(n×w)             | optional deque | 2–5× if implemented                |

Implement the running-aggregate versions for the additive stats first—highest ROI, clean code, future-proof. The rest is already near-optimal for this scale.
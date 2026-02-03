# ICtest.py Output Documentation

This document explains all output metrics produced by the `assess_alpha()` function in `ICtest.py`.

## Overview

`ICtest.py` assesses alpha factors using Spearman Rank IC (Information Coefficient). It computes the correlation between factor rankings and forward return rankings across time, providing comprehensive performance metrics.

## Usage

```bash
python ICtest.py <alpha_name> [horizon] [benchmark] [rolling_window]
```

- `alpha_name`: Number (1-191) or format 'alpha001'
- `horizon`: Forward return horizon in days (default: 20)
- `benchmark`: hs300, zz500, or zz800 (default: zz800)
- `rolling_window`: Rolling window for IC stability metrics (default: 60)

## Output Metrics

### Basic Information

| Metric | Type | Description |
|--------|------|-------------|
| `alpha` | str | Name of the alpha factor being tested (e.g., "alpha001") |
| `benchmark` | str | Benchmark used for testing (hs300, zz500, or zz800) |
| `horizon` | int | Forward return horizon in days |
| `rolling_window` | int | Window size used for rolling IC calculations |

---

## Core IC Metrics

### IC_mean
**Type:** float

**Description:** The mean (average) of the IC series across all dates.

**Interpretation:**
- Positive values indicate the factor has predictive power (higher factor values tend to precede higher returns)
- Negative values indicate inverse predictive power
- Values closer to 0 indicate weak or no predictive relationship
- Typical good values: > 0.02 for daily IC, > 0.05 for longer horizons

**Formula:**
```
IC_mean = mean(IC_t) for all t
```

---

### IC_std
**Type:** float

**Description:** The standard deviation of the IC series, measuring the volatility of the factor's predictive performance.

**Interpretation:**
- Lower values indicate more stable performance
- Higher values indicate more volatile performance
- Used in calculating ICIR (Information Coefficient Information Ratio)

**Formula:**
```
IC_std = std(IC_t) for all t
```

---

### IC_winrate
**Type:** float (0-1)

**Description:** The percentage of days with positive IC values.

**Interpretation:**
- Values > 0.5 indicate the factor is correct more often than wrong
- Values > 0.55 are generally considered good
- Values < 0.5 indicate the factor is wrong more often than right

**Formula:**
```
IC_winrate = count(IC_t > 0) / total_observations
```

---

### ICIR (Information Coefficient Information Ratio)
**Type:** float

**Description:** The ratio of IC mean to IC standard deviation, measuring the consistency of the factor's predictive power.

**Interpretation:**
- Higher values indicate more consistent performance
- Values > 0.5 are generally considered good
- Values > 1.0 are excellent
- Negative values indicate the factor consistently predicts in the wrong direction

**Formula:**
```
ICIR = IC_mean / IC_std
```

---

### t_stat
**Type:** float

**Description:** The t-statistic for the IC mean, testing whether the IC is significantly different from zero.

**Interpretation:**
- Values > 2.0 indicate statistical significance at 95% confidence level
- Values > 2.58 indicate statistical significance at 99% confidence level
- Higher absolute values indicate stronger evidence of predictive power

**Formula:**
```
t_stat = ICIR * sqrt(n_obs)
```

---

### n_obs
**Type:** int

**Description:** The number of observations (dates) in the IC series.

**Interpretation:**
- More observations generally lead to more reliable statistics
- Minimum of 30 observations required for IC calculation (per `fast_pearson` function)

---

### IC_series
**Type:** pandas.Series

**Description:** The full time series of IC values for each date.

**Interpretation:**
- Can be used for time-series analysis
- Useful for visualizing performance over time
- Can identify periods of strength or weakness

---

## Distribution Metrics

### IC_median
**Type:** float

**Description:** The median of the IC series.

**Interpretation:**
- Less sensitive to outliers than IC_mean
- Provides a robust measure of central tendency
- Compare with IC_mean to assess skewness impact
- If IC_median > IC_mean, distribution is left-skewed (more extreme negative values)
- If IC_median < IC_mean, distribution is right-skewed (more extreme positive values)

**Formula:**
```
IC_median = median(IC_t) for all t
```

---

### IC_skew
**Type:** float

**Description:** The skewness of the IC series distribution, measuring asymmetry.

**Interpretation:**
- Values near 0 indicate symmetric distribution
- Positive values indicate right-skewed (long tail of positive values)
- Negative values indicate left-skewed (long tail of negative values)
- Extreme skewness may indicate unstable performance or regime-dependent behavior

**Formula:**
```
IC_skew = scipy.stats.skew(IC_series)
```

---

## Stability Metrics

### rolling_IC_min
**Type:** float

**Description:** The minimum value of the rolling IC series.

**Interpretation:**
- Represents the worst performance over any rolling window period
- Lower values indicate periods of significant underperformance
- Important for understanding downside risk
- Used with `rolling_IC_max` to assess performance range

**Formula:**
```
rolling_IC_min = min(rolling_mean(IC_t, window=rolling_window))
```

---

### rolling_IC_max
**Type:** float

**Description:** The maximum value of the rolling IC series.

**Interpretation:**
- Represents the best performance over any rolling window period
- Higher values indicate periods of strong predictive power
- Used with `rolling_IC_min` to assess performance range
- Large spread between min and max indicates unstable performance

**Formula:**
```
rolling_IC_max = max(rolling_mean(IC_t, window=rolling_window))
```

---

### IC_max_drawdown
**Type:** float (typically negative)

**Description:** The maximum drawdown of the IC series, measuring the largest peak-to-trough decline.

**Interpretation:**
- More negative values indicate larger drawdowns
- Values closer to 0 indicate more stable performance
- Important for understanding worst-case scenarios
- Typical good values: > -0.05 (drawdown less than 5%)

**Formula:**
```
IC_cummax = cumulative_max(IC_series)
IC_drawdown = (IC_series - IC_cummax) / IC_cummax
IC_max_drawdown = min(IC_drawdown)
```

---

## Coverage Metrics

### avg_cs_size
**Type:** float

**Description:** The average cross-sectional size, i.e., the average number of stocks with valid factor values per date.

**Interpretation:**
- Higher values indicate better data coverage
- Lower values may indicate data quality issues or limited universe
- Important for understanding the factor's applicability
- Values too low may reduce statistical significance

**Formula:**
```
avg_cs_size = mean(count(non-NaN values per date in factor_matrix))
```

---

### min_cs_size
**Type:** int

**Description:** The minimum cross-sectional size, i.e., the minimum number of stocks with valid factor values on any date.

**Interpretation:**
- Indicates the worst-case coverage
- Very low values may indicate data gaps or issues
- Important for understanding the factor's reliability
- If too low, may need to filter out those dates

**Formula:**
```
min_cs_size = min(count(non-NaN values per date in factor_matrix))
```

---

## Directional Asymmetry Metrics

### IC_mean_pos
**Type:** float

**Description:** The mean of positive IC values only.

**Interpretation:**
- Measures the average performance when the factor is correct
- Higher values indicate stronger predictive power when right
- Compare with `IC_mean_neg` to understand asymmetry
- If `IC_mean_pos` >> `|IC_mean_neg|`, factor has strong upside when correct

**Formula:**
```
IC_mean_pos = mean(IC_t | IC_t > 0)
```

---

### IC_mean_neg
**Type:** float (typically negative)

**Description:** The mean of negative IC values only.

**Interpretation:**
- Measures the average performance when the factor is wrong
- More negative values indicate stronger inverse predictive power when wrong
- Compare with `IC_mean_pos` to understand asymmetry
- If `|IC_mean_neg|` >> `IC_mean_pos`, factor has strong downside when wrong

**Formula:**
```
IC_mean_neg = mean(IC_t | IC_t < 0)
```

---

## Metric Relationships

### Key Ratios

1. **ICIR = IC_mean / IC_std**
   - Measures consistency of predictive power

2. **IC_mean_pos / |IC_mean_neg|**
   - Measures directional asymmetry
   - > 1: Stronger upside when correct
   - < 1: Stronger downside when wrong

3. **rolling_IC_max / |rolling_IC_min|**
   - Measures performance stability
   - Lower values indicate more stable performance

### Interpretation Guidelines

| Metric | Excellent | Good | Fair | Poor |
|--------|----------|------|------|------|
| IC_mean | > 0.05 | 0.02-0.05 | 0.01-0.02 | < 0.01 |
| ICIR | > 1.0 | 0.5-1.0 | 0.3-0.5 | < 0.3 |
| IC_winrate | > 0.55 | 0.52-0.55 | 0.50-0.52 | < 0.50 |
| t_stat | > 3.0 | 2.0-3.0 | 1.5-2.0 | < 1.5 |
| IC_max_drawdown | > -0.02 | -0.02 to -0.05 | -0.05 to -0.10 | < -0.10 |

---

## Example Output

```python
{
    "alpha": "alpha001",
    "benchmark": "zz800",
    "horizon": 20,
    "rolling_window": 60,
    "IC_mean": 0.0234,
    "IC_std": 0.0456,
    "IC_winrate": 0.5423,
    "ICIR": 0.5123,
    "t_stat": 4.5678,
    "n_obs": 1200,
    "IC_series": pd.Series(...),
    "IC_median": 0.0212,
    "IC_skew": 0.3456,
    "rolling_IC_min": -0.0123,
    "rolling_IC_max": 0.0456,
    "IC_max_drawdown": -0.0345,
    "avg_cs_size": 750.5,
    "min_cs_size": 680,
    "IC_mean_pos": 0.0345,
    "IC_mean_neg": -0.0234
}
```

---

## Technical Details

### IC Calculation Method

The IC is calculated using Spearman Rank correlation:
1. Rank factor values cross-sectionally for each date
2. Rank forward returns cross-sectionally for each date
3. Compute Pearson correlation between the two rank series
4. Use Numba-accelerated computation for performance

### Minimum Requirements

- Minimum 30 valid observations per date for IC calculation
- NaN values are handled by the `fast_pearson` function
- Rolling window uses `min_periods=1` to handle edge cases

### Performance Considerations

- Numba JIT compilation accelerates IC calculations
- Factor and return matrices are stored as float32 for memory efficiency
- Cross-sectional ranking is performed using pandas `rank()` method

---

## References

- Spearman Rank Correlation: Non-parametric measure of rank correlation
- Information Coefficient (IC): Standard metric for factor performance
- Information Ratio (IR): Risk-adjusted measure of predictive power

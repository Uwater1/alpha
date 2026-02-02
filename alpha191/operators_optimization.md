Looking at your code, there are several opportunities for speed optimization:

## Major Optimization Opportunities

### 1. **`ts_min` and `ts_max` - Use Monotonic Deque**
Currently O(n*window), can be O(n) with a deque-based algorithm:

```python
@njit(cache=True)
def _ts_min_core_optimized(x: np.ndarray, n: int) -> np.ndarray:
    """O(n) monotonic deque approach instead of O(n*window)"""
    n_len = len(x)
    result = np.full(n_len, np.nan)
    
    # Deque stores (index, value) pairs
    # We'll use arrays to simulate deque in numba
    deque_idx = np.empty(n_len, dtype=np.int64)
    deque_val = np.empty(n_len, dtype=np.float64)
    front = 0
    back = 0
    
    for i in range(n_len):
        if np.isnan(x[i]):
            continue
            
        # Remove elements outside window
        while back > front and deque_idx[front] <= i - n:
            front += 1
        
        # Remove elements >= current (for min; reverse for max)
        while back > front and deque_val[back - 1] >= x[i]:
            back -= 1
        
        # Add current element
        deque_idx[back] = i
        deque_val[back] = x[i]
        back += 1
        
        # Record result
        if i >= n - 1 and back > front:
            result[i] = deque_val[front]
    
    return result
```

### 2. **`ts_prod` - Avoid Recomputation**
Current implementation recalculates entire product each window. Use rolling multiplication/division:

```python
@njit(cache=True)
def _ts_prod_core_optimized(x: np.ndarray, n: int) -> np.ndarray:
    """Use rolling product with careful zero/NaN handling"""
    n_len = len(x)
    result = np.full(n_len, np.nan)
    
    # Track product and count of zeros in window
    running_prod = 1.0
    zero_count = 0
    valid_count = 0
    
    # Initialize first window
    for j in range(n):
        if not np.isnan(x[j]):
            if x[j] == 0:
                zero_count += 1
            else:
                running_prod *= x[j]
            valid_count += 1
    
    if valid_count > 0:
        result[n - 1] = 0.0 if zero_count > 0 else running_prod
    
    # Slide window
    for i in range(n, n_len):
        # Remove old value
        if not np.isnan(x[i - n]):
            if x[i - n] == 0:
                zero_count -= 1
            else:
                running_prod /= x[i - n]
            valid_count -= 1
        
        # Add new value
        if not np.isnan(x[i]):
            if x[i] == 0:
                zero_count += 1
            else:
                running_prod *= x[i]
            valid_count += 1
        
        if valid_count > 0:
            result[i] = 0.0 if zero_count > 0 else running_prod
    
    return result
```

### 3. **`ts_count` - Already Optimized ✓**
Your implementation is already O(n) with rolling window.

### 4. **`_ts_rank_core` - Can Use Fenwick Tree**
Currently O(n*window²), can be O(n*window*log(window)) with a balanced tree structure, though this is complex in Numba.

### 5. **Decay Functions - Can Use Recursive Formula**
Both `decay_linear` and `decay_exp` recalculate sums. Can optimize with rolling approach:

```python
@njit(cache=True)
def _decay_linear_core_optimized(x: np.ndarray, d: int) -> np.ndarray:
    """Use rolling update formula instead of recalculating"""
    n_len = len(x)
    result = np.full(n_len, np.nan)
    
    # weights: [1, 2, 3, ..., d]
    # When sliding: new_sum = old_sum - x[i-d]*1 - x[i-d+1]*2 - ... 
    #                        + x[i-d+1]*1 + x[i-d+2]*2 + ... + x[i]*d
    # Simplifies to: new_sum = old_sum + x[i]*d - sum(x[i-d+1:i])
    
    # This is still complex; simpler to pre-accumulate
    # For now, current implementation is reasonable
    return _decay_linear_core(x, d)  # Keep current
```

## Performance Impact Summary

| Function | Current | Optimized | Speedup |
|----------|---------|-----------|---------|
| `ts_min/ts_max` | O(n·w) | O(n) | **~w× faster** |
| `ts_prod` | O(n·w) | O(n) | **~w× faster** |
| `ts_rank` | O(n·w²) | O(n·w·log w) | Modest |
| Others | Already O(n) | - | - |

## Additional Minor Optimizations

1. **Pre-allocate in `rank()`**: Use `np.empty` instead of `np.full` then fill
2. **Vectorize `sign()`**: Already optimal (just wraps `np.sign`)
3. **Cache frequently used values**: Store `0.9` as constant in `_wma_core`

## Recommendation

**Implement optimizations for `ts_min`, `ts_max`, and `ts_prod` first** - these will give the biggest gains, especially for large window sizes. The monotonic deque approach for min/max is well-established and straightforward to implement in Numba.

Would you like me to provide complete optimized implementations for any of these functions?
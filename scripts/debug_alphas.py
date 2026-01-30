"""
Debug script to understand alpha factor behavior.
"""

import sys
sys.path.insert(0, '..')

import numpy as np
import pandas as pd
from alpha191.alpha005 import alpha_005
from alpha191.alpha006 import alpha_006
from alpha191.alpha007 import alpha_007
from alpha191.alpha008 import alpha_008
from alpha191.alpha010 import alpha_010
from alpha191.alpha014 import alpha_014
from alpha191.alpha016 import alpha_016
from alpha191.alpha017 import alpha_017
from alpha191.alpha018 import alpha_018
from alpha191.alpha019 import alpha_019
from alpha191.alpha020 import alpha_020

# Create test data
np.random.seed(42)
n = 30
df = pd.DataFrame({
    'date': pd.date_range('2020-01-01', periods=n),
    'open': 100 + np.random.randn(n).cumsum(),
    'high': 102 + np.random.randn(n).cumsum(),
    'low': 98 + np.random.randn(n).cumsum(),
    'close': 101 + np.random.randn(n).cumsum(),
    'volume': np.random.randint(1000000, 5000000, n),
})

print("Testing alpha factors...")
print("=" * 60)

# Test alpha005
result = alpha_005(df)
print(f"alpha_005: {result.values[:10]}")
print(f"  NaN count: {np.sum(np.isnan(result.values[:10]))}")

# Test alpha006
result = alpha_006(df)
print(f"alpha_006: {result.values[:10]}")
print(f"  NaN count: {np.sum(np.isnan(result.values[:10]))}")

# Test alpha007
result = alpha_007(df)
print(f"alpha_007: {result.values[:10]}")
print(f"  NaN count: {np.sum(np.isnan(result.values[:10]))}")

# Test alpha008
result = alpha_008(df)
print(f"alpha_008: {result.values[:10]}")
print(f"  NaN count: {np.sum(np.isnan(result.values[:10]))}")

# Test alpha010
result = alpha_010(df)
print(f"alpha_010: {result.values[:10]}")
print(f"  NaN count: {np.sum(np.isnan(result.values[:10]))}")

# Test alpha014
result = alpha_014(df)
print(f"alpha_014: {result.values[:10]}")
print(f"  NaN count: {np.sum(np.isnan(result.values[:10]))}")

# Test alpha016
result = alpha_016(df)
print(f"alpha_016: {result.values[:10]}")
print(f"  NaN count: {np.sum(np.isnan(result.values[:10]))}")

# Test alpha017
result = alpha_017(df)
print(f"alpha_017: {result.values[:10]}")
print(f"  NaN count: {np.sum(np.isnan(result.values[:10]))}")

# Test alpha018
result = alpha_018(df)
print(f"alpha_018: {result.values[:10]}")
print(f"  NaN count: {np.sum(np.isnan(result.values[:10]))}")

# Test alpha019
result = alpha_019(df)
print(f"alpha_019: {result.values[:10]}")
print(f"  NaN count: {np.sum(np.isnan(result.values[:10]))}")

# Test alpha020
result = alpha_020(df)
print(f"alpha_020: {result.values[:10]}")
print(f"  NaN count: {np.sum(np.isnan(result.values[:10]))}")

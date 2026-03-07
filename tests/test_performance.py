"""
Unit tests for performance metrics in assessment.performance module.
"""

import pandas as pd
import numpy as np
import pytest
from assessment.performance import cumulative_returns

def test_cumulative_returns_basic():
    """Test basic positive and negative returns."""
    returns = pd.Series([0.01, 0.02, -0.01, 0.03])
    expected = pd.Series([
        (1 + 0.01) - 1,
        (1 + 0.01) * (1 + 0.02) - 1,
        (1 + 0.01) * (1 + 0.02) * (1 - 0.01) - 1,
        (1 + 0.01) * (1 + 0.02) * (1 - 0.01) * (1 + 0.03) - 1
    ])
    result = cumulative_returns(returns)
    pd.testing.assert_series_equal(result, expected)

def test_cumulative_returns_zero():
    """Test zero returns."""
    returns = pd.Series([0.0, 0.0, 0.0])
    expected = pd.Series([0.0, 0.0, 0.0])
    result = cumulative_returns(returns)
    pd.testing.assert_series_equal(result, expected)

def test_cumulative_returns_with_nan():
    """Test series with NaN values."""
    returns = pd.Series([0.01, np.nan, 0.02])
    # pandas cumprod skips NaN by default, but the NaN remains in the output series
    expected = pd.Series([0.01, np.nan, 1.0302 - 1])
    result = cumulative_returns(returns)
    pd.testing.assert_series_equal(result, expected)

def test_cumulative_returns_empty():
    """Test empty series."""
    returns = pd.Series([], dtype=float)
    result = cumulative_returns(returns)
    assert result.empty
    assert isinstance(result, pd.Series)

def test_cumulative_returns_single():
    """Test single value series."""
    returns = pd.Series([0.05])
    expected = pd.Series([0.05])
    result = cumulative_returns(returns)
    pd.testing.assert_series_equal(result, expected)

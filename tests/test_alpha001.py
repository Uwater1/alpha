"""
Unit tests for Alpha001 factor.
"""

import unittest
import numpy as np
import pandas as pd
from alpha191.operators import ts_rank, rolling_corr
from alpha191.alpha001 import alpha_001


class TestOperators(unittest.TestCase):
    """Tests for operator functions."""

    def test_ts_rank_known_values(self):
        """Test ts_rank with known values."""
        # Test data: ascending sequence 1, 2, 3, 4, 5, 6
        # With window=6, at position 5 (value=6), rank should be 1.0 (max)
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        result = ts_rank(x, window=6)

        # First 5 should be NaN
        self.assertTrue(np.all(np.isnan(result[:5])))

        # Position 5 (last element), rank should be 1.0 (6 is max in window)
        self.assertEqual(result[5], 1.0)

    def test_ts_rank_known_values_middle(self):
        """Test ts_rank with value in middle of window."""
        # Test data: sequence 1, 2, 3, 4, 5, 3
        # Values 3 appear at positions 2 and 5 (0-indexed)
        # rankdata([1,2,3,4,5,3]) = [1, 2, 3.5, 4, 5, 3.5] (ties get average rank)
        # At position 5, rank of 3 is 3.5
        # Normalized: (3.5 - 1) / (6 - 1) = 2.5 / 5 = 0.5
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 3.0])
        result = ts_rank(x, window=6)

        # Position 5, value=3, rankdata gives 3.5 for tied values
        # Normalized: (3.5 - 1) / 5 = 0.5
        self.assertAlmostEqual(result[5], 0.5, places=5)

    def test_ts_rank_with_nans(self):
        """Test ts_rank handles NaN values in window."""
        # Window with one NaN, last element is valid
        x = np.array([1.0, np.nan, 3.0, 4.0, 5.0, 6.0])
        result = ts_rank(x, window=6)

        # Should compute rank without errors
        self.assertTrue(np.all(np.isnan(result[:5])))
        # Last element (6.0) should have a valid rank
        # valid_data = [1.0, 3.0, 4.0, 5.0, 6.0] (5 elements)
        # rankdata([1, 3, 4, 5, 6]) = [1, 2, 3, 4, 5]
        # 6 is at position 4, rank = 5
        # Normalized: (5 - 1) / (5 - 1) = 1.0
        self.assertFalse(np.isnan(result[5]))
        self.assertAlmostEqual(result[5], 1.0, places=5)

    def test_rolling_corr_known_values(self):
        """Test rolling_corr with known correlation values."""
        # Perfect positive correlation
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        b = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        result = rolling_corr(a, b, window=6)

        # First 5 should be NaN
        self.assertTrue(np.all(np.isnan(result[:5])))

        # Perfect correlation should be ~1.0
        self.assertAlmostEqual(result[5], 1.0, places=5)

    def test_rolling_corr_negative(self):
        """Test rolling_corr with negative correlation."""
        # Perfect negative correlation
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        b = np.array([6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
        result = rolling_corr(a, b, window=6)

        # Perfect negative correlation should be ~-1.0
        self.assertAlmostEqual(result[5], -1.0, places=5)

    def test_rolling_corr_zero(self):
        """Test rolling_corr with zero correlation."""
        # Zero correlation
        a = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        b = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        result = rolling_corr(a, b, window=6)

        # Correlation should be NaN (zero std)
        self.assertTrue(np.isnan(result[5]))


class TestAlpha001(unittest.TestCase):
    """Tests for Alpha001 factor."""

    def setUp(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n = 20
        self.df = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=n),
            'open': 100 + np.random.randn(n).cumsum(),
            'high': 102 + np.random.randn(n).cumsum(),
            'low': 98 + np.random.randn(n).cumsum(),
            'close': 101 + np.random.randn(n).cumsum(),
            'volume': np.random.randint(1000000, 5000000, n),
            'amount': np.random.randint(100000000, 500000000, n),
        })

    def test_alpha001_first_five_nan(self):
        """Test that first 5 values are NaN."""
        result = alpha_001(self.df)

        # First 5 values should be NaN (due to rolling window=6)
        self.assertTrue(np.all(np.isnan(result.values[:5])))

    def test_alpha001_deterministic(self):
        """Test that output is deterministic."""
        result1 = alpha_001(self.df)
        result2 = alpha_001(self.df)

        # Same input should produce same output
        pd.testing.assert_series_equal(result1, result2)

    def test_alpha001_output_length(self):
        """Test that output length matches input length."""
        result = alpha_001(self.df)

        # Output length should match input length
        self.assertEqual(len(result), len(self.df))

    def test_alpha001_output_type(self):
        """Test that output is a pd.Series."""
        result = alpha_001(self.df)

        # Output should be a pandas Series
        self.assertIsInstance(result, pd.Series)

    def test_alpha001_date_index(self):
        """Test that output is indexed by date."""
        result = alpha_001(self.df)

        # Index should be datetime
        self.assertIsInstance(result.index, pd.DatetimeIndex)

    def test_alpha001_missing_columns(self):
        """Test error when required columns are missing."""
        df_missing = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=10),
            'close': [100] * 10,
        })

        with self.assertRaises(ValueError):
            alpha_001(df_missing)


if __name__ == '__main__':
    unittest.main()

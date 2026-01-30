"""
Unit tests for Alpha191 operator functions.
"""

import unittest
import numpy as np
from alpha191.operators import (
    delay, delta, rank, sign, ts_sum, ts_mean, ts_std, ts_min, ts_max, ts_count, ts_prod,
    covariance, regression_beta, regression_residual, sma, wma, decay_linear,
    sum_if, filter_array, high_day, low_day, sequence,
    compute_ret, compute_dtm, compute_dbm, compute_tr, compute_hd, compute_ld
)


class TestDelay(unittest.TestCase):
    """Tests for delay function."""

    def test_delay_basic(self):
        """Test delay with known values."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = delay(x, 2)

        # First 2 values should be NaN
        self.assertTrue(np.all(np.isnan(result[:2])))

        # Remaining values should be shifted
        expected = np.array([np.nan, np.nan, 1.0, 2.0, 3.0])
        np.testing.assert_array_equal(result[:2], expected[:2])
        np.testing.assert_array_equal(result[2:], expected[2:])

    def test_delay_n1(self):
        """Test delay with n=1."""
        x = np.array([10.0, 20.0, 30.0, 40.0])
        result = delay(x, 1)

        # First value should be NaN
        self.assertTrue(np.isnan(result[0]))

        # Rest should be shifted by 1
        expected = np.array([np.nan, 10.0, 20.0, 30.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_delay_zero(self):
        """Test delay with n=0 returns copy."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = delay(x, 0)

        # Should be equal to original
        np.testing.assert_array_almost_equal(result, x)

        # But should be a copy (different object)
        self.assertIsNot(result, x)

    def test_delay_nan_handling(self):
        """Test delay preserves NaN values at shifted positions."""
        x = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
        result = delay(x, 2)

        # First 2 should be NaN (insufficient data)
        self.assertTrue(np.all(np.isnan(result[:2])))

        # Position 2 should have value from original position 0
        self.assertEqual(result[2], 1.0)

        # Position 3 should have NaN (from original position 1)
        self.assertTrue(np.isnan(result[3]))

        # Position 4 should have value from original position 2
        self.assertEqual(result[4], 3.0)

    def test_delay_edge_cases_n_equals_length(self):
        """Test delay with n equal to array length."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = delay(x, 5)

        # All values should be NaN
        self.assertTrue(np.all(np.isnan(result)))

    def test_delay_edge_cases_n_greater_than_length(self):
        """Test delay with n greater than array length."""
        x = np.array([1.0, 2.0, 3.0])
        result = delay(x, 10)

        # All values should be NaN
        self.assertTrue(np.all(np.isnan(result)))

    def test_delay_negative_n(self):
        """Test delay with negative n raises ValueError."""
        x = np.array([1.0, 2.0, 3.0])

        with self.assertRaises(ValueError):
            delay(x, -1)

    def test_delay_empty_or_short(self):
        """Test delay with empty or short arrays."""
        # Empty array
        x_empty = np.array([], dtype=float)
        result = delay(x_empty, 1)
        self.assertEqual(len(result), 0)

        # Single element array, n=1 (all NaN)
        x_single = np.array([5.0])
        result = delay(x_single, 1)
        self.assertEqual(len(result), 1)
        self.assertTrue(np.isnan(result[0]))

        # Single element array, n=0 (returns copy)
        result = delay(x_single, 0)
        self.assertEqual(result[0], 5.0)


class TestDelta(unittest.TestCase):
    """Tests for delta function."""

    def test_delta_basic(self):
        """Test delta with known values."""
        x = np.array([1.0, 2.0, 4.0, 7.0, 11.0])
        result = delta(x, 1)

        # First value should be NaN
        self.assertTrue(np.isnan(result[0]))

        # Differences should be [nan, 1, 2, 3, 4]
        expected = np.array([np.nan, 1.0, 2.0, 3.0, 4.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_delta_n2(self):
        """Test delta with n=2."""
        x = np.array([10.0, 11.0, 13.0, 16.0, 20.0])
        result = delta(x, 2)

        # First 2 values should be NaN
        self.assertTrue(np.all(np.isnan(result[:2])))

        # Differences: [nan, nan, 3, 5, 7]
        expected = np.array([np.nan, np.nan, 3.0, 5.0, 7.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_delta_zero(self):
        """Test delta with n=0 returns all zeros."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = delta(x, 0)

        # Should be x - x = all zeros
        expected = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_delta_nan_handling(self):
        """Test delta handles NaN values correctly."""
        x = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
        result = delta(x, 1)

        # Position 0: NaN (no previous)
        self.assertTrue(np.isnan(result[0]))

        # Position 1: NaN - 1.0 = NaN
        self.assertTrue(np.isnan(result[1]))

        # Position 2: 3.0 - NaN = NaN
        self.assertTrue(np.isnan(result[2]))

        # Positions 3, 4: valid differences
        self.assertEqual(result[3], 1.0)  # 4.0 - 3.0
        self.assertEqual(result[4], 1.0)  # 5.0 - 4.0

    def test_delta_negative_n(self):
        """Test delta with negative n raises ValueError."""
        x = np.array([1.0, 2.0, 3.0])

        with self.assertRaises(ValueError):
            delta(x, -1)

    def test_delta_edge_cases(self):
        """Test delta edge cases."""
        # All same values
        x = np.array([5.0, 5.0, 5.0, 5.0])
        result = delta(x, 1)
        expected = np.array([np.nan, 0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_delta_empty_or_short(self):
        """Test delta with empty or short arrays."""
        # Empty array
        x_empty = np.array([], dtype=float)
        result = delta(x_empty, 1)
        self.assertEqual(len(result), 0)

        # Single element array, n=1 (all NaN)
        x_single = np.array([5.0])
        result = delta(x_single, 1)
        self.assertEqual(len(result), 1)
        self.assertTrue(np.isnan(result[0]))

        # Single element array, n=0 (returns 0)
        result = delta(x_single, 0)
        self.assertEqual(result[0], 0.0)


class TestRank(unittest.TestCase):
    """Tests for rank function."""

    def test_rank_basic(self):
        """Test rank with known values."""
        x = np.array([10.0, 5.0, 15.0, 5.0])
        result = rank(x)

        # rankdata([10, 5, 15, 5]) = [3, 1.5, 4, 1.5]
        # Normalized: (rank - 1) / (4 - 1)
        # [3, 1.5, 4, 1.5] -> [(3-1)/3, (1.5-1)/3, (4-1)/3, (1.5-1)/3]
        # = [2/3, 0.5/3, 3/3, 0.5/3] = [0.666..., 0.166..., 1.0, 0.166...]
        expected = np.array([2.0/3.0, 0.5/3.0, 1.0, 0.5/3.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_rank_ascending(self):
        """Test rank with ascending values."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = rank(x)

        # Should be [0.0, 0.25, 0.5, 0.75, 1.0]
        expected = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_rank_descending(self):
        """Test rank with descending values."""
        x = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        result = rank(x)

        # Should be [1.0, 0.75, 0.5, 0.25, 0.0]
        expected = np.array([1.0, 0.75, 0.5, 0.25, 0.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_rank_nan_handling(self):
        """Test rank handles NaN values correctly."""
        x = np.array([3.0, 1.0, 2.0, np.nan, 4.0])
        result = rank(x)

        # NaN should remain NaN
        self.assertTrue(np.isnan(result[3]))

        # Valid values: [3.0, 1.0, 2.0, 4.0]
        # rankdata([3.0, 1.0, 2.0, 4.0]) = [3, 1, 2, 4] (smallest gets rank 1)
        # Normalized: (rank - 1) / (4 - 1) = [2/3, 0, 1/3, 1]
        self.assertAlmostEqual(result[1], 0.0)      # 1.0 is smallest -> rank 1 -> 0
        self.assertAlmostEqual(result[2], 1.0/3.0)  # 2.0 -> rank 2 -> 1/3
        self.assertAlmostEqual(result[0], 2.0/3.0)  # 3.0 -> rank 3 -> 2/3
        self.assertAlmostEqual(result[4], 1.0)      # 4.0 is largest -> rank 4 -> 1

    def test_rank_all_nan(self):
        """Test rank with all NaN values."""
        x = np.array([np.nan, np.nan, np.nan])
        result = rank(x)

        # All should be NaN
        self.assertTrue(np.all(np.isnan(result)))

    def test_rank_single_value(self):
        """Test rank with single valid value."""
        x = np.array([5.0])
        result = rank(x)

        # Single value should have rank 0.5
        self.assertEqual(result[0], 0.5)

    def test_rank_single_with_nans(self):
        """Test rank with one valid value and rest NaN."""
        x = np.array([np.nan, 5.0, np.nan])
        result = rank(x)

        # Valid value should have rank 0.5
        self.assertEqual(result[1], 0.5)
        self.assertTrue(np.isnan(result[0]))
        self.assertTrue(np.isnan(result[2]))

    def test_rank_empty(self):
        """Test rank with empty array."""
        x = np.array([], dtype=float)
        result = rank(x)

        self.assertEqual(len(result), 0)

    def test_rank_two_values(self):
        """Test rank with two values."""
        x = np.array([10.0, 5.0])
        result = rank(x)

        # rankdata([10, 5]) = [2, 1]
        # Normalized: (rank - 1) / (2 - 1) = [1, 0]
        self.assertEqual(result[0], 1.0)
        self.assertEqual(result[1], 0.0)


class TestSign(unittest.TestCase):
    """Tests for sign function."""

    def test_sign_basic(self):
        """Test sign with known values."""
        x = np.array([1.5, -2.0, 0.0, 3.0, -4.5])
        result = sign(x)

        expected = np.array([1.0, -1.0, 0.0, 1.0, -1.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_sign_positive(self):
        """Test sign with all positive values."""
        x = np.array([1.0, 2.0, 0.5, 100.0])
        result = sign(x)

        expected = np.array([1.0, 1.0, 1.0, 1.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_sign_negative(self):
        """Test sign with all negative values."""
        x = np.array([-1.0, -2.0, -0.5, -100.0])
        result = sign(x)

        expected = np.array([-1.0, -1.0, -1.0, -1.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_sign_zeros(self):
        """Test sign with zeros."""
        x = np.array([0.0, 0.0, 0.0])
        result = sign(x)

        expected = np.array([0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_sign_nan_handling(self):
        """Test sign preserves NaN values."""
        x = np.array([1.0, np.nan, -3.0, np.nan, 0.0])
        result = sign(x)

        self.assertEqual(result[0], 1.0)
        self.assertTrue(np.isnan(result[1]))
        self.assertEqual(result[2], -1.0)
        self.assertTrue(np.isnan(result[3]))
        self.assertEqual(result[4], 0.0)

    def test_sign_all_nan(self):
        """Test sign with all NaN values."""
        x = np.array([np.nan, np.nan, np.nan])
        result = sign(x)

        self.assertTrue(np.all(np.isnan(result)))

    def test_sign_empty(self):
        """Test sign with empty array."""
        x = np.array([], dtype=float)
        result = sign(x)

        self.assertEqual(len(result), 0)

    def test_sign_edge_cases(self):
        """Test sign with edge cases."""
        # Very small positive
        x = np.array([1e-10])
        self.assertEqual(sign(x)[0], 1.0)

        # Very small negative
        x = np.array([-1e-10])
        self.assertEqual(sign(x)[0], -1.0)


class TestTsSum(unittest.TestCase):
    """Tests for ts_sum function."""

    def test_ts_sum_basic(self):
        """Test ts_sum with known values."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = ts_sum(x, 3)

        # First 2 values should be NaN
        self.assertTrue(np.all(np.isnan(result[:2])))

        # Rolling sums: [nan, nan, 1+2+3, 2+3+4, 3+4+5] = [nan, nan, 6, 9, 12]
        expected = np.array([np.nan, np.nan, 6.0, 9.0, 12.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_ts_sum_n1(self):
        """Test ts_sum with n=1."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = ts_sum(x, 1)

        # Should be same as input
        np.testing.assert_array_almost_equal(result, x)

    def test_ts_sum_nan_handling(self):
        """Test ts_sum handles NaN values correctly."""
        x = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
        result = ts_sum(x, 3)

        # First 2 values should be NaN
        self.assertTrue(np.all(np.isnan(result[:2])))

        # Position 2: 1.0 + NaN + 3.0 = 4.0 (NaN excluded)
        self.assertEqual(result[2], 4.0)

        # Position 3: NaN + 3.0 + 4.0 = 7.0 (NaN excluded)
        self.assertEqual(result[3], 7.0)

        # Position 4: 3.0 + 4.0 + 5.0 = 12.0
        self.assertEqual(result[4], 12.0)

    def test_ts_sum_all_nan_window(self):
        """Test ts_sum when entire window is NaN."""
        x = np.array([1.0, 2.0, np.nan, np.nan, np.nan, 3.0])
        result = ts_sum(x, 3)

        # Position 4 (0-indexed): window is [NaN, NaN, NaN] -> NaN
        self.assertTrue(np.isnan(result[4]))

    def test_ts_sum_edge_cases(self):
        """Test ts_sum edge cases."""
        # Window equals array length
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = ts_sum(x, 5)

        # Only last value should be non-NaN (sum of all)
        self.assertTrue(np.all(np.isnan(result[:-1])))
        self.assertEqual(result[-1], 15.0)

    def test_ts_sum_empty_or_short(self):
        """Test ts_sum with empty or short arrays."""
        # Empty array
        x_empty = np.array([], dtype=float)
        result = ts_sum(x_empty, 3)
        self.assertEqual(len(result), 0)

        # Short array (window larger than array)
        x_short = np.array([1.0, 2.0])
        result = ts_sum(x_short, 3)
        self.assertTrue(np.all(np.isnan(result)))

    def test_ts_sum_negative_n(self):
        """Test ts_sum with negative n raises ValueError."""
        x = np.array([1.0, 2.0, 3.0])
        with self.assertRaises(ValueError):
            ts_sum(x, -1)

    def test_ts_sum_zero_n(self):
        """Test ts_sum with n=0 raises ValueError."""
        x = np.array([1.0, 2.0, 3.0])
        with self.assertRaises(ValueError):
            ts_sum(x, 0)


class TestTsMean(unittest.TestCase):
    """Tests for ts_mean function."""

    def test_ts_mean_basic(self):
        """Test ts_mean with known values."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = ts_mean(x, 3)

        # First 2 values should be NaN
        self.assertTrue(np.all(np.isnan(result[:2])))

        # Rolling means: [nan, nan, 6/3, 9/3, 12/3] = [nan, nan, 2, 3, 4]
        expected = np.array([np.nan, np.nan, 2.0, 3.0, 4.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_ts_mean_n1(self):
        """Test ts_mean with n=1."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = ts_mean(x, 1)

        # Should be same as input
        np.testing.assert_array_almost_equal(result, x)

    def test_ts_mean_nan_handling(self):
        """Test ts_mean handles NaN values correctly."""
        x = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
        result = ts_mean(x, 3)

        # First 2 values should be NaN
        self.assertTrue(np.all(np.isnan(result[:2])))

        # Position 2: (1.0 + 3.0) / 2 = 2.0 (NaN excluded)
        self.assertEqual(result[2], 2.0)

        # Position 3: (3.0 + 4.0) / 2 = 3.5 (NaN excluded)
        self.assertEqual(result[3], 3.5)

        # Position 4: (3.0 + 4.0 + 5.0) / 3 = 4.0
        self.assertAlmostEqual(result[4], 4.0)

    def test_ts_mean_all_nan_window(self):
        """Test ts_mean when entire window is NaN."""
        x = np.array([1.0, 2.0, np.nan, np.nan, np.nan, 3.0])
        result = ts_mean(x, 3)

        # Position 4 (0-indexed): window is [NaN, NaN, NaN] -> NaN
        self.assertTrue(np.isnan(result[4]))

    def test_ts_mean_edge_cases(self):
        """Test ts_mean edge cases."""
        # Constant values
        x = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        result = ts_mean(x, 3)

        expected = np.array([np.nan, np.nan, 5.0, 5.0, 5.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_ts_mean_empty_or_short(self):
        """Test ts_mean with empty or short arrays."""
        # Empty array
        x_empty = np.array([], dtype=float)
        result = ts_mean(x_empty, 3)
        self.assertEqual(len(result), 0)

        # Short array (window larger than array)
        x_short = np.array([1.0, 2.0])
        result = ts_mean(x_short, 3)
        self.assertTrue(np.all(np.isnan(result)))

    def test_ts_mean_negative_n(self):
        """Test ts_mean with negative n raises ValueError."""
        x = np.array([1.0, 2.0, 3.0])
        with self.assertRaises(ValueError):
            ts_mean(x, -1)


class TestTsStd(unittest.TestCase):
    """Tests for ts_std function."""

    def test_ts_std_basic(self):
        """Test ts_std with known values."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = ts_std(x, 3)

        # First 2 values should be NaN
        self.assertTrue(np.all(np.isnan(result[:2])))

        # Window [1,2,3]: std = 1.0, window [2,3,4]: std = 1.0, window [3,4,5]: std = 1.0
        expected = np.array([np.nan, np.nan, 1.0, 1.0, 1.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_ts_std_sample_vs_population(self):
        """Test ts_std with different ddof values."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # Sample std (ddof=1, default)
        result_sample = ts_std(x, 3, ddof=1)

        # Population std (ddof=0)
        result_pop = ts_std(x, 3, ddof=0)

        # Population std should be smaller than sample std
        self.assertTrue(np.all(result_pop[2:] < result_sample[2:]))

    def test_ts_std_nan_handling(self):
        """Test ts_std handles NaN values correctly."""
        x = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
        result = ts_std(x, 3)

        # First 2 values should be NaN
        self.assertTrue(np.all(np.isnan(result[:2])))

        # Position 2: only 2 valid values [1.0, 3.0], std with ddof=1 possible
        self.assertFalse(np.isnan(result[2]))

        # Position 3: only 2 valid values [3.0, 4.0]
        self.assertFalse(np.isnan(result[3]))

    def test_ts_std_insufficient_data(self):
        """Test ts_std with insufficient valid data."""
        x = np.array([1.0, np.nan, np.nan, 4.0, 5.0])
        result = ts_std(x, 3)

        # Position 1: only 1 valid value -> NaN (need 2 for ddof=1)
        self.assertTrue(np.isnan(result[1]))

        # Position 2: only 1 valid value -> NaN
        self.assertTrue(np.isnan(result[2]))

    def test_ts_std_single_value_window(self):
        """Test ts_std with n=1."""
        x = np.array([1.0, 2.0, 3.0])
        result = ts_std(x, 1)

        # With n=1 and ddof=1, we have 0 degrees of freedom -> NaN
        self.assertTrue(np.all(np.isnan(result)))

        # With ddof=0, single value has std of 0
        result_ddof0 = ts_std(x, 1, ddof=0)
        np.testing.assert_array_almost_equal(result_ddof0, np.array([0.0, 0.0, 0.0]))

    def test_ts_std_edge_cases(self):
        """Test ts_std edge cases."""
        # Constant values - std should be 0
        x = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        result = ts_std(x, 3)

        expected = np.array([np.nan, np.nan, 0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_ts_std_empty_or_short(self):
        """Test ts_std with empty or short arrays."""
        # Empty array
        x_empty = np.array([], dtype=float)
        result = ts_std(x_empty, 3)
        self.assertEqual(len(result), 0)

        # Short array (window larger than array)
        x_short = np.array([1.0, 2.0])
        result = ts_std(x_short, 3)
        self.assertTrue(np.all(np.isnan(result)))


class TestTsMin(unittest.TestCase):
    """Tests for ts_min function."""

    def test_ts_min_basic(self):
        """Test ts_min with known values."""
        x = np.array([5.0, 2.0, 8.0, 1.0, 9.0])
        result = ts_min(x, 3)

        # First 2 values should be NaN
        self.assertTrue(np.all(np.isnan(result[:2])))

        # Rolling mins: [nan, nan, min(5,2,8), min(2,8,1), min(8,1,9)]
        expected = np.array([np.nan, np.nan, 2.0, 1.0, 1.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_ts_min_nan_handling(self):
        """Test ts_min handles NaN values correctly."""
        x = np.array([5.0, np.nan, 8.0, 1.0, 9.0])
        result = ts_min(x, 3)

        # First 2 values should be NaN
        self.assertTrue(np.all(np.isnan(result[:2])))

        # Position 2: min(5.0, 8.0) = 5.0 (NaN excluded)
        self.assertEqual(result[2], 5.0)

        # Position 3: min(8.0, 1.0) = 1.0 (NaN excluded)
        self.assertEqual(result[3], 1.0)

        # Position 4: min(8.0, 1.0, 9.0) = 1.0
        self.assertEqual(result[4], 1.0)

    def test_ts_min_all_nan_window(self):
        """Test ts_min when entire window is NaN."""
        x = np.array([1.0, 2.0, np.nan, np.nan, np.nan, 3.0])
        result = ts_min(x, 3)

        # Position 4 (0-indexed): window is [NaN, NaN, NaN] -> NaN
        self.assertTrue(np.isnan(result[4]))

    def test_ts_min_edge_cases(self):
        """Test ts_min edge cases."""
        # Descending values
        x = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        result = ts_min(x, 3)

        expected = np.array([np.nan, np.nan, 3.0, 2.0, 1.0])
        np.testing.assert_array_almost_equal(result, expected)

        # With zeros
        x = np.array([5.0, 0.0, 3.0, 0.0, 1.0])
        result = ts_min(x, 3)

        expected = np.array([np.nan, np.nan, 0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_ts_min_empty_or_short(self):
        """Test ts_min with empty or short arrays."""
        # Empty array
        x_empty = np.array([], dtype=float)
        result = ts_min(x_empty, 3)
        self.assertEqual(len(result), 0)

        # Short array (window larger than array)
        x_short = np.array([1.0, 2.0])
        result = ts_min(x_short, 3)
        self.assertTrue(np.all(np.isnan(result)))


class TestTsMax(unittest.TestCase):
    """Tests for ts_max function."""

    def test_ts_max_basic(self):
        """Test ts_max with known values."""
        x = np.array([5.0, 2.0, 8.0, 1.0, 9.0])
        result = ts_max(x, 3)

        # First 2 values should be NaN
        self.assertTrue(np.all(np.isnan(result[:2])))

        # Rolling maxs: [nan, nan, max(5,2,8), max(2,8,1), max(8,1,9)]
        expected = np.array([np.nan, np.nan, 8.0, 8.0, 9.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_ts_max_nan_handling(self):
        """Test ts_max handles NaN values correctly."""
        x = np.array([5.0, np.nan, 8.0, 1.0, 9.0])
        result = ts_max(x, 3)

        # First 2 values should be NaN
        self.assertTrue(np.all(np.isnan(result[:2])))

        # Position 2: max(5.0, 8.0) = 8.0 (NaN excluded)
        self.assertEqual(result[2], 8.0)

        # Position 3: max(8.0, 1.0) = 8.0 (NaN excluded)
        self.assertEqual(result[3], 8.0)

        # Position 4: max(8.0, 1.0, 9.0) = 9.0
        self.assertEqual(result[4], 9.0)

    def test_ts_max_all_nan_window(self):
        """Test ts_max when entire window is NaN."""
        x = np.array([1.0, 2.0, np.nan, np.nan, np.nan, 3.0])
        result = ts_max(x, 3)

        # Position 4 (0-indexed): window is [NaN, NaN, NaN] -> NaN
        self.assertTrue(np.isnan(result[4]))

    def test_ts_max_edge_cases(self):
        """Test ts_max edge cases."""
        # Ascending values
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = ts_max(x, 3)

        expected = np.array([np.nan, np.nan, 3.0, 4.0, 5.0])
        np.testing.assert_array_almost_equal(result, expected)

        # With negative values
        x = np.array([-5.0, -2.0, -8.0, -1.0, -9.0])
        result = ts_max(x, 3)

        expected = np.array([np.nan, np.nan, -2.0, -1.0, -1.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_ts_max_empty_or_short(self):
        """Test ts_max with empty or short arrays."""
        # Empty array
        x_empty = np.array([], dtype=float)
        result = ts_max(x_empty, 3)
        self.assertEqual(len(result), 0)

        # Short array (window larger than array)
        x_short = np.array([1.0, 2.0])
        result = ts_max(x_short, 3)
        self.assertTrue(np.all(np.isnan(result)))


class TestTsCount(unittest.TestCase):
    """Tests for ts_count function."""

    def test_ts_count_basic(self):
        """Test ts_count with known values."""
        condition = np.array([True, False, True, True, False])
        result = ts_count(condition, 3)

        # First 2 values should be NaN
        self.assertTrue(np.all(np.isnan(result[:2])))

        # Rolling counts: [nan, nan, count(T,F,T), count(F,T,T), count(T,T,F)]
        # = [nan, nan, 2, 2, 2]
        expected = np.array([np.nan, np.nan, 2.0, 2.0, 2.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_ts_count_numerical_values(self):
        """Test ts_count with numerical values (non-zero is True)."""
        x = np.array([1.0, 0.0, 3.0, 0.0, 2.0])
        result = ts_count(x, 3)

        # First 2 values should be NaN
        self.assertTrue(np.all(np.isnan(result[:2])))

        # Rolling counts: [nan, nan, 2, 1, 2]
        expected = np.array([np.nan, np.nan, 2.0, 1.0, 2.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_ts_count_nan_handling(self):
        """Test ts_count handles NaN values correctly (treated as False)."""
        x = np.array([1.0, np.nan, 3.0, 0.0, 2.0])
        result = ts_count(x, 3)

        # First 2 values should be NaN
        self.assertTrue(np.all(np.isnan(result[:2])))

        # Position 2: count non-zero in [1.0, NaN, 3.0] -> 2 (NaN is False)
        self.assertEqual(result[2], 2.0)

        # Position 3: count non-zero in [NaN, 3.0, 0.0] -> 1 (NaN is False)
        self.assertEqual(result[3], 1.0)

        # Position 4: count non-zero in [3.0, 0.0, 2.0] -> 2
        self.assertEqual(result[4], 2.0)

    def test_ts_count_all_nan_window(self):
        """Test ts_count when entire window is NaN."""
        x = np.array([1.0, 2.0, np.nan, np.nan, np.nan, 3.0])
        result = ts_count(x, 3)

        # Position 4 (0-indexed): window is [NaN, NaN, NaN] -> NaN
        self.assertTrue(np.isnan(result[4]))

    def test_ts_count_all_false(self):
        """Test ts_count when all values in window are False/zero."""
        x = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        result = ts_count(x, 3)

        # All counts should be 0 (not NaN, since valid values exist)
        expected = np.array([np.nan, np.nan, 0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_ts_count_empty_or_short(self):
        """Test ts_count with empty or short arrays."""
        # Empty array
        x_empty = np.array([], dtype=float)
        result = ts_count(x_empty, 3)
        self.assertEqual(len(result), 0)

        # Short array (window larger than array)
        x_short = np.array([1.0, 0.0])
        result = ts_count(x_short, 3)
        self.assertTrue(np.all(np.isnan(result)))


class TestTsProd(unittest.TestCase):
    """Tests for ts_prod function."""

    def test_ts_prod_basic(self):
        """Test ts_prod with known values."""
        x = np.array([2.0, 3.0, 4.0, 5.0])
        result = ts_prod(x, 3)

        # First 2 values should be NaN
        self.assertTrue(np.all(np.isnan(result[:2])))

        # Rolling products: [nan, nan, 2*3*4, 3*4*5] = [nan, nan, 24, 60]
        expected = np.array([np.nan, np.nan, 24.0, 60.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_ts_prod_with_ones(self):
        """Test ts_prod with ones."""
        x = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        result = ts_prod(x, 3)

        expected = np.array([np.nan, np.nan, 1.0, 1.0, 1.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_ts_prod_with_zero(self):
        """Test ts_prod with zero in window."""
        x = np.array([2.0, 0.0, 4.0, 5.0, 3.0])
        result = ts_prod(x, 3)

        # Position 2: 2*0*4 = 0
        self.assertEqual(result[2], 0.0)

        # Position 3: 0*4*5 = 0
        self.assertEqual(result[3], 0.0)

        # Position 4: 4*5*3 = 60
        self.assertEqual(result[4], 60.0)

    def test_ts_prod_nan_handling(self):
        """Test ts_prod handles NaN values correctly."""
        x = np.array([2.0, np.nan, 4.0, 5.0, 3.0])
        result = ts_prod(x, 3)

        # First 2 values should be NaN
        self.assertTrue(np.all(np.isnan(result[:2])))

        # Position 2: 2.0 * 4.0 = 8.0 (NaN excluded)
        self.assertEqual(result[2], 8.0)

        # Position 3: 4.0 * 5.0 = 20.0 (NaN excluded)
        self.assertEqual(result[3], 20.0)

        # Position 4: 4.0 * 5.0 * 3.0 = 60.0
        self.assertEqual(result[4], 60.0)

    def test_ts_prod_all_nan_window(self):
        """Test ts_prod when entire window is NaN."""
        x = np.array([1.0, 2.0, np.nan, np.nan, np.nan, 3.0])
        result = ts_prod(x, 3)

        # Position 4 (0-indexed): window is [NaN, NaN, NaN] -> NaN
        self.assertTrue(np.isnan(result[4]))

    def test_ts_prod_edge_cases(self):
        """Test ts_prod edge cases."""
        # Negative values
        x = np.array([-2.0, 3.0, -4.0, 5.0])
        result = ts_prod(x, 3)

        # Position 2: -2 * 3 * -4 = 24
        self.assertEqual(result[2], 24.0)

        # Position 3: 3 * -4 * 5 = -60
        self.assertEqual(result[3], -60.0)

    def test_ts_prod_empty_or_short(self):
        """Test ts_prod with empty or short arrays."""
        # Empty array
        x_empty = np.array([], dtype=float)
        result = ts_prod(x_empty, 3)
        self.assertEqual(len(result), 0)

        # Short array (window larger than array)
        x_short = np.array([1.0, 2.0])
        result = ts_prod(x_short, 3)
        self.assertTrue(np.all(np.isnan(result)))


class TestCovariance(unittest.TestCase):
    """Tests for covariance function."""

    def test_covariance_basic(self):
        """Test covariance with known values."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        result = covariance(x, y, 3)

        # First 2 values should be NaN
        self.assertTrue(np.all(np.isnan(result[:2])))

        # With perfect linear relationship y = 2*x, covariance should be constant
        # cov([1,2,3], [2,4,6]) = ((1-2)*(2-4) + (2-2)*(4-4) + (3-2)*(6-4)) / 2 = (2 + 0 + 2) / 2 = 2
        expected = np.array([np.nan, np.nan, 2.0, 2.0, 2.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_covariance_perfect_negative(self):
        """Test covariance with perfect negative correlation."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        result = covariance(x, y, 3)

        # First 2 values should be NaN
        self.assertTrue(np.all(np.isnan(result[:2])))

        # Negative covariance
        self.assertTrue(np.all(result[2:] < 0))

    def test_covariance_no_correlation(self):
        """Test covariance with no correlation (constant y)."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        result = covariance(x, y, 3)

        # First 2 values should be NaN
        self.assertTrue(np.all(np.isnan(result[:2])))

        # Covariance with constant should be 0
        np.testing.assert_array_almost_equal(result[2:], np.array([0.0, 0.0, 0.0]))

    def test_covariance_nan_handling(self):
        """Test covariance handles NaN values correctly."""
        # Use window size 4 to ensure enough valid pairs with NaN
        x = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0])
        y = np.array([2.0, 4.0, 6.0, 8.0, 10.0, 12.0])
        result = covariance(x, y, 4)

        # First 3 values should be NaN
        self.assertTrue(np.all(np.isnan(result[:3])))

        # Position 3: window [1, 2, NaN, 4] and [2, 4, 6, 8] -> valid pairs: (1,2), (2,4), (4,8)
        # mean_x = (1+2+4)/3 = 7/3, mean_y = (2+4+8)/3 = 14/3
        # cov = ((1-7/3)*(2-14/3) + (2-7/3)*(4-14/3) + (4-7/3)*(8-14/3)) / 2
        #     = ((-4/3)*(-8/3) + (-1/3)*(-2/3) + (5/3)*(10/3)) / 2
        #     = (32/9 + 2/9 + 50/9) / 2 = 84/9 / 2 = 42/9 = 14/3 ≈ 4.667
        self.assertAlmostEqual(result[3], 14.0/3.0, places=5)

    def test_covariance_both_nan(self):
        """Test covariance when both arrays have NaN at same positions."""
        # Use window size 4 to ensure enough valid pairs with NaN
        x = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0])
        y = np.array([2.0, 4.0, np.nan, 8.0, 10.0, 12.0])
        result = covariance(x, y, 4)

        # First 3 values should be NaN
        self.assertTrue(np.all(np.isnan(result[:3])))

        # Position 3: window [1, 2, NaN, 4] and [2, 4, NaN, 8] -> valid pairs: (1,2), (2,4), (4,8)
        # Same calculation as above, should equal 14/3
        self.assertAlmostEqual(result[3], 14.0/3.0, places=5)

    def test_covariance_edge_cases(self):
        """Test covariance edge cases."""
        # Window equals array length
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        result = covariance(x, y, 5)

        # Only last value should be non-NaN
        self.assertTrue(np.all(np.isnan(result[:-1])))
        self.assertFalse(np.isnan(result[-1]))

    def test_covariance_empty_or_short(self):
        """Test covariance with empty or short arrays."""
        # Empty array
        x_empty = np.array([], dtype=float)
        y_empty = np.array([], dtype=float)
        result = covariance(x_empty, y_empty, 3)
        self.assertEqual(len(result), 0)

        # Short array (window larger than array)
        x_short = np.array([1.0, 2.0])
        y_short = np.array([2.0, 4.0])
        result = covariance(x_short, y_short, 3)
        self.assertTrue(np.all(np.isnan(result)))

        # Single element arrays
        x_single = np.array([1.0])
        y_single = np.array([2.0])
        result = covariance(x_single, y_single, 1)
        # With n=1 and ddof=1, need 3 valid pairs but only have 1 -> NaN
        self.assertTrue(np.isnan(result[0]))

    def test_covariance_different_lengths(self):
        """Test covariance with different length arrays raises ValueError."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0])

        with self.assertRaises(ValueError):
            covariance(x, y, 2)

    def test_covariance_negative_n(self):
        """Test covariance with negative n raises ValueError."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0, 3.0])

        with self.assertRaises(ValueError):
            covariance(x, y, -1)

    def test_covariance_ddof(self):
        """Test covariance with different ddof values."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])

        # Sample covariance (ddof=1, default)
        result_sample = covariance(x, y, 3, ddof=1)

        # Population covariance (ddof=0)
        result_pop = covariance(x, y, 3, ddof=0)

        # Population covariance should be smaller than sample covariance
        self.assertTrue(np.all(result_pop[2:] < result_sample[2:]))


class TestRegressionBeta(unittest.TestCase):
    """Tests for regression_beta function."""

    def test_regression_beta_basic(self):
        """Test regression_beta with known values."""
        x = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = regression_beta(x, y, 3)

        # First 2 values should be NaN
        self.assertTrue(np.all(np.isnan(result[:2])))

        # x = 2*y, so beta should be 2
        expected = np.array([np.nan, np.nan, 2.0, 2.0, 2.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_regression_beta_inverse(self):
        """Test regression_beta with inverse relationship."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        result = regression_beta(x, y, 3)

        # First 2 values should be NaN
        self.assertTrue(np.all(np.isnan(result[:2])))

        # y = 2*x, so when regressing x on y: x = 0.5*y, beta = 0.5
        expected = np.array([np.nan, np.nan, 0.5, 0.5, 0.5])
        np.testing.assert_array_almost_equal(result, expected)

    def test_regression_beta_nan_handling(self):
        """Test regression_beta handles NaN values correctly."""
        x = np.array([2.0, np.nan, 6.0, 8.0, 10.0])
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = regression_beta(x, y, 3)

        # First 2 values should be NaN
        self.assertTrue(np.all(np.isnan(result[:2])))

        # Position 2: window [2, NaN, 6] and [1, 2, 3] -> valid pairs: (2,1) and (6,3)
        # beta = cov(x,y) / var(y) = ((2-4)*(1-2) + (6-4)*(3-2)) / ((1-2)^2 + (3-2)^2) = (2 + 2) / (1 + 1) = 2
        self.assertAlmostEqual(result[2], 2.0)

    def test_regression_beta_zero_variance(self):
        """Test regression_beta with zero variance in y."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([5.0, 5.0, 5.0, 5.0, 5.0])  # constant
        result = regression_beta(x, y, 3)

        # First 2 values should be NaN
        self.assertTrue(np.all(np.isnan(result[:2])))

        # Should be NaN when var(y) = 0 (division by zero)
        self.assertTrue(np.all(np.isnan(result[2:])))

    def test_regression_beta_edge_cases(self):
        """Test regression_beta edge cases."""
        # Negative relationship
        x = np.array([10.0, 8.0, 6.0, 4.0, 2.0])
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = regression_beta(x, y, 3)

        # First 2 values should be NaN
        self.assertTrue(np.all(np.isnan(result[:2])))

        # Negative beta
        self.assertTrue(np.all(result[2:] < 0))

    def test_regression_beta_empty_or_short(self):
        """Test regression_beta with empty or short arrays."""
        # Empty array
        x_empty = np.array([], dtype=float)
        y_empty = np.array([], dtype=float)
        result = regression_beta(x_empty, y_empty, 3)
        self.assertEqual(len(result), 0)

        # Short array (window larger than array)
        x_short = np.array([1.0, 2.0])
        y_short = np.array([2.0, 4.0])
        result = regression_beta(x_short, y_short, 3)
        self.assertTrue(np.all(np.isnan(result)))

        # Single element arrays with n=1 (need at least 2 valid pairs)
        x_single = np.array([1.0])
        y_single = np.array([2.0])
        result = regression_beta(x_single, y_single, 1)
        self.assertTrue(np.isnan(result[0]))

    def test_regression_beta_different_lengths(self):
        """Test regression_beta with different length arrays raises ValueError."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0])

        with self.assertRaises(ValueError):
            regression_beta(x, y, 2)

    def test_regression_beta_negative_n(self):
        """Test regression_beta with negative n raises ValueError."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0, 3.0])

        with self.assertRaises(ValueError):
            regression_beta(x, y, -1)


class TestRegressionResidual(unittest.TestCase):
    """Tests for regression_residual function."""

    def test_regression_residual_basic(self):
        """Test regression_residual with known values."""
        x = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = regression_residual(x, y, 3)

        # First 2 values should be NaN
        self.assertTrue(np.all(np.isnan(result[:2])))

        # x = 2*y exactly, so residuals should be approximately 0
        np.testing.assert_array_almost_equal(result[2:], np.array([0.0, 0.0, 0.0]))

    def test_regression_residual_with_noise(self):
        """Test regression_residual with noisy data."""
        # x = 2*y + noise
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        x = np.array([2.1, 3.9, 6.2, 7.8, 10.1])  # approximately 2*y
        result = regression_residual(x, y, 3)

        # First 2 values should be NaN
        self.assertTrue(np.all(np.isnan(result[:2])))

        # Residuals should be small but non-zero
        self.assertTrue(np.all(np.abs(result[2:]) < 0.2))

    def test_regression_residual_nan_handling(self):
        """Test regression_residual handles NaN values correctly."""
        x = np.array([2.0, np.nan, 6.0, 8.0, 10.0])
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = regression_residual(x, y, 3)

        # First 2 values should be NaN (position 1 has NaN in x)
        self.assertTrue(np.all(np.isnan(result[:2])))

        # Position 3: valid x and y, compute residual
        # Window: x=[NaN, 6, 8], y=[2, 3, 4] -> valid: (6,3) and (8,4)
        # beta = 2, alpha = 0, residual = 8 - (0 + 2*4) = 0
        self.assertAlmostEqual(result[3], 0.0, places=5)

    def test_regression_residual_current_nan(self):
        """Test regression_residual when current value is NaN."""
        x = np.array([2.0, 4.0, np.nan, 8.0, 10.0])
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = regression_residual(x, y, 3)

        # Position 2 should be NaN because x[2] is NaN
        self.assertTrue(np.isnan(result[2]))

        # Other positions should be valid
        self.assertFalse(np.isnan(result[3]))
        self.assertFalse(np.isnan(result[4]))

    def test_regression_residual_zero_variance(self):
        """Test regression_residual with zero variance in y."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([5.0, 5.0, 5.0, 5.0, 5.0])  # constant
        result = regression_residual(x, y, 3)

        # First 2 values should be NaN
        self.assertTrue(np.all(np.isnan(result[:2])))

        # Should be NaN when var(y) = 0
        self.assertTrue(np.all(np.isnan(result[2:])))

    def test_regression_residual_edge_cases(self):
        """Test regression_residual edge cases."""
        # Perfect fit with offset
        # x = 1 + 2*y
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        x = np.array([3.0, 5.0, 7.0, 9.0, 11.0])  # x = 1 + 2*y
        result = regression_residual(x, y, 5)

        # Last value should have residual ~0
        np.testing.assert_almost_equal(result[-1], 0.0, decimal=5)

    def test_regression_residual_empty_or_short(self):
        """Test regression_residual with empty or short arrays."""
        # Empty array
        x_empty = np.array([], dtype=float)
        y_empty = np.array([], dtype=float)
        result = regression_residual(x_empty, y_empty, 3)
        self.assertEqual(len(result), 0)

        # Short array (window larger than array)
        x_short = np.array([1.0, 2.0])
        y_short = np.array([2.0, 4.0])
        result = regression_residual(x_short, y_short, 3)
        self.assertTrue(np.all(np.isnan(result)))

        # Single element arrays with n=1 (need at least 2 valid pairs)
        x_single = np.array([1.0])
        y_single = np.array([2.0])
        result = regression_residual(x_single, y_single, 1)
        self.assertTrue(np.isnan(result[0]))

    def test_regression_residual_different_lengths(self):
        """Test regression_residual with different length arrays raises ValueError."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0])

        with self.assertRaises(ValueError):
            regression_residual(x, y, 2)

    def test_regression_residual_negative_n(self):
        """Test regression_residual with negative n raises ValueError."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0, 3.0])

        with self.assertRaises(ValueError):
            regression_residual(x, y, -1)


class TestSma(unittest.TestCase):
    """Tests for sma (Special Moving Average) function."""

    def test_sma_basic(self):
        """Test sma with known values."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = sma(x, n=3, m=1)

        # Y[0] = A[0] = 1
        self.assertEqual(result[0], 1.0)

        # Y[1] = (1*2 + (3-1)*1) / 3 = (2 + 2) / 3 = 4/3 ≈ 1.333
        expected_y1 = (1 * 2.0 + 2 * 1.0) / 3.0
        self.assertAlmostEqual(result[1], expected_y1, places=5)

        # Y[2] = (1*3 + (3-1)*4/3) / 3 = (3 + 8/3) / 3 = 17/9 ≈ 1.889
        expected_y2 = (1 * 3.0 + 2 * expected_y1) / 3.0
        self.assertAlmostEqual(result[2], expected_y2, places=5)

    def test_sma_example_from_doc(self):
        """Test sma with example from docstring."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = sma(x, n=3, m=1)

        # Expected from docstring: array([1., 1.33, 1.89, 2.59, 3.40])
        self.assertAlmostEqual(result[0], 1.0, places=2)
        self.assertAlmostEqual(result[1], 1.33, places=2)
        self.assertAlmostEqual(result[2], 1.89, places=2)
        self.assertAlmostEqual(result[3], 2.59, places=2)
        self.assertAlmostEqual(result[4], 3.40, places=2)

    def test_sma_nan_handling(self):
        """Test sma handles NaN values correctly (NaN propagates forward)."""
        x = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        result = sma(x, n=3, m=1)

        # Y[0] = 1.0
        self.assertEqual(result[0], 1.0)

        # Y[1] = (1*2 + 2*1) / 3 = 4/3
        self.assertAlmostEqual(result[1], 4.0/3.0, places=5)

        # Y[2] = NaN because x[2] is NaN
        self.assertTrue(np.isnan(result[2]))

        # Y[3] = NaN because Y[2] is NaN (NaN propagates)
        self.assertTrue(np.isnan(result[3]))

        # Y[4] = NaN because Y[3] is NaN
        self.assertTrue(np.isnan(result[4]))

    def test_sma_nan_at_start(self):
        """Test sma with NaN at first position."""
        x = np.array([np.nan, 2.0, 3.0, 4.0, 5.0])
        result = sma(x, n=3, m=1)

        # All values should be NaN because Y[0] = NaN
        self.assertTrue(np.all(np.isnan(result)))

    def test_sma_edge_cases(self):
        """Test sma edge cases."""
        # Equal weights (m = n)
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = sma(x, n=2, m=2)

        # Y[0] = 1.0
        self.assertEqual(result[0], 1.0)

        # Y[t] = (2*x[t] + 0*Y[t-1]) / 2 = x[t]
        np.testing.assert_array_almost_equal(result, x)

    def test_sma_empty_or_short(self):
        """Test sma with empty or short arrays."""
        # Empty array
        x_empty = np.array([], dtype=float)
        result = sma(x_empty, n=3, m=1)
        self.assertEqual(len(result), 0)

        # Single element array
        x_single = np.array([5.0])
        result = sma(x_single, n=3, m=1)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], 5.0)

        # Single element with NaN
        x_nan = np.array([np.nan])
        result = sma(x_nan, n=3, m=1)
        self.assertEqual(len(result), 1)
        self.assertTrue(np.isnan(result[0]))

    def test_sma_invalid_n(self):
        """Test sma with invalid n raises ValueError."""
        x = np.array([1.0, 2.0, 3.0])

        with self.assertRaises(ValueError):
            sma(x, n=0, m=1)

        with self.assertRaises(ValueError):
            sma(x, n=-1, m=1)

    def test_sma_invalid_m(self):
        """Test sma with invalid m raises ValueError."""
        x = np.array([1.0, 2.0, 3.0])

        with self.assertRaises(ValueError):
            sma(x, n=3, m=0)

        with self.assertRaises(ValueError):
            sma(x, n=3, m=-1)

        with self.assertRaises(ValueError):
            sma(x, n=3, m=4)  # m > n


class TestWma(unittest.TestCase):
    """Tests for wma (Weighted Moving Average) function."""

    def test_wma_basic(self):
        """Test wma with known values."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = wma(x, n=3)

        # First 2 values should be NaN
        self.assertTrue(np.all(np.isnan(result[:2])))

        # weights = [0.81, 0.9, 1.0], sum = 2.71
        # Position 2: (1.0*0.81 + 2.0*0.9 + 3.0*1.0) / 2.71 = 5.61/2.71 ≈ 2.070
        weight_sum = 0.81 + 0.9 + 1.0
        expected_2 = (1.0*0.81 + 2.0*0.9 + 3.0*1.0) / weight_sum
        self.assertAlmostEqual(result[2], expected_2, places=5)

        # Position 3: (2.0*0.81 + 3.0*0.9 + 4.0*1.0) / 2.71 = 8.52/2.71 ≈ 3.144
        expected_3 = (2.0*0.81 + 3.0*0.9 + 4.0*1.0) / weight_sum
        self.assertAlmostEqual(result[3], expected_3, places=5)

    def test_wma_nan_handling(self):
        """Test wma handles NaN values correctly (excludes NaN, renormalizes)."""
        x = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
        result = wma(x, n=3)

        # First 2 values should be NaN
        self.assertTrue(np.all(np.isnan(result[:2])))

        # Position 2: window [1.0, NaN, 3.0], valid weights [0.81, 1.0], sum = 1.81
        # (1.0*0.81 + 3.0*1.0) / 1.81 = 3.81/1.81 ≈ 2.104
        expected_2 = (1.0*0.81 + 3.0*1.0) / (0.81 + 1.0)
        self.assertAlmostEqual(result[2], expected_2, places=5)

        # Position 3: window [NaN, 3.0, 4.0], valid weights [0.9, 1.0], sum = 1.9
        # (3.0*0.9 + 4.0*1.0) / 1.9 = 6.7/1.9 ≈ 3.526
        expected_3 = (3.0*0.9 + 4.0*1.0) / (0.9 + 1.0)
        self.assertAlmostEqual(result[3], expected_3, places=5)

    def test_wma_all_nan_window(self):
        """Test wma when entire window is NaN."""
        x = np.array([1.0, 2.0, np.nan, np.nan, np.nan, 3.0])
        result = wma(x, n=3)

        # Position 4 (0-indexed): window is [NaN, NaN, NaN] -> NaN
        self.assertTrue(np.isnan(result[4]))

    def test_wma_edge_cases(self):
        """Test wma edge cases."""
        # Window equals array length
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = wma(x, n=5)

        # Only last value should be non-NaN
        self.assertTrue(np.all(np.isnan(result[:-1])))
        self.assertFalse(np.isnan(result[-1]))

        # n=1 should return the original values
        result_n1 = wma(x, n=1)
        np.testing.assert_array_almost_equal(result_n1, x)

    def test_wma_empty_or_short(self):
        """Test wma with empty or short arrays."""
        # Empty array
        x_empty = np.array([], dtype=float)
        result = wma(x_empty, n=3)
        self.assertEqual(len(result), 0)

        # Short array (window larger than array)
        x_short = np.array([1.0, 2.0])
        result = wma(x_short, n=3)
        self.assertTrue(np.all(np.isnan(result)))

    def test_wma_negative_n(self):
        """Test wma with negative n raises ValueError."""
        x = np.array([1.0, 2.0, 3.0])

        with self.assertRaises(ValueError):
            wma(x, n=-1)

    def test_wma_zero_n(self):
        """Test wma with n=0 raises ValueError."""
        x = np.array([1.0, 2.0, 3.0])

        with self.assertRaises(ValueError):
            wma(x, n=0)


class TestDecayLinear(unittest.TestCase):
    """Tests for decay_linear (Linear Decay Weighted Average) function."""

    def test_decay_linear_basic(self):
        """Test decay_linear with known values."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = decay_linear(x, d=3)

        # First 2 values should be NaN
        self.assertTrue(np.all(np.isnan(result[:2])))

        # weights = [1, 2, 3], sum = 6 (oldest gets weight 1, newest gets weight 3)
        # Position 2: (1.0*1 + 2.0*2 + 3.0*3) / 6 = 14/6 ≈ 2.333
        expected_2 = (1.0*1 + 2.0*2 + 3.0*3) / 6.0
        self.assertAlmostEqual(result[2], expected_2, places=5)

        # Position 3: (2.0*1 + 3.0*2 + 4.0*3) / 6 = 20/6 ≈ 3.333
        expected_3 = (2.0*1 + 3.0*2 + 4.0*3) / 6.0
        self.assertAlmostEqual(result[3], expected_3, places=5)

        # Position 4: (3.0*1 + 4.0*2 + 5.0*3) / 6 = 26/6 ≈ 4.333
        expected_4 = (3.0*1 + 4.0*2 + 5.0*3) / 6.0
        self.assertAlmostEqual(result[4], expected_4, places=5)

    def test_decay_linear_nan_handling(self):
        """Test decay_linear handles NaN values correctly (excludes NaN, renormalizes)."""
        x = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
        result = decay_linear(x, d=3)

        # First 2 values should be NaN
        self.assertTrue(np.all(np.isnan(result[:2])))

        # Position 2: window [1.0, NaN, 3.0], valid weights [1, 3], sum = 4
        # (1.0*1 + 3.0*3) / 4 = 10/4 = 2.5
        expected_2 = (1.0*1 + 3.0*3) / (1.0 + 3.0)
        self.assertAlmostEqual(result[2], expected_2, places=5)

        # Position 3: window [NaN, 3.0, 4.0], valid weights [2, 3], sum = 5
        # (3.0*2 + 4.0*3) / 5 = 18/5 = 3.6
        expected_3 = (3.0*2 + 4.0*3) / (2.0 + 3.0)
        self.assertAlmostEqual(result[3], expected_3, places=5)

    def test_decay_linear_all_nan_window(self):
        """Test decay_linear when entire window is NaN."""
        x = np.array([1.0, 2.0, np.nan, np.nan, np.nan, 3.0])
        result = decay_linear(x, d=3)

        # Position 4 (0-indexed): window is [NaN, NaN, NaN] -> NaN
        self.assertTrue(np.isnan(result[4]))

    def test_decay_linear_edge_cases(self):
        """Test decay_linear edge cases."""
        # Window equals array length
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = decay_linear(x, d=5)

        # Only last value should be non-NaN
        self.assertTrue(np.all(np.isnan(result[:-1])))
        self.assertFalse(np.isnan(result[-1]))

        # d=1 should return the original values
        result_d1 = decay_linear(x, d=1)
        np.testing.assert_array_almost_equal(result_d1, x)

    def test_decay_linear_empty_or_short(self):
        """Test decay_linear with empty or short arrays."""
        # Empty array
        x_empty = np.array([], dtype=float)
        result = decay_linear(x_empty, d=3)
        self.assertEqual(len(result), 0)

        # Short array (window larger than array)
        x_short = np.array([1.0, 2.0])
        result = decay_linear(x_short, d=3)
        self.assertTrue(np.all(np.isnan(result)))

    def test_decay_linear_negative_d(self):
        """Test decay_linear with negative d raises ValueError."""
        x = np.array([1.0, 2.0, 3.0])

        with self.assertRaises(ValueError):
            decay_linear(x, d=-1)

    def test_decay_linear_zero_d(self):
        """Test decay_linear with d=0 raises ValueError."""
        x = np.array([1.0, 2.0, 3.0])

        with self.assertRaises(ValueError):
            decay_linear(x, d=0)


class TestSumIf(unittest.TestCase):
    """Tests for sum_if function."""

    def test_sum_if_basic(self):
        """Test sum_if with known values."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        condition = np.array([True, False, True, True, False])
        result = sum_if(x, 3, condition)

        # First 2 values should be NaN
        self.assertTrue(np.all(np.isnan(result[:2])))

        # Position 2: window [1, 2, 3], condition [T, F, T] -> 1 + 3 = 4
        self.assertEqual(result[2], 4.0)

        # Position 3: window [2, 3, 4], condition [F, T, T] -> 3 + 4 = 7
        self.assertEqual(result[3], 7.0)

        # Position 4: window [3, 4, 5], condition [T, T, F] -> 3 + 4 = 7
        self.assertEqual(result[4], 7.0)

    def test_sum_if_numerical_condition(self):
        """Test sum_if with numerical condition (non-zero is True)."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        condition = np.array([1.0, 0.0, 2.0, 0.0, 3.0])  # non-zero is True
        result = sum_if(x, 3, condition)

        # First 2 values should be NaN
        self.assertTrue(np.all(np.isnan(result[:2])))

        # Position 2: window [1, 2, 3], condition [1, 0, 2] -> 1 + 3 = 4
        self.assertEqual(result[2], 4.0)

    def test_sum_if_nan_handling(self):
        """Test sum_if handles NaN values correctly."""
        x = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
        condition = np.array([True, True, True, True, True])
        result = sum_if(x, 3, condition)

        # First 2 values should be NaN
        self.assertTrue(np.all(np.isnan(result[:2])))

        # Position 2: window [1, NaN, 3], condition all True -> 1 + 3 = 4 (NaN excluded)
        self.assertEqual(result[2], 4.0)

        # Position 3: window [NaN, 3, 4], condition all True -> 3 + 4 = 7
        self.assertEqual(result[3], 7.0)

    def test_sum_if_condition_nan(self):
        """Test sum_if with NaN in condition (treated as False)."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        condition = np.array([True, np.nan, True, True, False])
        result = sum_if(x, 3, condition)

        # First 2 values should be NaN
        self.assertTrue(np.all(np.isnan(result[:2])))

        # Position 2: window [1, 2, 3], condition [T, NaN, T] -> 1 + 3 = 4 (NaN excluded)
        self.assertEqual(result[2], 4.0)

    def test_sum_if_all_false(self):
        """Test sum_if when all conditions are False in window."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        condition = np.array([False, False, False, True, True])
        result = sum_if(x, 3, condition)

        # First 2 values should be NaN
        self.assertTrue(np.all(np.isnan(result[:2])))

        # Position 2: window [1, 2, 3], condition [F, F, F] -> all False -> result is 0
        self.assertEqual(result[2], 0.0)

        # Position 3: window [2, 3, 4], condition [F, F, T] -> one True -> result is 4
        self.assertEqual(result[3], 4.0)

        # Position 4: window [3, 4, 5], condition [F, T, T] -> 4 + 5 = 9
        self.assertEqual(result[4], 9.0)

    def test_sum_if_edge_cases(self):
        """Test sum_if edge cases."""
        # Single True condition
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        condition = np.array([False, False, True, False, False])
        result = sum_if(x, 3, condition)

        # First 2 values should be NaN
        self.assertTrue(np.all(np.isnan(result[:2])))

        # Position 2: only 3.0 has True condition -> 3.0
        self.assertEqual(result[2], 3.0)

    def test_sum_if_empty_or_short(self):
        """Test sum_if with empty or short arrays."""
        # Empty array
        x_empty = np.array([], dtype=float)
        condition_empty = np.array([], dtype=bool)
        result = sum_if(x_empty, 3, condition_empty)
        self.assertEqual(len(result), 0)

        # Short array (window larger than array)
        x_short = np.array([1.0, 2.0])
        condition_short = np.array([True, True])
        result = sum_if(x_short, 3, condition_short)
        self.assertTrue(np.all(np.isnan(result)))

    def test_sum_if_negative_n(self):
        """Test sum_if with negative n raises ValueError."""
        x = np.array([1.0, 2.0, 3.0])
        condition = np.array([True, True, True])

        with self.assertRaises(ValueError):
            sum_if(x, -1, condition)

    def test_sum_if_different_lengths(self):
        """Test sum_if with different length arrays raises ValueError."""
        x = np.array([1.0, 2.0, 3.0])
        condition = np.array([True, True])

        with self.assertRaises(ValueError):
            sum_if(x, 2, condition)


class TestFilterArray(unittest.TestCase):
    """Tests for filter_array function."""

    def test_filter_array_basic(self):
        """Test filter_array with known values."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        condition = np.array([True, False, True, False, True])
        result = filter_array(x, condition)

        expected = np.array([1.0, 3.0, 5.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_filter_array_numerical_condition(self):
        """Test filter_array with numerical condition (non-zero is True)."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        condition = np.array([1.0, 0.0, 2.0, 0.0, 3.0])
        result = filter_array(x, condition)

        expected = np.array([1.0, 3.0, 5.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_filter_array_nan_handling(self):
        """Test filter_array handles NaN in condition as False."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        condition = np.array([True, np.nan, True, np.nan, True])
        result = filter_array(x, condition)

        # NaN in condition should be treated as False
        expected = np.array([1.0, 3.0, 5.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_filter_array_all_true(self):
        """Test filter_array with all True condition."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        condition = np.array([True, True, True, True, True])
        result = filter_array(x, condition)

        # Should return all values
        np.testing.assert_array_almost_equal(result, x)

    def test_filter_array_all_false(self):
        """Test filter_array with all False condition."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        condition = np.array([False, False, False, False, False])
        result = filter_array(x, condition)

        # Should return empty array
        self.assertEqual(len(result), 0)

    def test_filter_array_empty(self):
        """Test filter_array with empty arrays."""
        x_empty = np.array([], dtype=float)
        condition_empty = np.array([], dtype=bool)
        result = filter_array(x_empty, condition_empty)
        self.assertEqual(len(result), 0)

    def test_filter_array_different_lengths(self):
        """Test filter_array with different length arrays raises ValueError."""
        x = np.array([1.0, 2.0, 3.0])
        condition = np.array([True, True])

        with self.assertRaises(ValueError):
            filter_array(x, condition)


class TestHighDay(unittest.TestCase):
    """Tests for high_day function."""

    def test_high_day_basic(self):
        """Test high_day with known values."""
        x = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
        result = high_day(x, 3)

        # First 2 values should be NaN
        self.assertTrue(np.all(np.isnan(result[:2])))

        # Position 2: window [1, 3, 2], max at position 1 (value 3)
        # Days since max = 2 - 1 = 1
        self.assertEqual(result[2], 1.0)

        # Position 3: window [3, 2, 5], max at position 3 (value 5)
        # Days since max = 3 - 3 = 0
        self.assertEqual(result[3], 0.0)

        # Position 4: window [2, 5, 4], max at position 3 (value 5)
        # Days since max = 4 - 3 = 1
        self.assertEqual(result[4], 1.0)

    def test_high_day_current_is_max(self):
        """Test high_day when current value is the max."""
        x = np.array([1.0, 2.0, 5.0, 3.0, 4.0])
        result = high_day(x, 3)

        # Position 2: window [1, 2, 5], max at position 2 (current)
        self.assertEqual(result[2], 0.0)

    def test_high_day_nan_handling(self):
        """Test high_day handles NaN values correctly."""
        x = np.array([1.0, np.nan, 5.0, 3.0, 4.0])
        result = high_day(x, 3)

        # First 2 values should be NaN
        self.assertTrue(np.all(np.isnan(result[:2])))

        # Position 2: window [1, NaN, 5], max at position 2 (value 5)
        self.assertEqual(result[2], 0.0)

        # Position 3: window [NaN, 5, 3], max at position 2 (value 5)
        # Days since max = 3 - 2 = 1
        self.assertEqual(result[3], 1.0)

    def test_high_day_all_nan_window(self):
        """Test high_day when entire window is NaN."""
        x = np.array([1.0, 2.0, np.nan, np.nan, np.nan, 3.0])
        result = high_day(x, 3)

        # Position 4: window is [NaN, NaN, NaN] -> NaN
        self.assertTrue(np.isnan(result[4]))

    def test_high_day_multiple_max(self):
        """Test high_day with multiple maximum values."""
        # If multiple maxes, should use first occurrence
        x = np.array([5.0, 2.0, 5.0, 3.0, 4.0])
        result = high_day(x, 3)

        # First 2 values should be NaN
        self.assertTrue(np.all(np.isnan(result[:2])))

        # Position 2: window [5, 2, 5], max at positions 0 and 2
        # First occurrence is at position 0, days since = 2 - 0 = 2
        self.assertEqual(result[2], 2.0)

    def test_high_day_empty_or_short(self):
        """Test high_day with empty or short arrays."""
        # Empty array
        x_empty = np.array([], dtype=float)
        result = high_day(x_empty, 3)
        self.assertEqual(len(result), 0)

        # Short array (window larger than array)
        x_short = np.array([1.0, 2.0])
        result = high_day(x_short, 3)
        self.assertTrue(np.all(np.isnan(result)))

    def test_high_day_negative_n(self):
        """Test high_day with negative n raises ValueError."""
        x = np.array([1.0, 2.0, 3.0])

        with self.assertRaises(ValueError):
            high_day(x, -1)


class TestLowDay(unittest.TestCase):
    """Tests for low_day function."""

    def test_low_day_basic(self):
        """Test low_day with known values."""
        x = np.array([5.0, 3.0, 4.0, 1.0, 2.0])
        result = low_day(x, 3)

        # First 2 values should be NaN
        self.assertTrue(np.all(np.isnan(result[:2])))

        # Position 2: window [5, 3, 4], min at position 1 (value 3)
        # Days since min = 2 - 1 = 1
        self.assertEqual(result[2], 1.0)

        # Position 3: window [3, 4, 1], min at position 3 (value 1)
        # Days since min = 3 - 3 = 0
        self.assertEqual(result[3], 0.0)

        # Position 4: window [4, 1, 2], min at position 3 (value 1)
        # Days since min = 4 - 3 = 1
        self.assertEqual(result[4], 1.0)

    def test_low_day_current_is_min(self):
        """Test low_day when current value is the min."""
        x = np.array([5.0, 4.0, 1.0, 3.0, 2.0])
        result = low_day(x, 3)

        # Position 2: window [5, 4, 1], min at position 2 (current)
        self.assertEqual(result[2], 0.0)

    def test_low_day_nan_handling(self):
        """Test low_day handles NaN values correctly."""
        x = np.array([5.0, np.nan, 1.0, 3.0, 4.0])
        result = low_day(x, 3)

        # First 2 values should be NaN
        self.assertTrue(np.all(np.isnan(result[:2])))

        # Position 2: window [5, NaN, 1], min at position 2 (value 1)
        self.assertEqual(result[2], 0.0)

        # Position 3: window [NaN, 1, 3], min at position 2 (value 1)
        # Days since min = 3 - 2 = 1
        self.assertEqual(result[3], 1.0)

    def test_low_day_all_nan_window(self):
        """Test low_day when entire window is NaN."""
        x = np.array([5.0, 4.0, np.nan, np.nan, np.nan, 3.0])
        result = low_day(x, 3)

        # Position 4: window is [NaN, NaN, NaN] -> NaN
        self.assertTrue(np.isnan(result[4]))

    def test_low_day_multiple_min(self):
        """Test low_day with multiple minimum values."""
        # If multiple mins, should use first occurrence
        x = np.array([1.0, 4.0, 1.0, 3.0, 2.0])
        result = low_day(x, 3)

        # First 2 values should be NaN
        self.assertTrue(np.all(np.isnan(result[:2])))

        # Position 2: window [1, 4, 1], min at positions 0 and 2
        # First occurrence is at position 0, days since = 2 - 0 = 2
        self.assertEqual(result[2], 2.0)

    def test_low_day_empty_or_short(self):
        """Test low_day with empty or short arrays."""
        # Empty array
        x_empty = np.array([], dtype=float)
        result = low_day(x_empty, 3)
        self.assertEqual(len(result), 0)

        # Short array (window larger than array)
        x_short = np.array([5.0, 4.0])
        result = low_day(x_short, 3)
        self.assertTrue(np.all(np.isnan(result)))

    def test_low_day_negative_n(self):
        """Test low_day with negative n raises ValueError."""
        x = np.array([5.0, 4.0, 3.0])

        with self.assertRaises(ValueError):
            low_day(x, -1)


class TestSequence(unittest.TestCase):
    """Tests for sequence function."""

    def test_sequence_basic(self):
        """Test sequence with known values."""
        result = sequence(5)

        expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_sequence_n1(self):
        """Test sequence with n=1."""
        result = sequence(1)

        expected = np.array([1.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_sequence_large_n(self):
        """Test sequence with larger n."""
        result = sequence(10)

        expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_sequence_returns_float(self):
        """Test sequence returns float array."""
        result = sequence(5)

        self.assertEqual(result.dtype, np.float64)

    def test_sequence_negative_n(self):
        """Test sequence with negative n raises ValueError."""
        with self.assertRaises(ValueError):
            sequence(-1)

    def test_sequence_zero_n(self):
        """Test sequence with n=0 raises ValueError."""
        with self.assertRaises(ValueError):
            sequence(0)


class TestComputeRet(unittest.TestCase):
    """Tests for compute_ret function."""
    
    def test_compute_ret_basic(self):
        """Test compute_ret with known values."""
        close = np.array([100, 102, 99, 105])
        result = compute_ret(close)
        
        self.assertTrue(np.isnan(result[0]))
        self.assertAlmostEqual(result[1], 0.02, places=4)
        self.assertAlmostEqual(result[2], -0.0294, places=4)
        self.assertAlmostEqual(result[3], 0.0606, places=4)
    
    def test_compute_ret_nan_handling(self):
        """Test compute_ret handles NaN values correctly."""
        close = np.array([100, np.nan, 102, 99])
        result = compute_ret(close)
        
        self.assertTrue(np.isnan(result[0]))
        self.assertTrue(np.isnan(result[1]))
        self.assertTrue(np.isnan(result[2]))
        self.assertAlmostEqual(result[3], -0.0294, places=4)
    
    def test_compute_ret_edge_cases(self):
        """Test compute_ret with edge cases."""
        # Single value
        close_single = np.array([100])
        result = compute_ret(close_single)
        self.assertEqual(len(result), 1)
        self.assertTrue(np.isnan(result[0]))
        
        # Empty array
        close_empty = np.array([], dtype=float)
        result = compute_ret(close_empty)
        self.assertEqual(len(result), 0)
    
    def test_compute_ret_typical_usage(self):
        """Test compute_ret with typical OHLCV data."""
        close = np.array([100.0, 101.5, 102.0, 101.8, 103.2])
        result = compute_ret(close)
        
        self.assertTrue(np.isnan(result[0]))
        self.assertAlmostEqual(result[1], 0.015, places=4)
        self.assertAlmostEqual(result[2], 0.0049, places=4)
        self.assertAlmostEqual(result[3], -0.00196, places=4)
        self.assertAlmostEqual(result[4], 0.01376, places=4)


class TestComputeDtm(unittest.TestCase):
    """Tests for compute_dtm function."""
    
    def test_compute_dtm_basic(self):
        """Test compute_dtm with known values."""
        open_p = np.array([10, 11, 10, 12])
        high = np.array([12, 13, 11, 14])
        result = compute_dtm(open_p, high)
        
        self.assertTrue(np.isnan(result[0]))
        self.assertEqual(result[1], 2.0)  # MAX(13-11, 11-10) = MAX(2, 1) = 2
        self.assertEqual(result[2], 0.0)  # OPEN <= PREV_OPEN
        self.assertEqual(result[3], 2.0)  # MAX(14-12, 12-10) = MAX(2, 2) = 2
    
    def test_compute_dtm_nan_handling(self):
        """Test compute_dtm handles NaN values correctly."""
        open_p = np.array([10, np.nan, 11, 10])
        high = np.array([12, 13, np.nan, 11])
        result = compute_dtm(open_p, high)
        
        self.assertTrue(np.isnan(result[0]))
        self.assertTrue(np.isnan(result[1]))
        self.assertTrue(np.isnan(result[2]))
        self.assertEqual(result[3], 0.0)
    
    def test_compute_dtm_edge_cases(self):
        """Test compute_dtm with edge cases."""
        # Single value
        open_single = np.array([10])
        high_single = np.array([12])
        result = compute_dtm(open_single, high_single)
        self.assertEqual(len(result), 1)
        self.assertTrue(np.isnan(result[0]))
        
        # Empty array
        open_empty = np.array([], dtype=float)
        high_empty = np.array([], dtype=float)
        result = compute_dtm(open_empty, high_empty)
        self.assertEqual(len(result), 0)
    
    def test_compute_dtm_typical_usage(self):
        """Test compute_dtm with typical OHLCV data."""
        open_p = np.array([100, 101, 100.5, 102, 101.8])
        high = np.array([102, 103, 101.5, 104, 102.5])
        result = compute_dtm(open_p, high)
        
        self.assertTrue(np.isnan(result[0]))
        self.assertEqual(result[1], 2.0)    # MAX(103-101, 101-100) = 2
        self.assertEqual(result[2], 0.0)    # OPEN <= PREV_OPEN
        self.assertEqual(result[3], 2.0)    # MAX(104-102, 102-100.5) = 2
        self.assertEqual(result[4], 0.0)    # OPEN <= PREV_OPEN


class TestComputeDbm(unittest.TestCase):
    """Tests for compute_dbm function."""
    
    def test_compute_dbm_basic(self):
        """Test compute_dbm with known values."""
        open_p = np.array([10, 9, 11, 8])
        low = np.array([9, 8, 10, 7])
        result = compute_dbm(open_p, low)
        
        self.assertTrue(np.isnan(result[0]))
        self.assertEqual(result[1], 1.0)  # MAX(9-8, 9-10) = MAX(1, -1) = 1
        self.assertEqual(result[2], 0.0)  # OPEN >= PREV_OPEN
        self.assertEqual(result[3], 1.0)  # MAX(8-7, 8-11) = MAX(1, -3) = 1
    
    def test_compute_dbm_nan_handling(self):
        """Test compute_dbm handles NaN values correctly."""
        open_p = np.array([10, np.nan, 9, 11])
        low = np.array([9, 8, np.nan, 10])
        result = compute_dbm(open_p, low)
        
        self.assertTrue(np.isnan(result[0]))
        self.assertTrue(np.isnan(result[1]))
        self.assertTrue(np.isnan(result[2]))
        self.assertEqual(result[3], 0.0)
    
    def test_compute_dbm_edge_cases(self):
        """Test compute_dbm with edge cases."""
        # Single value
        open_single = np.array([10])
        low_single = np.array([8])
        result = compute_dbm(open_single, low_single)
        self.assertEqual(len(result), 1)
        self.assertTrue(np.isnan(result[0]))
        
        # Empty array
        open_empty = np.array([], dtype=float)
        low_empty = np.array([], dtype=float)
        result = compute_dbm(open_empty, low_empty)
        self.assertEqual(len(result), 0)
    
    def test_compute_dbm_typical_usage(self):
        """Test compute_dbm with typical OHLCV data."""
        open_p = np.array([100, 99, 100.5, 99.8, 100])
        low = np.array([98, 97, 99.5, 98.5, 99])
        result = compute_dbm(open_p, low)
        
        self.assertTrue(np.isnan(result[0]))
        self.assertEqual(result[1], 2.0)    # MAX(99-97, 99-100) = 2
        self.assertEqual(result[2], 0.0)    # OPEN >= PREV_OPEN
        self.assertAlmostEqual(result[3], 1.3, places=3)    # MAX(99.8-98.5, 99.8-100.5) = 1.3
        self.assertEqual(result[4], 0.0)    # OPEN >= PREV_OPEN


class TestComputeTr(unittest.TestCase):
    """Tests for compute_tr function."""
    
    def test_compute_tr_basic(self):
        """Test compute_tr with known values."""
        high = np.array([12, 13, 11, 14])
        low = np.array([10, 11, 9, 12])
        close = np.array([11, 12, 10, 13])
        result = compute_tr(high, low, close)
        
        self.assertTrue(np.isnan(result[0]))
        self.assertEqual(result[1], 2.0)  # MAX(13-11, |13-11|, |11-11|) = 2
        self.assertEqual(result[2], 3.0)  # MAX(11-9, |11-12|, |9-12|) = 3
        self.assertEqual(result[3], 4.0)  # MAX(14-12, |14-10|, |12-10|) = 4
    
    def test_compute_tr_nan_handling(self):
        """Test compute_tr handles NaN values correctly."""
        high = np.array([12, np.nan, 11, 14])
        low = np.array([10, 11, np.nan, 12])
        close = np.array([11, 12, 10, np.nan])
        result = compute_tr(high, low, close)
        
        self.assertTrue(np.isnan(result[0]))
        self.assertTrue(np.isnan(result[1]))
        self.assertTrue(np.isnan(result[2]))
        self.assertEqual(result[3], 4.0)  # MAX(14-12, |14-10|, |12-10|) = 4
    
    def test_compute_tr_edge_cases(self):
        """Test compute_tr with edge cases."""
        # Single value
        high_single = np.array([12])
        low_single = np.array([10])
        close_single = np.array([11])
        result = compute_tr(high_single, low_single, close_single)
        self.assertEqual(len(result), 1)
        self.assertTrue(np.isnan(result[0]))
        
        # Empty array
        high_empty = np.array([], dtype=float)
        low_empty = np.array([], dtype=float)
        close_empty = np.array([], dtype=float)
        result = compute_tr(high_empty, low_empty, close_empty)
        self.assertEqual(len(result), 0)
    
    def test_compute_tr_typical_usage(self):
        """Test compute_tr with typical OHLCV data."""
        high = np.array([102, 103, 101.5, 104, 102.5])
        low = np.array([98, 97, 99.5, 98.5, 99])
        close = np.array([100, 99, 100.5, 99.8, 100])
        result = compute_tr(high, low, close)
        
        self.assertTrue(np.isnan(result[0]))
        self.assertEqual(result[1], 6.0)    # MAX(103-97, |103-100|, |97-100|) = 6
        self.assertEqual(result[2], 2.5)    # MAX(101.5-99.5, |101.5-99|, |99.5-99|) = 2.5
        self.assertEqual(result[3], 5.5)    # MAX(104-98.5, |104-100.5|, |98.5-100.5|) = 5.5
        self.assertEqual(result[4], 3.5)    # MAX(102.5-99, |102.5-99.8|, |99-99.8|) = 3.5


class TestComputeHd(unittest.TestCase):
    """Tests for compute_hd function."""
    
    def test_compute_hd_basic(self):
        """Test compute_hd with known values."""
        high = np.array([10, 12, 11, 13, 12])
        result = compute_hd(high)
        
        self.assertTrue(np.isnan(result[0]))
        self.assertEqual(result[1], 2.0)
        self.assertEqual(result[2], -1.0)
        self.assertEqual(result[3], 2.0)
        self.assertEqual(result[4], -1.0)
    
    def test_compute_hd_nan_handling(self):
        """Test compute_hd handles NaN values correctly."""
        high = np.array([10, np.nan, 11, np.nan, 12])
        result = compute_hd(high)
        
        self.assertTrue(np.isnan(result[0]))
        self.assertTrue(np.isnan(result[1]))
        self.assertTrue(np.isnan(result[2]))
        self.assertTrue(np.isnan(result[3]))
        self.assertTrue(np.isnan(result[4]))
    
    def test_compute_hd_edge_cases(self):
        """Test compute_hd with edge cases."""
        # Single value
        high_single = np.array([10])
        result = compute_hd(high_single)
        self.assertEqual(len(result), 1)
        self.assertTrue(np.isnan(result[0]))
        
        # Empty array
        high_empty = np.array([], dtype=float)
        result = compute_hd(high_empty)
        self.assertEqual(len(result), 0)
    
    def test_compute_hd_typical_usage(self):
        """Test compute_hd with typical OHLCV data."""
        high = np.array([100, 101, 102, 101.5, 103])
        result = compute_hd(high)
        
        self.assertTrue(np.isnan(result[0]))
        self.assertEqual(result[1], 1.0)
        self.assertEqual(result[2], 1.0)
        self.assertEqual(result[3], -0.5)
        self.assertEqual(result[4], 1.5)


class TestComputeLd(unittest.TestCase):
    """Tests for compute_ld function."""
    
    def test_compute_ld_basic(self):
        """Test compute_ld with known values."""
        low = np.array([10, 8, 9, 7, 8])
        result = compute_ld(low)
        
        self.assertTrue(np.isnan(result[0]))
        self.assertEqual(result[1], 2.0)
        self.assertEqual(result[2], -1.0)
        self.assertEqual(result[3], 2.0)
        self.assertEqual(result[4], -1.0)
    
    def test_compute_ld_nan_handling(self):
        """Test compute_ld handles NaN values correctly."""
        low = np.array([10, np.nan, 9, np.nan, 8])
        result = compute_ld(low)
        
        self.assertTrue(np.isnan(result[0]))
        self.assertTrue(np.isnan(result[1]))
        self.assertTrue(np.isnan(result[2]))
        self.assertTrue(np.isnan(result[3]))
        self.assertTrue(np.isnan(result[4]))
    
    def test_compute_ld_edge_cases(self):
        """Test compute_ld with edge cases."""
        # Single value
        low_single = np.array([10])
        result = compute_ld(low_single)
        self.assertEqual(len(result), 1)
        self.assertTrue(np.isnan(result[0]))
        
        # Empty array
        low_empty = np.array([], dtype=float)
        result = compute_ld(low_empty)
        self.assertEqual(len(result), 0)
    
    def test_compute_ld_typical_usage(self):
        """Test compute_ld with typical OHLCV data."""
        low = np.array([100, 99, 98, 98.5, 97])
        result = compute_ld(low)
        
        self.assertTrue(np.isnan(result[0]))
        self.assertEqual(result[1], 1.0)
        self.assertEqual(result[2], 1.0)
        self.assertEqual(result[3], -0.5)
        self.assertEqual(result[4], 1.5)


if __name__ == '__main__':
    unittest.main()
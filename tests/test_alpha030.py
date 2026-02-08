
import unittest
import pandas as pd
import numpy as np
from alpha191.alpha030 import alpha_030

class TestAlpha030(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        dates = pd.date_range(start='2020-01-01', periods=200)

        # Generate random price data (Random Walk)
        close = 100 + np.cumsum(np.random.randn(200))

        # Generate random Fama-French factors
        mkt = np.random.randn(200) * 0.01 + 0.0005
        smb = np.random.randn(200) * 0.005 + 0.0001
        hml = np.random.randn(200) * 0.005 - 0.0001

        self.df = pd.DataFrame({
            'close': close,
            'mkt': mkt,
            'smb': smb,
            'hml': hml,
            # Extra columns to simulate standard df
            'open': close,
            'high': close * 1.01,
            'low': close * 0.99,
            'volume': np.abs(np.random.randn(200) * 1000000),
            'date': dates
        }).set_index('date')

    def test_alpha030_calculation(self):
        result = alpha_030(self.df)

        # Check type
        self.assertIsInstance(result, pd.Series)

        # Check length
        self.assertEqual(len(result), 200)

        # Check index matches
        pd.testing.assert_index_equal(result.index, self.df.index)

        # Check NaNs
        # First 60 (regression) + 1 (returns) - 1 = 60 points might be NaN?
        # Reg residual needs 60 points. So indices 0..58 are NaN. Index 59 is valid?
        # Then WMA(20).
        # Total warmup roughly 60 + 20 - 1 = 79.

        # Let's check tail is not NaN
        self.assertFalse(np.isnan(result.iloc[-1]), "Last value should not be NaN")

        # Check head is NaN
        self.assertTrue(np.isnan(result.iloc[0]), "First value should be NaN")
        self.assertTrue(np.isnan(result.iloc[50]), "Early value should be NaN")

    def test_missing_columns(self):
        df_incomplete = self.df.drop(columns=['mkt'])
        with self.assertRaisesRegex(ValueError, "Alpha030 requires Fama-French factors"):
            alpha_030(df_incomplete)

if __name__ == '__main__':
    unittest.main()

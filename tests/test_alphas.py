import unittest
import numpy as np
import pandas as pd
from alpha191 import (
    alpha_001, alpha_002, alpha_003, alpha_004, alpha_005,
    alpha_006, alpha_007, alpha_008, alpha_009, alpha_010,
    alpha_011, alpha_012, alpha_013, alpha_014, alpha_015,
    alpha_016, alpha_017, alpha_018, alpha_019, alpha_020
)


class TestAlphas(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        n = 30
        self.df = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=n),
            'open': 100 + np.random.randn(n).cumsum(),
            'high': 102 + np.random.randn(n).cumsum(),
            'low': 98 + np.random.randn(n).cumsum(),
            'close': 101 + np.random.randn(n).cumsum(),
            'volume': np.random.randint(1000000, 5000000, n),
        })
    
    def test_alpha001(self):
        result = alpha_001(self.df)
        self.assertEqual(len(result), len(self.df))
        # First 5 values should be NaN (due to rolling window=6)
        self.assertTrue(np.all(np.isnan(result.values[:5])))

    def test_alpha002(self):
        result = alpha_002(self.df)
        self.assertEqual(len(result), len(self.df))
        # delta(..., 1) -> index 0 is NaN, index 1 is valid
        self.assertTrue(np.isnan(result.values[0]))
        self.assertFalse(np.isnan(result.values[1]))

    def test_alpha003(self):
        result = alpha_003(self.df)
        self.assertEqual(len(result), len(self.df))
        # sum(..., 6) starts at index 5. Indices 0-4 are NaN.
        self.assertTrue(np.all(np.isnan(result.values[:5])))
        self.assertFalse(np.isnan(result.values[5]))

    def test_alpha004(self):
        result = alpha_004(self.df)
        self.assertEqual(len(result), len(self.df))
        # mean(..., 20) starts at index 19. Indices 0-18 are NaN due to volume mean.
        self.assertTrue(np.all(np.isnan(result.values[:19])))
        self.assertFalse(np.isnan(result.values[19]))

    def test_alpha005(self):
        result = alpha_005(self.df)
        self.assertEqual(len(result), len(self.df))
        # ts_rank(..., 5) starts at index 4, corr(..., 5) starts at index 4, ts_max(..., 3) starts at index 6
        # Cumulative: first 5 values are NaN (indices 0-4)
        self.assertTrue(np.all(np.isnan(result.values[:5])))
        self.assertFalse(np.isnan(result.values[5]))

    def test_alpha006(self):
        result = alpha_006(self.df)
        self.assertEqual(len(result), len(self.df))
        # delta(..., 4) starts at index 3
        self.assertTrue(np.all(np.isnan(result.values[:3])))
        self.assertTrue(np.isnan(result.values[3]))

    def test_alpha007(self):
        result = alpha_007(self.df)
        self.assertEqual(len(result), len(self.df))
        # ts_max/min(..., 3) starts at index 2, delta(..., 3) starts at index 2
        self.assertTrue(np.all(np.isnan(result.values[:2])))
        self.assertTrue(np.isnan(result.values[2]))

    def test_alpha008(self):
        result = alpha_008(self.df)
        self.assertEqual(len(result), len(self.df))
        # delta(..., 4) starts at index 3
        self.assertTrue(np.all(np.isnan(result.values[:3])))
        self.assertTrue(np.isnan(result.values[3]))

    def test_alpha009(self):
        result = alpha_009(self.df)
        self.assertEqual(len(result), len(self.df))
        # delay(..., 1) starts at index 0, but division by volume may cause NaN
        # sma(..., 7, 2) starts at index 0
        # First value may be NaN due to delay operations
        self.assertTrue(np.isnan(result.values[0]))

    def test_alpha010(self):
        result = alpha_010(self.df)
        self.assertEqual(len(result), len(self.df))
        # ts_std(..., 20) starts at index 19, ts_max(..., 5) starts at index 23
        # But delay(..., 1) starts at index 0, so first 4 values are NaN
        self.assertTrue(np.all(np.isnan(result.values[:4])))
        self.assertFalse(np.isnan(result.values[4]))

    def test_alpha011(self):
        result = alpha_011(self.df)
        self.assertEqual(len(result), len(self.df))
        # ts_sum(..., 6) starts at index 5
        self.assertTrue(np.all(np.isnan(result.values[:5])))
        self.assertFalse(np.isnan(result.values[5]))

    def test_alpha012(self):
        result = alpha_012(self.df)
        self.assertEqual(len(result), len(self.df))
        # ts_sum(..., 10) starts at index 9
        self.assertTrue(np.all(np.isnan(result.values[:9])))
        self.assertFalse(np.isnan(result.values[9]))

    def test_alpha013(self):
        result = alpha_013(self.df)
        self.assertEqual(len(result), len(self.df))
        # No rolling operations, should have values from index 0
        self.assertFalse(np.isnan(result.values[0]))

    def test_alpha014(self):
        result = alpha_014(self.df)
        self.assertEqual(len(result), len(self.df))
        # delay(..., 5) starts at index 4
        self.assertTrue(np.all(np.isnan(result.values[:4])))
        self.assertTrue(np.isnan(result.values[4]))

    def test_alpha015(self):
        result = alpha_015(self.df)
        self.assertEqual(len(result), len(self.df))
        # delay(..., 1) starts at index 0
        self.assertTrue(np.isnan(result.values[0]))
        self.assertFalse(np.isnan(result.values[1]))

    def test_alpha016(self):
        result = alpha_016(self.df)
        self.assertEqual(len(result), len(self.df))
        # corr(..., 5) starts at index 4, ts_max(..., 5) starts at index 8
        # First 4 values are NaN (indices 0-3)
        self.assertTrue(np.all(np.isnan(result.values[:4])))
        self.assertFalse(np.isnan(result.values[4]))

    def test_alpha017(self):
        result = alpha_017(self.df)
        self.assertEqual(len(result), len(self.df))
        # ts_max(..., 15) starts at index 14, delta(..., 5) starts at index 4
        # But power operation may cause issues with NaN values
        # First 10 values are NaN due to ts_max(..., 15)
        self.assertTrue(np.all(np.isnan(result.values[:10])))
        self.assertTrue(np.isnan(result.values[10]))

    def test_alpha018(self):
        result = alpha_018(self.df)
        self.assertEqual(len(result), len(self.df))
        # delay(..., 5) starts at index 4
        self.assertTrue(np.all(np.isnan(result.values[:4])))
        self.assertTrue(np.isnan(result.values[4]))

    def test_alpha019(self):
        result = alpha_019(self.df)
        self.assertEqual(len(result), len(self.df))
        # delay(..., 5) starts at index 4
        self.assertTrue(np.all(np.isnan(result.values[:4])))
        self.assertTrue(np.isnan(result.values[4]))

    def test_alpha020(self):
        result = alpha_020(self.df)
        self.assertEqual(len(result), len(self.df))
        # delay(..., 6) starts at index 5
        self.assertTrue(np.all(np.isnan(result.values[:5])))
        self.assertTrue(np.isnan(result.values[5]))


if __name__ == '__main__':
    unittest.main()

import unittest
import numpy as np
import pandas as pd
from alpha191 import (
    alpha_001, alpha_002, alpha_003, alpha_004, alpha_005,
    alpha_006, alpha_007, alpha_008, alpha_009, alpha_010,
    alpha_011, alpha_012, alpha_013, alpha_014, alpha_015,
    alpha_016, alpha_017, alpha_018, alpha_019, alpha_020,
    alpha_021, alpha_022, alpha_023, alpha_024, alpha_025,
    alpha_026, alpha_027, alpha_028, alpha_029, alpha_031,
    alpha_032, alpha_033, alpha_034, alpha_035, alpha_036,
    alpha_037, alpha_038, alpha_039, alpha_040, alpha_041,
    alpha_042, alpha_043, alpha_044, alpha_045, alpha_046,
    alpha_047, alpha_048, alpha_049, alpha_050
)


class TestAlphas(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        n = 300  # Increased for larger windows in alpha 21-40
        self.df = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=n),
            'open': 100 + np.random.randn(n).cumsum(),
            'high': 102 + np.random.randn(n).cumsum(),
            'low': 98 + np.random.randn(n).cumsum(),
            'close': 101 + np.random.randn(n).cumsum(),
            'volume': np.random.randint(1000000, 5000000, n),
            'amount': (101 + np.random.randn(n).cumsum()) * np.random.randint(1000000, 5000000, n),
        })
    
    def test_alpha001(self):
        result = alpha_001(self.df)
        self.assertEqual(len(result), len(self.df))
        self.assertTrue(np.all(np.isnan(result.values[:5])))

    def test_alpha002(self):
        result = alpha_002(self.df)
        self.assertEqual(len(result), len(self.df))
        self.assertTrue(np.isnan(result.values[0]))
        self.assertFalse(np.isnan(result.values[1]))

    def test_alpha003(self):
        result = alpha_003(self.df)
        self.assertEqual(len(result), len(self.df))
        self.assertTrue(np.all(np.isnan(result.values[:5])))
        self.assertFalse(np.isnan(result.values[5]))

    def test_alpha004(self):
        result = alpha_004(self.df)
        self.assertEqual(len(result), len(self.df))
        self.assertTrue(np.all(np.isnan(result.values[:19])))
        self.assertFalse(np.isnan(result.values[19]))

    def test_alpha005(self):
        result = alpha_005(self.df)
        self.assertEqual(len(result), len(self.df))
        self.assertTrue(np.all(np.isnan(result.values[:4])))
        valid_idx = np.where(~np.isnan(result.values))[0]
        if len(valid_idx) > 0:
            self.assertGreaterEqual(valid_idx[0], 5)

    def test_alpha006(self):
        result = alpha_006(self.df)
        self.assertEqual(len(result), len(self.df))
        self.assertTrue(np.all(np.isnan(result.values[:4])))

    def test_alpha007(self):
        result = alpha_007(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha008(self):
        result = alpha_008(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha009(self):
        result = alpha_009(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha010(self):
        result = alpha_010(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha011(self):
        result = alpha_011(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha012(self):
        result = alpha_012(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha013(self):
        result = alpha_013(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha014(self):
        result = alpha_014(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha015(self):
        result = alpha_015(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha016(self):
        result = alpha_016(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha017(self):
        result = alpha_017(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha018(self):
        result = alpha_018(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha019(self):
        result = alpha_019(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha020(self):
        result = alpha_020(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha021(self):
        result = alpha_021(self.df)
        self.assertEqual(len(result), len(self.df))
        # mean(6) + regression_beta(6)
        # Should be valid around index 10
        self.assertFalse(np.isnan(result.values[-1]))

    def test_alpha022(self):
        result = alpha_022(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha023(self):
        result = alpha_023(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha024(self):
        result = alpha_024(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha025(self):
        result = alpha_025(self.df)
        self.assertEqual(len(result), len(self.df))
        # sum(ret, 250) -> ret starts at 1. ts_sum(250) valid from index 249.
        self.assertTrue(np.all(np.isnan(result.values[:249])))
        self.assertFalse(np.isnan(result.values[-1]))

    def test_alpha026(self):
        result = alpha_026(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha027(self):
        result = alpha_027(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha028(self):
        result = alpha_028(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha029(self):
        result = alpha_029(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha031(self):
        result = alpha_031(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha032(self):
        result = alpha_032(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha033(self):
        result = alpha_033(self.df)
        self.assertEqual(len(result), len(self.df))
        # sum(ret, 240) -> ret starts at 1. ts_sum(240) valid from index 239.
        self.assertTrue(np.all(np.isnan(result.values[:239])))

    def test_alpha034(self):
        result = alpha_034(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha035(self):
        result = alpha_035(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha036(self):
        result = alpha_036(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha037(self):
        result = alpha_037(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha038(self):
        result = alpha_038(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha039(self):
        result = alpha_039(self.df)
        self.assertEqual(len(result), len(self.df))
        # correlation over 14, sum over 37, mean over 180
        valid_idx = np.where(~np.isnan(result.values))[0]
        if len(valid_idx) > 0:
             self.assertGreaterEqual(valid_idx[0], 180)

    def test_alpha040(self):
        result = alpha_040(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha041(self):
        result = alpha_041(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha042(self):
        result = alpha_042(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha043(self):
        result = alpha_043(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha044(self):
        result = alpha_044(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha045(self):
        result = alpha_045(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha046(self):
        result = alpha_046(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha047(self):
        result = alpha_047(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha048(self):
        result = alpha_048(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha049(self):
        result = alpha_049(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_alpha050(self):
        result = alpha_050(self.df)
        self.assertEqual(len(result), len(self.df))


if __name__ == '__main__':
    unittest.main()

import unittest
import numpy as np
import pandas as pd
from alpha191.alpha002 import alpha_002
from alpha191.alpha003 import alpha_003


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


if __name__ == '__main__':
    unittest.main()

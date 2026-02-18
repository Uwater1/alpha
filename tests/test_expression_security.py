import sys
from unittest.mock import MagicMock

# Mock dependencies before importing the module under test
mock_lark = MagicMock()
sys.modules['lark'] = mock_lark
sys.modules['pandas'] = MagicMock()
sys.modules['numpy'] = MagicMock()
sys.modules['numba'] = MagicMock()
sys.modules['scipy'] = MagicMock()
sys.modules['scipy.stats'] = MagicMock()
sys.modules['baostock'] = MagicMock()
sys.modules['jqdatasdk'] = MagicMock()

import unittest
import re

# We need to mock the operators too because get_func imports them
sys.modules['alpha191.operators'] = MagicMock()

from alpha191.expression.expression import ExpressionAlpha

class TestExpressionSecurity(unittest.TestCase):
    def test_func_name_validation(self):
        # Mock the parser to avoid errors during init
        mock_lark.Lark.return_value = MagicMock()

        ea = ExpressionAlpha("rank(close)")

        # Malicious function names
        malicious_names = [
            "alpha_expr(df):\n    import os; os.system('ls')",
            "alpha_expr; import os; os.system('ls')",
            "alpha_expr'); import os; os.system('ls'); #",
            "123_invalid",
            "func-name",
            "func name",
            "alpha_expr\""
        ]

        for name in malicious_names:
            with self.subTest(name=name):
                with self.assertRaises(ValueError) as cm:
                    ea.to_python(func_name=name)
                self.assertIn("Invalid function name", str(cm.exception))

                with self.assertRaises(ValueError) as cm:
                    ea.get_func(func_name=name)
                self.assertIn("Invalid function name", str(cm.exception))

    def test_valid_func_name(self):
        # Mock the parser to avoid errors during init
        mock_lark.Lark.return_value = MagicMock()
        ea = ExpressionAlpha("rank(close)")

        # This should not raise ValueError for the name
        # It might still fail later because we mocked transformer.transform, but that's fine
        try:
            ea.to_python(func_name="valid_name_123")
        except ValueError as e:
            if "Invalid function name" in str(e):
                self.fail(f"Valid function name 'valid_name_123' was rejected: {e}")
        except Exception:
            # Other exceptions are expected due to mocking
            pass

if __name__ == "__main__":
    unittest.main()

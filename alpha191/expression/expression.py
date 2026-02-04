import itertools
import os
from lark import Lark, Transformer
import pandas as pd
import numpy as np

class ExpressionTransformer(Transformer):
    def __init__(self):
        self.cmdlist = []
        self.vcounter = itertools.count()
        self.stack = []
        self.inputs = set()
        # Initial columns to extract from df
        self.columns_needed = set()

    def _get_v(self):
        return f"v{next(self.vcounter)}"

    def number(self, items):
        self.stack.append(str(items[0].value))

    # Basic Data
    def identifier(self, items):
        name = str(items[0])
        # If it's a known column, add it to columns_needed
        # For now, let's treat any unknown identifier as a column from df
        self.columns_needed.add(name)
        self.stack.append(name)

    def close(self, items):
        self.columns_needed.add('close')
        self.stack.append('close')

    def opens(self, items):
        self.columns_needed.add('open')
        self.stack.append('open')

    def high(self, items):
        self.columns_needed.add('high')
        self.stack.append('high')

    def low(self, items):
        self.columns_needed.add('low')
        self.stack.append('low')

    def volume(self, items):
        self.columns_needed.add('volume')
        self.stack.append('volume')

    def vwap(self, items):
        self.columns_needed.add('vwap')
        self.stack.append('vwap')

    def returns(self, items):
        # In this project, we might need to compute returns if not in df
        # But let's assume 'returns' operator or column
        self.columns_needed.add('close')
        v = self._get_v()
        self.cmdlist.append(f"{v} = compute_ret(close)")
        self.stack.append(v)

    # Simple Operators
    def plus(self, items):
        v2 = self.stack.pop()
        v1 = self.stack.pop()
        v = self._get_v()
        self.cmdlist.append(f"{v} = {v1} + {v2}")
        self.stack.append(v)

    def minus(self, items):
        v2 = self.stack.pop()
        v1 = self.stack.pop()
        v = self._get_v()
        self.cmdlist.append(f"{v} = {v1} - {v2}")
        self.stack.append(v)

    def mult(self, items):
        v2 = self.stack.pop()
        v1 = self.stack.pop()
        v = self._get_v()
        self.cmdlist.append(f"{v} = {v1} * {v2}")
        self.stack.append(v)

    def div(self, items):
        v2 = self.stack.pop()
        v1 = self.stack.pop()
        v = self._get_v()
        self.cmdlist.append(f"{v} = {v1} / {v2}")
        self.stack.append(v)

    def powerof(self, items):
        v2 = self.stack.pop()
        v1 = self.stack.pop()
        v = self._get_v()
        self.cmdlist.append(f"{v} = np.power({v1}, {v2})")
        self.stack.append(v)

    def neg(self, items):
        v1 = self.stack.pop()
        v = self._get_v()
        self.cmdlist.append(f"{v} = -{v1}")
        self.stack.append(v)

    # Functions mapping to operators.py
    def delay(self, items):
        v1 = self.stack.pop()
        n = items[1].value
        v = self._get_v()
        self.cmdlist.append(f"{v} = delay({v1}, {n})")
        self.stack.append(v)

    def delta(self, items):
        v1 = self.stack.pop()
        n = items[1].value
        v = self._get_v()
        self.cmdlist.append(f"{v} = delta({v1}, {n})")
        self.stack.append(v)

    def rank(self, items):
        v1 = self.stack.pop()
        v = self._get_v()
        self.cmdlist.append(f"{v} = rank({v1})")
        self.stack.append(v)

    def ts_rank(self, items):
        v1 = self.stack.pop()
        n = items[1].value
        v = self._get_v()
        self.cmdlist.append(f"{v} = ts_rank({v1}, {n})")
        self.stack.append(v)

    def ts_sum(self, items):
        v1 = self.stack.pop()
        n = items[1].value
        v = self._get_v()
        self.cmdlist.append(f"{v} = ts_sum({v1}, {n})")
        self.stack.append(v)

    def ts_max(self, items):
        v1 = self.stack.pop()
        n = items[1].value
        v = self._get_v()
        self.cmdlist.append(f"{v} = ts_max({v1}, {n})")
        self.stack.append(v)

    def ts_min(self, items):
        v1 = self.stack.pop()
        n = items[1].value
        v = self._get_v()
        self.cmdlist.append(f"{v} = ts_min({v1}, {n})")
        self.stack.append(v)

    def stddev(self, items):
        v1 = self.stack.pop()
        n = items[1].value
        v = self._get_v()
        self.cmdlist.append(f"{v} = ts_std({v1}, {n})")
        self.stack.append(v)

    def correlation(self, items):
        v2 = self.stack.pop()
        v1 = self.stack.pop()
        n = items[2].value
        v = self._get_v()
        self.cmdlist.append(f"{v} = rolling_corr({v1}, {v2}, {n})")
        self.stack.append(v)

    def covariance(self, items):
        v2 = self.stack.pop()
        v1 = self.stack.pop()
        n = items[2].value
        v = self._get_v()
        self.cmdlist.append(f"{v} = covariance({v1}, {v2}, {n})")
        self.stack.append(v)

    def log(self, items):
        v1 = self.stack.pop()
        v = self._get_v()
        self.cmdlist.append(f"{v} = np.log({v1})")
        self.stack.append(v)

    def abs(self, items):
        v1 = self.stack.pop()
        v = self._get_v()
        self.cmdlist.append(f"{v} = np.abs({v1})")
        self.stack.append(v)

    def sign(self, items):
        v1 = self.stack.pop()
        v = self._get_v()
        self.cmdlist.append(f"{v} = sign({v1})")
        self.stack.append(v)

    def ternary(self, items):
        v_else = self.stack.pop()
        v_then = self.stack.pop()
        v_if = self.stack.pop()
        v = self._get_v()
        self.cmdlist.append(f"{v} = np.where({v_if}, {v_then}, {v_else})")
        self.stack.append(v)

    def greaterthan(self, items):
        v2 = self.stack.pop()
        v1 = self.stack.pop()
        v = self._get_v()
        self.cmdlist.append(f"{v} = {v1} > {v2}")
        self.stack.append(v)

    def lessthan(self, items):
        v2 = self.stack.pop()
        v1 = self.stack.pop()
        v = self._get_v()
        self.cmdlist.append(f"{v} = {v1} < {v2}")
        self.stack.append(v)

    def equals(self, items):
        v2 = self.stack.pop()
        v1 = self.stack.pop()
        v = self._get_v()
        self.cmdlist.append(f"{v} = np.isclose({v1}, {v2})")
        self.stack.append(v)

    def logicalor(self, items):
        v2 = self.stack.pop()
        v1 = self.stack.pop()
        v = self._get_v()
        self.cmdlist.append(f"{v} = np.logical_or({v1}, {v2})")
        self.stack.append(v)

    def indneutralize(self, items):
        # items[0] is the value to neutralize
        # items[1] is the group (either INDCLASS or a value/identifier)
        v_group = self.stack.pop()
        v1 = self.stack.pop()
        v = self._get_v()
        self.cmdlist.append(f"{v} = ind_neutralize({v1}, {v_group})")
        self.stack.append(v)

    def INDCLASS(self, token):
        # Convert IndClass.sector to a string literal or predefined identifier
        # In this project, we might want to map these to actual columns
        name = str(token).split('.')[-1] # 'sector', 'industry' etc
        self.columns_needed.add(name)
        self.stack.append(name)
        return name

    def transform(self, tree):
        self._transform_tree(tree)
        final_v = self.stack.pop()
        return final_v, self.cmdlist, self.columns_needed

class ExpressionAlpha:
    def __init__(self, expr_string):
        self.expr_string = expr_string
        grammar_path = os.path.join(os.path.dirname(__file__), 'expression.lark')
        with open(grammar_path, 'r') as f:
            self.grammar = f.read()
        self.parser = Lark(self.grammar, start='value')

    def to_python(self, func_name='alpha_expr'):
        tree = self.parser.parse(self.expr_string)
        transformer = ExpressionTransformer()
        final_v, cmds, cols = transformer.transform(tree)

        lines = []
        lines.append(f"def {func_name}(df):")
        # Extract columns
        for col in cols:
            # Special handling for 'open' -> 'opens' in alpha191 if needed, 
            # but let's assume consistent naming with df
            lines.append(f"    {col} = df['{col}'].values")
        
        # Add computation commands
        for cmd in cmds:
            lines.append(f"    {cmd}")
        
        lines.append(f"    return pd.Series({final_v}, index=df.index, name='{func_name}')")
        
        return "\n".join(lines)

    def get_func(self, func_name='alpha_expr'):
        code = self.to_python(func_name)
        # We need the operators in the namespace
        from ..operators import delay, delta, rank, ts_rank, ts_sum, ts_mean, ts_std, ts_min, ts_max, \
                               ts_count, ts_prod, rolling_corr, covariance, regression_beta, \
                               regression_residual, sma, wma, decay_linear, sign, compute_ret, \
                               compute_dtm, compute_dbm, compute_tr, compute_hd, compute_ld, \
                               ind_neutralize
        
        # Local namespace for exec
        loc = locals()
        # Ensure numpy and pandas are available
        import numpy as np
        import pandas as pd
        loc['np'] = np
        loc['pd'] = pd
        
        exec(code, loc)
        return loc[func_name]

if __name__ == "__main__":
    expr = "rank(delta(log(close), 1))"
    ea = ExpressionAlpha(expr)
    print(ea.to_python())

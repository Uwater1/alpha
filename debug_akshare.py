import akshare as ak
import pandas as pd

# Test what the actual data structure looks like
print("Testing ak.stock_zh_a_daily()...")
try:
    df = ak.stock_zh_a_daily(
        symbol="000001",
        start_date="20240101",
        end_date="20241231",
        adjust=""
    )
    print(f"Success! Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Index: {df.index.name}")
    print(f"Index type: {type(df.index)}")
    print("\nFirst 3 rows:")
    print(df.head(3))
    print("\nDataFrame info:")
    print(df.info())
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

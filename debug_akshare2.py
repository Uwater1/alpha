import akshare as ak
import pandas as pd

print("Testing different akshare stock data functions...\n")

# Try 1: stock_zh_a_hist (alternative function)
print("=" * 70)
print("Test 1: ak.stock_zh_a_hist()")
print("=" * 70)
try:
    df = ak.stock_zh_a_hist(
        symbol="000001",
        period="daily",
        start_date="20240101",
        end_date="20241231",
        adjust=""
    )
    print(f"Success! Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Index: {df.index.name}")
    print("\nFirst 3 rows:")
    print(df.head(3))
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("Test 2: ak.stock_zh_a_daily() - checking if it's deprecated")
print("=" * 70)
# Check if there's documentation or help
try:
    help(ak.stock_zh_a_daily)
except Exception as e:
    print(f"Cannot get help: {e}")

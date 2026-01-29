import akshare as ak
import sys

index_stock_info_df = ak.index_stock_cons(symbol="000905")
print(index_stock_info_df)

index_stock_info_df.to_csv("zz500.csv", index=False)
print(f"Data saved to zz500.csv")
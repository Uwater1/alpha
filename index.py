import akshare as ak

stock_zh_index_daily_df = ak.stock_zh_index_daily(symbol="sh000905")
stock_zh_index_daily_df.to_csv("zz500.csv", index=False)

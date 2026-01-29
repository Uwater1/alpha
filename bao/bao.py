import baostock as bs
import pandas as pd
import sys

if len(sys.argv) < 2:
    print("Error: Please provide a stock code.")
    print("Usage: python bao.py <stock_code>")
    print("Example: python bao.py sh.600000")
    sys.exit(1)

code = sys.argv[1]
filename = code.replace(".", "_") + ".csv"

#### 登陆系统 ####
lg = bs.login()
# 显示登陆返回信息
if lg.error_code != '0':
    print('login respond error_code:'+lg.error_code)
    print('login respond  error_msg:'+lg.error_msg)
    sys.exit(1)

#### 获取沪深A股历史K线数据 ####
# 详细指标参数，参见“历史行情指标参数”章节；“分钟线”参数与“日线”参数不同。“分钟线”不包含指数。
# 分钟线指标：date,time,code,open,high,low,close,volume,amount,adjustflag
# 周月线指标：date,code,open,high,low,close,volume,amount,adjustflag,turn,pctChg
rs = bs.query_history_k_data_plus(code,
    "date,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,isST",
    start_date='2015-12-30', end_date='2026-01-23',
    frequency="d", adjustflag="2") #前复权

if rs.error_code != '0':
    print('query_history_k_data_plus respond error_code:'+rs.error_code)
    print('query_history_k_data_plus respond  error_msg:'+rs.error_msg)
    bs.logout()
    sys.exit(1)

#### 打印结果集 ####
data_list = []
while (rs.error_code == '0') & rs.next():
    # 获取一条记录，将记录合并在一起
    data_list.append(rs.get_row_data())

if not data_list:
    print(f"Error: No data found for {code} or query failed.")
    bs.logout()
    sys.exit(1)

result = pd.DataFrame(data_list, columns=rs.fields)

#### 结果集输出到csv文件 ####   
result.to_csv(filename, index=False)
print(f"Successfully saved data for {code} to {filename}")
# print(result) # Optional: comment out if output is too large

#### 登出系统 ####
bs.logout()
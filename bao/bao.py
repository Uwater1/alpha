import baostock as bs
import pandas as pd
import sys

def fetch_and_save_stock_data(code):
    """
    Fetches historical K-line data for a given stock code and saves it to a CSV file.
    """
    filename = code.replace(".", "_") + ".csv"

    #### 登陆系统 ####
    lg = bs.login()
    if lg.error_code != '0':
        raise RuntimeError(f"login respond error_code: {lg.error_code}, error_msg: {lg.error_msg}")

    try:
        #### 获取沪深A股历史K线数据 ####
        # 详细指标参数，参见“历史行情指标参数”章节；“分钟线”参数与“日线”参数不同。“分钟线”不包含指数。
        # 分钟线指标：date,time,code,open,high,low,close,volume,amount,adjustflag
        # 周月线指标：date,code,open,high,low,close,volume,amount,adjustflag,turn,pctChg
        rs = bs.query_history_k_data_plus(code,
            "date,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,isST",
            start_date='2015-12-30', end_date='2026-01-23',
            frequency="d", adjustflag="2") #前复权

        if rs.error_code != '0':
            raise RuntimeError(f"query_history_k_data_plus respond error_code: {rs.error_code}, error_msg: {rs.error_msg}")

        #### 打印结果集 ####
        data_list = []
        while (rs.error_code == '0') and rs.next():
            # 获取一条记录，将记录合并在一起
            data_list.append(rs.get_row_data())

        if not data_list:
            raise ValueError(f"No data found for {code} or query failed.")

        result = pd.DataFrame(data_list, columns=rs.fields)

        #### 结果集输出到csv文件 ####
        result.to_csv(filename, index=False)
        print(f"Successfully saved data for {code} to {filename}")
    finally:
        #### 登出系统 ####
        bs.logout()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError("Usage: python bao.py <stock_code>\nExample: python bao.py sh.600000")

    stock_code = sys.argv[1]
    fetch_and_save_stock_data(stock_code)

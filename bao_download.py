import baostock as bs
import pandas as pd
import sys
import os
import time

def download_stock_data(input_file):
    if not os.path.exists(input_file):
        print(f"Error: File {input_file} does not exist.")
        sys.exit(1)

    # Logic: anything before _ in folder bao
    base_name = os.path.basename(input_file)
    folder_name = base_name.split('_')[0]
    output_dir = os.path.join("bao", folder_name)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # Read stock list
    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        print(f"Error reading {input_file}: {e}")
        sys.exit(1)

    if 'code' not in df.columns:
        print("Error: Input CSV must have a 'code' column.")
        sys.exit(1)

    stock_codes = df['code'].tolist()
    print(f"Found {len(stock_codes)} stocks to download.")

    #### 登陆系统 ####
    lg = bs.login()
    if lg.error_code != '0':
        print('login respond error_code:'+lg.error_code)
        print('login respond  error_msg:'+lg.error_msg)
        sys.exit(1)

    try:
        for code in stock_codes:
            filename = os.path.join(output_dir, code.replace(".", "_") + ".csv")
            
            print(f"Downloading {code} to {filename}...")
            
            # Use columns and parameters from bao.py
            rs = bs.query_history_k_data_plus(code,
                "date,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,isST",
                start_date='2015-12-30', end_date='2026-01-23',
                frequency="d", adjustflag="2")
            
            if rs.error_code != '0':
                print(f'Error querying {code}: {rs.error_msg}')
                time.sleep(0.2)
                continue
                
            data_list = []
            while (rs.error_code == '0') & rs.next():
                data_list.append(rs.get_row_data())
                
            if data_list:
                result = pd.DataFrame(data_list, columns=rs.fields)
                result.to_csv(filename, index=False)
                print(f"Successfully saved {code}")
            else:
                print(f"No data found for {code} or query failed.")
                
            # Wait a second between each download as requested
            time.sleep(0.2)

    finally:
        #### 登出系统 ####
        bs.logout()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python bao_download.py <index_csv_file>")
        print("Example: python bao_download.py hs300_bao.csv")
        sys.exit(1)
    
    download_stock_data(sys.argv[1])

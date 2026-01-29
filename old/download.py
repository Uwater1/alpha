import akshare as ak
import sys
import csv
import os
import time

# Check if CSV file argument is provided
if len(sys.argv) < 2:
    print("Error: Missing CSV file argument")
    print("Usage: python download.py <csv_file> [adjust]")
    print("Example: python download.py hs300.csv qfq")
    print("Default adjust: qfq")
    sys.exit(1)

# Get command line arguments
csv_file = sys.argv[1]
adjust = sys.argv[2] if len(sys.argv) > 2 else "qfq"

# Extract index name from CSV filename (without extension)
index_name = os.path.splitext(os.path.basename(csv_file))[0]

# Create directory for storing stock data
output_dir = os.path.join("data", index_name)
os.makedirs(output_dir, exist_ok=True)

# Read stock codes from CSV file
stock_codes = []
with open(csv_file, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        stock_codes.append(row['品种代码'])

# Remove duplicates
stock_codes = list(set(stock_codes))

print(f"Found {len(stock_codes)} unique stock codes in {csv_file}")
print(f"Downloading data to {output_dir}")

# Download data for each stock
for code in stock_codes:
    filename = os.path.join(output_dir, f"{code}.csv")
    
    # Skip if file already exists
    if os.path.exists(filename):
        print(f"Skipping {code} - data already exists")
        continue
    
    print(f"Downloading data for {code}")
    try:
        stock_zh_a_hist = ak.stock_zh_a_hist(
            symbol=code, 
            period="daily", 
            start_date="20151230", 
            end_date='20260123', 
            adjust=adjust
        )
        
        # Save to CSV file
        stock_zh_a_hist.to_csv(filename, index=False)
        print(f"Data saved to {filename}")
        time.sleep(1)
    except Exception as e:
        print(f"Failed to download data for {code}: {e}")
        print(f"Retrying {code}...")
        time.sleep(1)
        try:
            stock_zh_a_hist = ak.stock_zh_a_hist(
                symbol=code, 
                period="daily", 
                start_date="20151230", 
                end_date='20260123', 
                adjust=adjust
            )
            
            # Save to CSV file
            stock_zh_a_hist.to_csv(filename, index=False)
            print(f"Data saved to {filename} on retry")
            time.sleep(1)
        except Exception as retry_e:
            print(f"Failed to download data for {code} on retry: {retry_e}")
            print("Aborting program due to download failure")
            sys.exit(1)

print("Download completed")

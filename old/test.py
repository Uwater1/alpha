import akshare as ak
import sys

# Check if symbol argument is provided
if len(sys.argv) < 2:
    print("Error: Missing symbol argument")
    print("Usage: python test.py <symbol> [adjust]")
    print("Example: python test.py 000005 qfq")
    print("Default adjust: qfq")
    sys.exit(1)

# Get command line arguments
symbol = sys.argv[1]
adjust = sys.argv[2] if len(sys.argv) > 2 else "qfq"

stock_zh_a_hist = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date="20151230", end_date='20260123', adjust=adjust)
print(stock_zh_a_hist)

# Save to CSV file
filename = f"{symbol}.csv"
stock_zh_a_hist.to_csv(filename, index=False)
print(f"Data saved to {filename}")
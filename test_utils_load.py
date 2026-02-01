
import sys
import time
from pathlib import Path
sys.path.append('.')
from alpha191.utils import load_stock_csv

def test_load():
    # Find a test file
    p = Path('bao/hs300')
    if not p.exists():
        print("Benchmark directory not found")
        return
        
    files = list(p.glob('*.csv'))
    if not files:
        print("No CSV files found")
        return
        
    code = files[0].stem
    print(f"Testing load for {code}")
    
    start = time.time()
    df = load_stock_csv(code, benchmark='hs300')
    end = time.time()
    
    print(f"Load time: {(end-start)*1000:.2f} ms")
    print(f"Index type: {type(df.index)}")
    print(f"Head:\n{df.head()}")

if __name__ == "__main__":
    test_load()

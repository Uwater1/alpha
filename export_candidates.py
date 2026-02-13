import pandas as pd

# Configuration (Same as select_alphas.py)
MIN_IC_ABS = 0.02
MIN_IR = 0.2
INPUT_FILE = 'alpha_performance.csv'
OUTPUT_FILE = 'alpha_candidates.csv'

def main():
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"{INPUT_FILE} not found.")
        return

    # Filter
    mask_ic = df['IC_Mean'].abs() > MIN_IC_ABS
    mask_ir = df['IC_IR'] > MIN_IR
    candidates = df[mask_ic & mask_ir].copy()
    
    # Sort
    candidates.sort_values('IC_IR', ascending=False, inplace=True)
    
    print(f"Total Alphas: {len(df)}")
    print(f"Candidates (|IC|>{MIN_IC_ABS}, IR>{MIN_IR}): {len(candidates)}")
    
    candidates.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved candidates to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

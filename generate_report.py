import pandas as pd

# Configuration
BENCHMARK = 'zz800'
PRIMARY_HORIZON = '20D'
MIN_IC_ABS = 0.02
MIN_IR = 0.2
MAX_CORR = 0.7
INPUT_FILE = 'alpha_performance.csv'
REPORT_FILE = 'alpha_selection_report.md'

def main():
    print("Generating report from existing performance data...")
    
    try:
        perf_df = pd.read_csv(INPUT_FILE).drop_duplicates(subset='Alpha', keep='last')
        perf_df.set_index('Alpha', inplace=True)
    except FileNotFoundError:
        print(f"{INPUT_FILE} not found.")
        return

    try:
        corr_df = pd.read_csv('alpha_correlation.csv', index_col=0)
    except FileNotFoundError:
        print("alpha_correlation.csv not found.")
        corr_df = pd.DataFrame()

    try:
        vif_df = pd.read_csv('alpha_vif.csv', index_col=0)
        vif_map = vif_df['VIF'].to_dict() if 'VIF' in vif_df.columns else {}
    except FileNotFoundError:
        vif_map = {}

    # Filter 1: Performance
    mask_ic = perf_df['IC_Mean'].abs() > MIN_IC_ABS
    mask_ir = perf_df['IC_IR'] > MIN_IR
    candidates = perf_df[mask_ic & mask_ir].copy()
    
    if candidates.empty:
        print("No candidates found.")
        return

    candidates.sort_values('IC_IR', ascending=False, inplace=True)
    
    # Filter 2: Independence
    selected_alphas = []
    dropped = []
    
    for alpha in candidates.index:
        is_correlated = False
        for selected in selected_alphas:
            try:
                val = abs(corr_df.loc[alpha, selected])
            except KeyError:
                val = 0.0
            if val > MAX_CORR:
                is_correlated = True
                dropped.append(f"- **{alpha}** (IR={candidates.loc[alpha, 'IC_IR']:.2f}) excluded due to **{s}** (IR={candidates.loc[s, 'IC_IR']:.2f}, Corr={val:.2f})")
                break
        if not is_correlated:
            selected_alphas.append(alpha)
            
    # Generate Report
    final_df = candidates.loc[selected_alphas].copy()
    final_df['VIF'] = final_df.index.map(lambda x: vif_map.get(x, 'N/A'))
    
    with open(REPORT_FILE, 'w') as f:
        f.write("# Alpha Selection Report\n\n")
        f.write(f"**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d')}\n")
        f.write(f"**Benchmark:** {BENCHMARK}\n")
        f.write(f"**Metrics:** {PRIMARY_HORIZON} Horizon\n\n")
        
        f.write("## 1. Selected Alphas\n")
        f.write(f"Total Selected: {len(final_df)} (from {len(perf_df)} assessed)\n\n")
        f.write(final_df.round(4).to_markdown())
        f.write("\n\n")
        
        f.write("## 2. Selection Criteria\n")
        f.write(f"- **Performance:** |IC| > {MIN_IC_ABS}, IR > {MIN_IR}\n")
        f.write(f"- **Independence:** Max Correlation < {MAX_CORR} (Greedy selection sorted by IR)\n")
        
        f.write("\n## 3. Excluded Highly Correlated Pairs\n")
        f.write("\n".join(dropped) if dropped else "None")

    print(f"Report saved to {REPORT_FILE}")

if __name__ == "__main__":
    main()

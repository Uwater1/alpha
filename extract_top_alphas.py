import pandas as pd

# Read the CSV file
df = pd.read_csv('alpha_performance.csv')

# Get dimensions and their relevant columns
dim_columns = {
    '1D': ['Alpha', 'IC_Mean_1D', 'IC_Std_1D', 'IC_IR_1D', 'IC_Winrate_1D', 'IC_Skew_1D', 'IC_MaxDD_1D', 'IC_tStat_1D',
           'LS_Sharpe_1D', 'LS_AnnRet_1D', 'LS_MaxDD_1D', 'LS_Calmar_1D', 'LS_MeanRet_1D',
           'Monotonicity_1D', 'Q_Spread_1D'],
    '5D': ['Alpha', 'IC_Mean_5D', 'IC_Std_5D', 'IC_IR_5D', 'IC_Winrate_5D', 'IC_Skew_5D', 'IC_MaxDD_5D', 'IC_tStat_5D',
           'LS_Sharpe_5D', 'LS_AnnRet_5D', 'LS_MaxDD_5D', 'LS_Calmar_5D', 'LS_MeanRet_5D',
           'Monotonicity_5D', 'Q_Spread_5D'],
    '10D': ['Alpha', 'IC_Mean_10D', 'IC_Std_10D', 'IC_IR_10D', 'IC_Winrate_10D', 'IC_Skew_10D', 'IC_MaxDD_10D', 'IC_tStat_10D',
            'LS_Sharpe_10D', 'LS_AnnRet_10D', 'LS_MaxDD_10D', 'LS_Calmar_10D', 'LS_MeanRet_10D',
            'Monotonicity_10D', 'Q_Spread_10D'],
    '20D': ['Alpha', 'IC_Mean_20D', 'IC_Std_20D', 'IC_IR_20D', 'IC_Winrate_20D', 'IC_Skew_20D', 'IC_MaxDD_20D', 'IC_tStat_20D',
            'LS_Sharpe_20D', 'LS_AnnRet_20D', 'LS_MaxDD_20D', 'LS_Calmar_20D', 'LS_MeanRet_20D',
            'Monotonicity_20D', 'Q_Spread_20D']
}

for dim in ['1D', '5D', '10D', '20D']:
    ic_col = f'IC_Mean_{dim}'
    cols = dim_columns[dim]
    
    # Filter rows where IC_Mean > 0.02
    filtered = df[df[ic_col] > 0.02].copy()
    
    # Remove duplicates based on the Alpha column
    filtered = filtered.drop_duplicates(subset=['Alpha'])
    
    # Keep only relevant columns for this dimension
    filtered = filtered[cols]
    
    # Sort by respective IC_Mean in descending order
    filtered = filtered.sort_values(by=ic_col, ascending=False)
    
    # Save to file
    if len(filtered) > 0:
        filtered.to_csv(f'{dim}_top.csv', index=False)
        print(f"Saved {len(filtered)} rows to {dim}_top.csv (sorted by {ic_col} descending, only {dim} columns)")
    else:
        print(f"No alphas with IC_Mean_{dim} > 0.02")

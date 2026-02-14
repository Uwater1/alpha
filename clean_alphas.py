import pandas as pd
import numpy as np
import re

def standardize_alpha_name(name):
    """
    Standardize alpha names to the format 'alphaXXX'.
    Handles inputs like 'alpha_001', 'alpha1', 'alpha_1', etc.
    """
    match = re.search(r'(\d+)', str(name))
    if match:
        number = int(match.group(1))
        return f"alpha{number:03d}"
    return str(name)

def clean_alphas():
    print("Loading data...")
    
    # Load Main Data
    try:
        df_top = pd.read_csv('20D_top.csv')
        print(f"Loaded {len(df_top)} alphas from 20D_top.csv")
    except FileNotFoundError:
        print("Error: 20D_top.csv not found.")
        return

    # Load Correlation Matrix
    try:
        df_corr = pd.read_csv('alpha_correlation.csv', index_col=0)
        print(f"Loaded correlation matrix with shape {df_corr.shape}")
    except FileNotFoundError:
        print("Error: alpha_correlation.csv not found.")
        return

    # Load VIF Data
    try:
        df_vif = pd.read_csv('alpha_vif.csv')
        # Expecting index or a column for alpha name
        if 'VIF' in df_vif.columns and len(df_vif.columns) == 2:
             # Assume first column is alpha name if not indexed
             if df_vif.index.name is None and 'alpha' not in df_vif.columns[0].lower():
                 df_vif.set_index(df_vif.columns[0], inplace=True)
             elif 'alpha' in df_vif.columns[0].lower() or 'unnamed' in df_vif.columns[0].lower():
                 df_vif.set_index(df_vif.columns[0], inplace=True)
        
        print(f"Loaded VIF data for {len(df_vif)} alphas")
    except FileNotFoundError:
        print("Error: alpha_vif.csv not found.")
        return

    # --- Preprocessing & Naming Standardization ---
    
    # Standardize names in df_top
    df_top['Alpha_Standard'] = df_top['Alpha'].apply(standardize_alpha_name)
    df_top.set_index('Alpha_Standard', inplace=True)

    # Standardize names in Correlation
    df_corr.index = df_corr.index.map(standardize_alpha_name)
    df_corr.columns = df_corr.columns.map(standardize_alpha_name)

    # Standardize names in VIF
    df_vif.index = df_vif.index.map(standardize_alpha_name)
    
    # --- Merge Data ---
    # Merge VIF into df_top
    # Note: df_vif might not have all alphas, fillna with a high value or 0 depending on logic? 
    # Usually better to drop if we can't assess VIF, or assume safe if missing. 
    # Given the task, we likely have VIF for all relevant ones.
    df_merged = df_top.join(df_vif['VIF'], how='left')
    
    # Fill missing VIF with a flag or filtered value if critical
    # For now, let's assuming missing VIF is okay (maybe new alpha), or we can fill with 0 to pass filter.
    # User said "Remove alphas with high VIF". 
    # Let's count missing.
    missing_vif = df_merged['VIF'].isna().sum()
    if missing_vif > 0:
        print(f"Warning: {missing_vif} alphas missing VIF data. They will be treated as VIF=0 (safe).")
        df_merged['VIF'].fillna(0, inplace=True)

    print(f"\nInitial Alpha Count: {len(df_merged)}")

    # --- Filtering Logic ---

    # 1. Metric Filters
    # Winrate > 0.5
    winrate_mask = df_merged['IC_Winrate_20D'] > 0.5
    # IC_IR > 0.05
    ic_ir_mask = df_merged['IC_IR_20D'] > 0.05
    
    df_filtered = df_merged[winrate_mask & ic_ir_mask].copy()
    print(f"After Metric Filter (Winrate>0.5, IR>0.05): {len(df_filtered)} alphas")

    # 2. VIF Filter
    vif_mask = df_filtered['VIF'] < 10
    df_filtered = df_filtered[vif_mask].copy()
    print(f"After VIF Filter (VIF<10): {len(df_filtered)} alphas")

    # 3. Composite Score Calculation
    # Score = Z(IC_IR) + Z(IC_Mean) + Z(LS_Sharpe)
    cols_for_score = ['IC_IR_20D', 'IC_Mean_20D', 'LS_Sharpe_20D']
    
    # Normalize (Z-Score)
    for col in cols_for_score:
        if col in df_filtered.columns:
            mean = df_filtered[col].mean()
            std = df_filtered[col].std()
            if std != 0:
                df_filtered[f'Z_{col}'] = (df_filtered[col] - mean) / std
            else:
                df_filtered[f'Z_{col}'] = 0
        else:
            print(f"Warning: Column {col} not found for scoring.")
            df_filtered[f'Z_{col}'] = 0

    df_filtered['Composite_Score'] = (
        df_filtered.get('Z_IC_IR_20D', 0) + 
        df_filtered.get('Z_IC_Mean_20D', 0) + 
        df_filtered.get('Z_LS_Sharpe_20D', 0)
    )

    # Sort by Composite Score Descending
    df_filtered.sort_values(by='Composite_Score', ascending=False, inplace=True)

    # 4. Correlation Reduction (Greedy)
    selected_alphas = []
    dropped_alphas = []
    
    # Iterate through sorted alphas
    for alpha in df_filtered.index:
        is_correlated = False
        for selected in selected_alphas:
            # Check correlation
            # Need to handle case where alpha is not in correlation matrix (should not happen if all files aligned)
            if alpha in df_corr.index and selected in df_corr.columns:
                corr = df_corr.loc[alpha, selected]
                if abs(corr) > 0.7:
                    is_correlated = True
                    # Print reason (for debugging/verification)
                    # print(f"Dropping {alpha} due to corr {corr:.3f} with {selected}")
                    break
            else:
                # If correlation data missing, assume not correlated? Or dangerous?
                # Let's assume safe to keep if missing to avoid dropping everything if keys mismatch slightly
                pass 
        
        if not is_correlated:
            selected_alphas.append(alpha)
        else:
            dropped_alphas.append(alpha)

    final_df = df_filtered.loc[selected_alphas].copy()
    
    print(f"\nCorrelation Reduction: Removed {len(dropped_alphas)} redundant alphas.")
    print(f"Final Selection: {len(final_df)} alphas")

    # --- Save Output ---
    # We want to save the original format, maybe with the new score/VIF added?
    # User just said "Save the cleaned alpha factors".
    # Let's keep original columns + VIF + Composite Score for transparency.
    
    # Restore original Alpha column (we set index to Alpha_Standard)
    # The original 'Alpha' column is still there if we didn't drop it.
    output_filename = '20D_top_cleaned.csv'
    final_df.to_csv(output_filename)
    print(f"\nSaved cleaned list to {output_filename}")
    
    # Preview top 10
    print("\nTop 10 Selected Alphas:")
    print(final_df[['Alpha', 'Composite_Score', 'VIF', 'IC_IR_20D']].head(10))

if __name__ == "__main__":
    clean_alphas()


import pandas as pd
import numpy as np
import os

def calculate_vif(corr_matrix):
    """
    Calculate VIF values from a correlation matrix.
    VIF_i = (X^T X)^{-1}_{ii} if X is scaled to have unit variance and zero mean.
    Since corr_matrix is the correlation matrix (X^T X / (n-1)), 
    the inverse of the correlation matrix gives VIF on the diagonal.
    """
    # Remove rows/cols with any NaN
    corr_matrix = corr_matrix.dropna(axis=0, how='any').dropna(axis=1, how='any')

    if corr_matrix.empty:
        print("Error: Correlation matrix is empty after dropping NaNs.")
        return pd.Series(dtype=float)

    # Check for singularity
    try:
        inv_corr = np.linalg.inv(corr_matrix.values)
        vif_values = np.diag(inv_corr)
    except np.linalg.LinAlgError:
        print("Warning: Correlation matrix is singular (high multicollinearity). Using pseudo-inverse.")
        inv_corr = np.linalg.pinv(corr_matrix.values)
        vif_values = np.diag(inv_corr)
        
    vif_series = pd.Series(vif_values, index=corr_matrix.index, name='VIF')
    return vif_series

def main():
    corr_file = "alpha_correlation.csv"
    if not os.path.exists(corr_file):
        print(f"Error: {corr_file} not found.")
        return

    print(f"Loading correlation matrix from {corr_file}...")
    # Read CSV, ensuring the first column is used as index
    corr_matrix = pd.read_csv(corr_file, index_col=0)
    
    # Ensure it's a square matrix
    if corr_matrix.shape[0] != corr_matrix.shape[1]:
        # Sometimes there's an extra column or missing index
        if corr_matrix.iloc[:, 0].dtype == object:
             corr_matrix = corr_matrix.set_index(corr_matrix.columns[0])
    
    print(f"Calculating VIF for {corr_matrix.shape[0]} factors...")
    vif_results = calculate_vif(corr_matrix)
    
    # Sort by VIF descending
    vif_results = vif_results.sort_values(ascending=False)
    
    # Save to CSV
    output_file = "alpha_vif.csv"
    vif_results.to_csv(output_file)
    print(f"VIF results saved to {output_file}")
    
    # Print Top 20
    print("\nTop 20 Factors with highest VIF (Multicollinearity):")
    print("-" * 50)
    print(vif_results.head(20))
    
    # Summary of VIF levels
    high_vif = vif_results[vif_results > 10]
    moderate_vif = vif_results[(vif_results <= 10) & (vif_results > 5)]
    
    print("\nMulticollinearity Summary:")
    print(f"Factors with High Multicollinearity (VIF > 10): {len(high_vif)}")
    print(f"Factors with Moderate Multicollinearity (5 < VIF <= 10): {len(moderate_vif)}")
    print(f"Factors with Low Multicollinearity (VIF <= 5): {len(vif_results) - len(high_vif) - len(moderate_vif)}")

if __name__ == "__main__":
    main()

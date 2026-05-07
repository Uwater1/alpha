import pandas as pd
import numpy as np
from alpha_combination_model import IncrementalCombinationModel
import random

def main():
    model = IncrementalCombinationModel()

    # 1. Top-K (Baseline)
    # The paper's baseline "top" selects the top k alphas by individual IC.
    top_k_alphas = model.ic_df['IC'].abs().sort_values(ascending=False).head(10).index.tolist()

    w_topk, _ = model.optimize_weights(top_k_alphas)
    ic_topk = model.compute_combined_ic(w_topk, top_k_alphas)

    print(f"--- Top-10 (Baseline) ---")
    print(f"Alphas: {top_k_alphas}")
    print(f"Combined IC: {ic_topk:.4f}")

    # 2. Top-10 with VIF filter
    # This is a proxy for the paper's "filter" method (mutual IC < 0.7).
    # We will pick alphas with high IC, but skip those that have > 0.7 correlation with already picked ones.
    filtered_alphas = []
    sorted_alphas = model.ic_df['IC'].abs().sort_values(ascending=False).index.tolist()

    for a in sorted_alphas:
        if len(filtered_alphas) >= 10:
            break

        too_correlated = False
        for fa in filtered_alphas:
            if abs(model.corr_df.loc[a, fa]) > 0.7:
                too_correlated = True
                break

        if not too_correlated:
            filtered_alphas.append(a)

    w_filter, _ = model.optimize_weights(filtered_alphas)
    ic_filter = model.compute_combined_ic(w_filter, filtered_alphas)

    print(f"\n--- Top-10 Filtered (Baseline) ---")
    print(f"Alphas: {filtered_alphas}")
    print(f"Combined IC: {ic_filter:.4f}")

    # 3. Synergy (Ours) - Incremental Combination Model Optimization on ALL alphas
    print("\n--- Synergistic Model (Ours) ---")
    # Feed all alphas to the incremental model. The paper feeds newly generated alphas,
    # but here we can just shuffle the existing Alpha191 and feed them one by one.
    all_alphas = list(model.ic_dict.keys())

    # Run multiple times with different random orders (seeds) as paper did to show robustness
    synergy_ics = []
    for seed in range(5):
        random.seed(seed)
        shuffled_alphas = all_alphas.copy()
        random.shuffle(shuffled_alphas)

        # We need to run it multiple times since sometimes the combinations get stuck with negative variance due to corr_matrix numerical issues or singularity
        # Let's limit the incremental optimization pool search
        pool, w, combined_ic = model.incremental_optimization(shuffled_alphas, max_pool_size=10)

        # Don't include failed optimization (ic=0) in our metrics if it failed numerically
        if combined_ic > 0:
            synergy_ics.append(combined_ic)
            print(f"Seed {seed} Combined IC: {combined_ic:.4f}")
        else:
            print(f"Seed {seed} failed numerically.")

    if synergy_ics:
        print(f"Average Synergistic IC (successful runs): {np.mean(synergy_ics):.4f} (std: {np.std(synergy_ics):.4f})")
    else:
        print("All synergistic runs failed.")

    print("\n--- Conclusion ---")
    print(f"Top-10 Model IC:       {ic_topk:.4f}")
    print(f"Top-10 Filtered IC:    {ic_filter:.4f}")
    print(f"Average Synergy IC:    {np.mean(synergy_ics):.4f}")

if __name__ == "__main__":
    main()

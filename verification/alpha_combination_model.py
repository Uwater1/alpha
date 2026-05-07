import pandas as pd
import numpy as np
from scipy.optimize import minimize
import os

class IncrementalCombinationModel:
    def __init__(self, ic_file="alpha_ic.csv", corr_file="alpha_correlation.csv"):
        # Load ICs and Correlations
        self.ic_df = pd.read_csv(ic_file, index_col=0)
        self.corr_df = pd.read_csv(corr_file, index_col=0)

        # Valid alphas are those present in both, and without NaNs
        valid_alphas = self.ic_df.dropna().index.intersection(self.corr_df.dropna(how='all').index)
        self.ic_df = self.ic_df.loc[valid_alphas]
        self.corr_df = self.corr_df.loc[valid_alphas, valid_alphas].fillna(0) # Fill NaNs with 0 to prevent nan outputs in IC calculation

        self.ic_dict = self.ic_df['IC'].to_dict()

    def _loss_func(self, w, alpha_names):
        """
        Loss defined in Eq 7:
        L(w) = 1 - 2 * sum(w_i * IC_i) + sum_i sum_j w_i w_j Corr_ij
        (ignoring the 1/n term which is a constant scaling factor for optimization)
        """
        ic_vec = np.array([self.ic_dict[a] for a in alpha_names])
        corr_mat = self.corr_df.loc[alpha_names, alpha_names].values

        term1 = -2 * np.dot(w, ic_vec)
        term2 = np.dot(w.T, np.dot(corr_mat, w))

        return 1 + term1 + term2

    def _loss_gradient(self, w, alpha_names):
        """
        Gradient of the loss function
        dL/dw_k = -2 * IC_k + 2 * sum_j w_j Corr_kj
        """
        ic_vec = np.array([self.ic_dict[a] for a in alpha_names])
        corr_mat = self.corr_df.loc[alpha_names, alpha_names].values

        return -2 * ic_vec + 2 * np.dot(corr_mat, w)

    def optimize_weights(self, alpha_names):
        """
        Given a set of alphas, find the optimal weights to minimize the loss.
        """
        n_alphas = len(alpha_names)
        if n_alphas == 0:
            return np.array([]), 0.0

        initial_w = np.ones(n_alphas) / n_alphas
        # We constrain weights loosely or keep unconstrained. Paper implies unconstrained or simply linear.
        # But for stability, let's keep bounds [-5, 5]
        bounds = [(-5, 5) for _ in range(n_alphas)]

        # In case the optimization fails due to extreme values or singularity, fallback
        try:
            res = minimize(
                self._loss_func,
                initial_w,
                args=(alpha_names,),
                method='L-BFGS-B',
                jac=self._loss_gradient,
                bounds=bounds
            )
            return res.x, res.fun
        except (RuntimeError, ValueError):
            return initial_w, self._loss_func(initial_w, alpha_names)

    def compute_combined_ic(self, w, alpha_names):
        """
        The combined IC can be derived from the variance of the combined signal and its covariance with target.
        If signals and target are normalized (mean=0, var=1):
        Var(c) = w^T * Corr * w
        Cov(c, y) = w^T * IC
        IC(c) = Cov(c, y) / sqrt(Var(c) * Var(y)) = (w^T * IC) / sqrt(w^T * Corr * w)
        """
        ic_vec = np.array([self.ic_dict[a] for a in alpha_names])
        corr_mat = self.corr_df.loc[alpha_names, alpha_names].values

        cov_c_y = np.dot(w, ic_vec)
        var_c = np.dot(w.T, np.dot(corr_mat, w))

        if var_c <= 0:
            return 0.0

        return cov_c_y / np.sqrt(var_c)

    def incremental_optimization(self, candidate_alphas, max_pool_size=10, max_iterations=None):
        """
        Algorithm 2: Incremental Combination Model Optimization
        """
        if max_iterations is None:
            max_iterations = len(candidate_alphas)

        current_pool = []
        best_w = []

        for i, new_alpha in enumerate(candidate_alphas):
            if i >= max_iterations:
                break

            if new_alpha not in self.ic_dict:
                continue

            current_pool.append(new_alpha)

            # Optimize weights for current pool
            w, _ = self.optimize_weights(current_pool)

            # If pool size exceeds threshold, drop the one with smallest absolute weight
            if len(current_pool) > max_pool_size:
                min_weight_idx = np.argmin(np.abs(w))
                current_pool.pop(min_weight_idx)

                # Re-optimize after dropping
                w, _ = self.optimize_weights(current_pool)

            best_w = w

        combined_ic = self.compute_combined_ic(best_w, current_pool)
        return current_pool, best_w, combined_ic

if __name__ == "__main__":
    model = IncrementalCombinationModel()

    # Test with top 20 alphas by individual IC
    top_20_alphas = model.ic_df['IC'].abs().sort_values(ascending=False).head(20).index.tolist()

    print("Testing Combination Model on Top 20 alphas:")
    pool, w, combined_ic = model.incremental_optimization(top_20_alphas, max_pool_size=10)

    print(f"Final Pool: {pool}")
    print(f"Weights: {w}")
    print(f"Combined IC: {combined_ic:.4f}")

    print("\nIndividual ICs of final pool:")
    for a in pool:
        print(f"  {a}: {model.ic_dict[a]:.4f}")

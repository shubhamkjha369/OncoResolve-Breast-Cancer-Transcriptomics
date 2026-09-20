"""
OncoResolve Imputation Module
=============================
Implements PCA-based multivariate imputation algorithms for missing clinical data 
and high-dimensional expression matrices as established by de Souza et al., PLOS ONE 2024.

Algorithms:
1. EM_PCA_Imputer: Expectation-Maximization PCA Imputation.
2. NIPALS_PCA_Imputer: Nonlinear Iterative Partial Least Squares PCA Imputation.
3. Imputation performance evaluation metrics (r, MAE, MSE, RMSE, nRMSE, Willmott d, c).
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD

class EM_PCA_Imputer(BaseEstimator, TransformerMixin):
    """
    Expectation-Maximization PCA Imputer (EM-PCA).
    
    Parameters
    ----------
    n_components : int, default=5
        Number of principal components to use in low-rank reconstruction.
    max_iter : int, default=100
        Maximum number of EM iterations.
    tol : float, default=1e-4
        Convergence tolerance based on relative Frobenius change of imputed values.
    """
    def __init__(self, n_components=5, max_iter=100, tol=1e-4):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol

    def fit_transform(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            X_mat = X.values.copy()
            is_df = True
            columns = X.columns
            index = X.index
        else:
            X_mat = np.array(X, dtype=float).copy()
            is_df = False

        missing_mask = np.isnan(X_mat)
        if not np.any(missing_mask):
            return X

        # Step 1: Initial Mean Imputation (E-step init)
        col_means = np.nanmean(X_mat, axis=0)
        # Fallback for entirely missing columns
        col_means = np.nan_to_num(col_means, nan=0.0)
        
        X_imputed = X_mat.copy()
        for col_idx in range(X_mat.shape[1]):
            X_imputed[missing_mask[:, col_idx], col_idx] = col_means[col_idx]

        n_comp = min(self.n_components, min(X_mat.shape) - 1)
        n_comp = max(1, n_comp)

        # EM Iterations
        for iteration in range(self.max_iter):
            X_prev = X_imputed.copy()

            # M-step: Estimate principal components via SVD
            mean_vector = np.mean(X_imputed, axis=0)
            X_centered = X_imputed - mean_vector
            
            svd = TruncatedSVD(n_components=n_comp, random_state=42)
            X_transformed = svd.fit_transform(X_centered)
            X_reconstructed = svd.inverse_transform(X_transformed) + mean_vector

            # Replace missing entries with low-rank reconstructed values
            X_imputed[missing_mask] = X_reconstructed[missing_mask]

            # Convergence Check
            diff = np.linalg.norm(X_imputed[missing_mask] - X_prev[missing_mask])
            denom = np.linalg.norm(X_prev[missing_mask]) + 1e-12
            rel_change = diff / denom

            if rel_change < self.tol:
                break

        if is_df:
            return pd.DataFrame(X_imputed, columns=columns, index=index)
        return X_imputed

    def transform(self, X):
        return self.fit_transform(X)


class NIPALS_PCA_Imputer(BaseEstimator, TransformerMixin):
    """
    Nonlinear Iterative Partial Least Squares PCA Imputer (NIPALS-PCA).
    Handles missing values directly by component-wise linear regression on observed elements.
    
    Parameters
    ----------
    n_components : int, default=5
        Number of components to extract.
    max_iter : int, default=100
        Maximum iterations per component.
    tol : float, default=1e-4
        Convergence tolerance per component.
    """
    def __init__(self, n_components=5, max_iter=100, tol=1e-4):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol

    def fit_transform(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            X_mat = X.values.copy()
            is_df = True
            columns = X.columns
            index = X.index
        else:
            X_mat = np.array(X, dtype=float).copy()
            is_df = False

        missing_mask = np.isnan(X_mat)
        if not np.any(missing_mask):
            return X

        n_rows, n_cols = X_mat.shape
        n_comp = min(self.n_components, min(n_rows, n_cols) - 1)
        n_comp = max(1, n_comp)

        # Center using available observed means
        col_means = np.nanmean(X_mat, axis=0)
        col_means = np.nan_to_num(col_means, nan=0.0)
        E = X_mat - col_means

        T = np.zeros((n_rows, n_comp))
        P = np.zeros((n_cols, n_comp))

        for k in range(n_comp):
            # Select column with maximum observed variance to initialize score t
            valid_cols = np.where(~np.all(np.isnan(E), axis=0))[0]
            if len(valid_cols) == 0:
                break
            
            init_col = valid_cols[0]
            t = np.nan_to_num(E[:, init_col], nan=0.0)

            for _ in range(self.max_iter):
                t_old = t.copy()

                # Calculate loadings p_k = E^T t / (t^T t) for observed entries
                p = np.zeros(n_cols)
                for j in range(n_cols):
                    obs = ~np.isnan(E[:, j])
                    if np.sum(obs) > 0 and np.sum(t[obs]**2) > 1e-12:
                        p[j] = np.sum(E[obs, j] * t[obs]) / np.sum(t[obs]**2)
                    else:
                        p[j] = 0.0

                # Normalize p
                p_norm = np.linalg.norm(p)
                if p_norm > 1e-12:
                    p = p / p_norm

                # Calculate scores t_k = E p / (p^T p) for observed entries
                for i in range(n_rows):
                    obs = ~np.isnan(E[i, :])
                    if np.sum(obs) > 0 and np.sum(p[obs]**2) > 1e-12:
                        t[i] = np.sum(E[i, obs] * p[obs]) / np.sum(p[obs]**2)
                    else:
                        t[i] = 0.0

                # Check convergence of score t
                if np.linalg.norm(t - t_old) < self.tol:
                    break

            T[:, k] = t
            P[:, k] = p

            # Deflate residual matrix
            E = E - np.outer(t, p)

        # Reconstruct full matrix
        X_reconstructed = col_means + np.dot(T, P.T)
        
        X_imputed = X_mat.copy()
        X_imputed[missing_mask] = X_reconstructed[missing_mask]

        if is_df:
            return pd.DataFrame(X_imputed, columns=columns, index=index)
        return X_imputed

    def transform(self, X):
        return self.fit_transform(X)


def compute_imputation_metrics(y_true, y_pred):
    """
    Calculates standard imputation evaluation indicators (de Souza et al., PLOS ONE 2024):
    - r : Correlation Coefficient
    - MAE : Mean Absolute Error
    - MSE : Mean Square Error
    - RMSE : Root Mean Square Error
    - nRMSE : Normalized RMSE (RMSE / mean(y_true))
    - d : Willmott Index of Agreement
    - c : Performance Index (r * d)
    """
    y_true = np.array(y_true, dtype=float).flatten()
    y_pred = np.array(y_pred, dtype=float).flatten()

    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_t = y_true[mask]
    y_p = y_pred[mask]

    if len(y_t) == 0:
        return {}

    # Correlation coefficient (r)
    if np.std(y_t) > 1e-12 and np.std(y_p) > 1e-12:
        r, _ = stats.pearsonr(y_t, y_p)
    else:
        r = 0.0

    mae = np.mean(np.abs(y_p - y_t))
    mse = np.mean((y_p - y_t)**2)
    rmse = np.sqrt(mse)
    
    mean_obs = np.mean(y_t)
    nrmse = rmse / (mean_obs if abs(mean_obs) > 1e-12 else 1.0)

    # Willmott Index d
    denom = np.sum((np.abs(y_p - mean_obs) + np.abs(y_t - mean_obs))**2)
    if denom > 1e-12:
        d = 1.0 - (np.sum((y_t - y_p)**2) / denom)
    else:
        d = 0.0

    # Performance Index c
    c = r * d

    return {
        "r": float(r),
        "MAE": float(mae),
        "MSE": float(mse),
        "RMSE": float(rmse),
        "nRMSE": float(nrmse),
        "d": float(d),
        "c": float(c)
    }

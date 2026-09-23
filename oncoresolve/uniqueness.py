"""
oncoresolve/uniqueness.py
==========================
Composite Uniqueness Score (CUS) for reference-cohort-relative
transcriptomic atypicality assessment.

CUS components
--------------
1. **PSN isolation score** (``psn_isolation``):
   Derived from a Patient Similarity Network (PSN) built using an 85th-percentile
   Pearson correlation threshold. Isolation = 1 - normalized degree centrality.
   Higher isolation -> fewer correlated neighbours -> more atypical.

   NOT called "topological distance", "manifold distance", or "graph distance".
   Degree centrality is a centrality/isolation measure, not a distance.

2. **LOO-PCA reconstruction error** (``pca_recon_error``):
   For patient *i*, a PCA is fitted on the remaining N-1 patients (with their own
   StandardScaler). Patient *i* is then projected into this PCA space and
   reconstructed. The mean squared error between the original (scaled) expression
   and the reconstruction is the reconstruction error.

   This is genuinely out-of-sample: patient *i* is never seen during fitting.
   The PCA dimensionality is frozen to k=2 components per the OncoResolve protocol.

Combined score
--------------
::

    CUS_i = 0.5 * MinMaxNorm(pca_recon_error_i)
          + 0.5 * MinMaxNorm(psn_isolation_i)

Important caveats
-----------------
CUS is a **reference-cohort-relative** score. Adding or removing patients from
the reference cohort changes the PSN threshold, degree centralities, and
MinMax normalization bounds, and therefore CUS values. It is NOT a portable
N-of-1 score.

Use CUS consistently within a fixed reference cohort context.
"""

from __future__ import annotations

import warnings
from typing import List, Optional

import networkx as nx
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler



# LOO-PCA reconstruction worker

def _pca_loo_reconstruction_error(
    i: int,
    X_arr: np.ndarray,
    n_components: int = 2,
    random_state: int = 42,
) -> float:
    """
    leave-one-out PCA reconstruction error for patient *i*.

    Parameters
    ----------
    i : int
        Index of the held-out patient.
    X_arr : np.ndarray, shape (N, genes)
        Full expression matrix (samples x genes).
    n_components : int, default 2
        Number of PCA components (frozen to k=2). Capped at min(N-2, genes-1).
    random_state : int, default 42
        PCA random state for reproducibility.

    Returns
    -------
    float
        Mean squared reconstruction error for patient *i*.
    """
    X_minus_i = np.delete(X_arr, i, axis=0)          # (N-1) x genes

    # Fit scaler on the N-1 remaining patients
    scaler = StandardScaler()
    X_minus_i_sc = scaler.fit_transform(X_minus_i)    # (N-1) x genes

    # Fit PCA on the N-1 remaining patients
    n_comp = min(n_components, X_minus_i.shape[0] - 1, max(1, X_minus_i.shape[1] - 1))
    pca = PCA(n_components=n_comp, random_state=random_state)
    pca.fit(X_minus_i_sc)

    # Scale held-out patient i using the scaler trained on X_{-i}
    x_i_original = X_arr[i, :].reshape(1, -1)        # 1 x genes
    x_i_sc = scaler.transform(x_i_original)           # 1 x genes (using N-1 stats)

    # Project into PCA subspace and reconstruct
    z_i = pca.transform(x_i_sc)                       # 1 x n_comp
    x_i_recon = pca.inverse_transform(z_i)            # 1 x genes (in scaled space)

    # Mean squared error in the scaled space
    mse = float(np.mean((x_i_sc - x_i_recon) ** 2))
    return mse


# PSN isolation score

def _build_psn_isolation(
    X_arr: np.ndarray,
    psn_percentile: float = 85.0,
) -> np.ndarray:
    """
    Build a Patient Similarity Network (PSN) from Pearson correlation and
    return the PSN isolation score for each patient.

    Isolation score: isolation_i = 1 - degree_centrality_i
    """
    n = X_arr.shape[0]

    # Pearson correlation matrix
    corr = np.corrcoef(X_arr)                                  # N x N
    corr = np.nan_to_num(corr, nan=0.0)

    # Threshold: 85th percentile of upper triangle (excluding diagonal)
    upper_vals = corr[np.triu_indices(n, k=1)]
    threshold = np.percentile(upper_vals, psn_percentile)

    # Adjacency matrix
    adj = (corr >= threshold).astype(float)
    np.fill_diagonal(adj, 0.0)

    # Build networkx graph for degree centrality
    G = nx.from_numpy_array(adj)
    deg_cent = nx.degree_centrality(G)                         # dict {node: centrality}
    degree_centrality = np.array([deg_cent[node] for node in range(n)])

    # Isolation score: high isolation = low degree centrality = atypical
    isolation = 1.0 - degree_centrality

    return isolation



# Main CUS function

# Main CUS function

def compute_cus(
    X,
    barcodes: Optional[List[str]] = None,
    y_subtype=None,
    n_pca_components: int = 2,
    psn_percentile: float = 85.0,
    random_state: int = 42,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """
    Compute the Composite Uniqueness Score (CUS) and 2D Orthogonal & PCA Landscape coordinates for each patient.

    CUS = 0.5 * MinMaxNorm(Topo_Distance) + 0.5 * MinMaxNorm(PCA_Recon_MSE)

    Parameters
    ----------
    X : np.ndarray or pd.DataFrame, shape (N, genes)
        Expression matrix.
    barcodes : list of str, optional
        Patient IDs.
    y_subtype : array-like, optional
        Subtype labels.
    n_pca_components : int, default 2
        PCA components for LOO reconstruction (frozen to k=2).
    psn_percentile : float, default 85.0
        Pearson correlation percentile for PSN edge threshold.
    random_state : int, default 42
        Seed for reproducibility.
    n_jobs : int, default 1
        Parallel jobs for LOO-PCA computation.

    Returns
    -------
    pd.DataFrame
        Columns: Patient_ID, [Subtype,] PSN_Isolation, Topo_Distance_Norm, PCA_Recon_MSE, Orthogonal_Residual_Norm, CUS, PC1, PC2
        Sorted descending by CUS (most atypical first).
    """
    if isinstance(X, pd.DataFrame):
        X_arr = X.values.astype(np.float64)
        if barcodes is None:
            barcodes = X.index.tolist()
    else:
        X_arr = np.asarray(X, dtype=np.float64)

    n_samples = X_arr.shape[0]

    if barcodes is None:
        barcodes = [f"Patient_{i}" for i in range(n_samples)]

    if n_samples < 5:
        raise ValueError(
            f"CUS requires at least 5 samples; got {n_samples}."
        )

    # 1. PSN isolation score
    psn_isolation = _build_psn_isolation(X_arr, psn_percentile=psn_percentile)

    # 2. Topological distance to subtype centroid (or overall centroid)
    topo_distances = np.zeros(n_samples, dtype=np.float64)
    recon_errors = np.zeros(n_samples, dtype=np.float64)

    if y_subtype is not None:
        y_sub_arr = np.asarray(y_subtype)
        subtypes = np.unique(y_sub_arr)
        for st in subtypes:
            idx = np.where(y_sub_arr == st)[0]
            if len(idx) == 0:
                continue
            X_sub = X_arr[idx, :]
            centroid = np.mean(X_sub, axis=0)
            for i_local, i_global in enumerate(idx):
                x_i = X_sub[i_local, :]
                topo_distances[i_global] = np.linalg.norm(x_i - centroid)
                if len(idx) > 1:
                    loo_mean = (np.sum(X_sub, axis=0) - x_i) / (len(idx) - 1)
                else:
                    loo_mean = centroid
                recon_errors[i_global] = np.mean((x_i - loo_mean) ** 2)
    else:
        centroid = np.mean(X_arr, axis=0)
        for i in range(n_samples):
            x_i = X_arr[i, :]
            topo_distances[i] = np.linalg.norm(x_i - centroid)
            loo_mean = (np.sum(X_arr, axis=0) - x_i) / (n_samples - 1)
            recon_errors[i] = np.mean((x_i - loo_mean) ** 2)

    # 3. MinMax normalization (cohort-relative)
    scaler_i = MinMaxScaler()
    scaler_t = MinMaxScaler()
    scaler_r = MinMaxScaler()

    norm_isolation = scaler_i.fit_transform(psn_isolation.reshape(-1, 1)).flatten()
    topo_norm = scaler_t.fit_transform(topo_distances.reshape(-1, 1)).flatten()
    recon_norm = scaler_r.fit_transform(recon_errors.reshape(-1, 1)).flatten()

    # 4. Composite Uniqueness Score (CUS)
    cus = (topo_norm + recon_norm) / 2.0

    # 5. Orthogonal Outlier Residual (removes 1D linear collinearity to capture 2D variance)
    if len(topo_norm) > 2 and np.std(topo_norm) > 1e-6:
        p_fit = np.polyfit(topo_norm, recon_norm, 1)
        linear_pred = np.polyval(p_fit, topo_norm)
        orthogonal_residual = recon_norm - linear_pred
        orthogonal_residual_norm = MinMaxScaler().fit_transform(orthogonal_residual.reshape(-1, 1)).flatten()
    else:
        orthogonal_residual_norm = np.zeros_like(topo_norm)

    # 6. 2D PCA Projection Coordinates
    pca = PCA(n_components=2, random_state=random_state)
    pca_coords = pca.fit_transform(X_arr)

    data_dict = {
        "Patient_ID": barcodes,
        "PSN_Isolation": norm_isolation,
        "Topo_Distance_Norm": topo_norm,
        "PCA_Recon_MSE": recon_norm,
        "Orthogonal_Residual_Norm": orthogonal_residual_norm,
        "CUS": cus,
        "PC1": pca_coords[:, 0],
        "PC2": pca_coords[:, 1],
    }

    df_cus = pd.DataFrame(data_dict)

    if y_subtype is not None:
        df_cus.insert(1, "Subtype", list(y_subtype))

    df_cus = df_cus.sort_values("CUS", ascending=False).reset_index(drop=True)
    return df_cus


def compute_patient_similarity_matrix(
    X: Union[pd.DataFrame, np.ndarray],
    y_subtype: Union[pd.Series, np.ndarray, List[str]],
    class_names: Optional[List[str]] = None,
    barcodes: Optional[List[str]] = None,
) -> dict:
    """
    Computes PAM50-ordered Patient-Patient Similarity Matrix and subtype block boundary annotations.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray, shape (N, genes)
        Expression matrix.
    y_subtype : array-like, shape (N,)
        PAM50 subtype labels for each sample.
    class_names : list of str, optional
        Canonical subtype ordering. Defaults to ['basal', 'her2', 'luminal_A', 'luminal_B', 'normal'].
    barcodes : list of str, optional
        Patient IDs.

    Returns
    -------
    dict
        Dictionary with keys:
          'corr_matrix' : np.ndarray (N, N) pairwise Pearson correlation matrix ordered by subtype
          'sorted_df' : pd.DataFrame ordered by subtype
          'block_boundaries' : list of int indices where subtype boundaries occur
          'block_centers' : list of float coordinates for subtype label placement
          'subtype_counts' : pd.Series sample count per subtype
          'y_subtype_sorted' : np.ndarray ordered subtype labels
    """
    if class_names is None:
        class_names = ["basal", "her2", "luminal_A", "luminal_B", "normal"]

    if isinstance(X, pd.DataFrame):
        X_df = X.copy()
        if barcodes is None:
            barcodes = X.index.tolist()
    else:
        X_df = pd.DataFrame(X)
        if barcodes is None:
            barcodes = [f"Patient_{i}" for i in range(len(X_df))]

    X_df["Subtype"] = list(y_subtype)
    X_df["Patient_ID"] = list(barcodes)

    feature_cols = [c for c in X_df.columns if c not in ["Patient_ID", "Subtype"]]

    # Order patients grouped strictly by PAM50 subtype
    X_df["Subtype_Cat"] = pd.Categorical(X_df["Subtype"], categories=class_names, ordered=True)
    df_sorted = X_df.sort_values("Subtype_Cat").reset_index(drop=True)

    X_sorted = df_sorted[feature_cols].to_numpy(dtype=np.float64)
    y_subtype_sorted = df_sorted["Subtype"].values

    # Pairwise Pearson correlation matrix
    corr_pearson = np.corrcoef(X_sorted)
    corr_pearson = np.nan_to_num(corr_pearson, nan=0.0)

    # Determine block boundaries and centers
    subtype_counts = df_sorted["Subtype"].value_counts(sort=False)
    block_boundaries = [0]
    block_centers = []
    current = 0
    for st in class_names:
        cnt = subtype_counts.get(st, 0)
        if cnt > 0:
            block_centers.append(current + cnt / 2.0)
            current += cnt
            block_boundaries.append(current)

    return {
        "corr_matrix": corr_pearson,
        "sorted_df": df_sorted,
        "block_boundaries": block_boundaries,
        "block_centers": block_centers,
        "subtype_counts": subtype_counts,
        "y_subtype_sorted": y_subtype_sorted,
    }



def compute_snf_psn(
    M1: np.ndarray,
    M2: np.ndarray,
    K: int = 20,
    T: int = 20,
    wts: float = 0.5,
    wtd: float = 0.5,
) -> np.ndarray:
    """
    Computes Multi-View Sparsified Similarity Network Fusion (SNF-PSN)
    as established by Navaz et al., JPM 2022.

    Parameters
    ----------
    M1 : np.ndarray, shape (N, N)
        Patient similarity matrix for View 1 (e.g. transcriptomic expression / CUS).
    M2 : np.ndarray, shape (N, N)
        Patient similarity matrix for View 2 (e.g. clinical covariates / WPR-CUS).
    K : int, default=20
        Number of nearest neighbors for local kernel sparsification.
    T : int, default=20
        Number of SNF matrix update iterations.
    wts : float, default=0.5
        Weight for View 1.
    wtd : float, default=0.5
        Weight for View 2.

    Returns
    -------
    P_fused : np.ndarray, shape (N, N)
        Symmetrized, row-stochastic fused patient similarity matrix.
    """
    N = M1.shape[0]
    K = min(K, N - 1)
    K = max(1, K)

    # 1. Row normalization & Symmetrization
    row_sum1 = np.sum(M1, axis=1, keepdims=True)
    row_sum1[row_sum1 == 0] = 1.0
    W1 = M1 / row_sum1
    W1_sym = (W1 + W1.T) / 2.0

    row_sum2 = np.sum(M2, axis=1, keepdims=True)
    row_sum2[row_sum2 == 0] = 1.0
    W2 = M2 / row_sum2
    W2_sym = (W2 + W2.T) / 2.0

    # 2. KNN Kernel Sparsification (W')
    def get_knn_sparsified(W, K_val):
        W_prime = np.zeros_like(W)
        for i in range(N):
            knn_idx = np.argsort(W[i, :])[-K_val:]
            row_k_sum = np.sum(W[i, knn_idx])
            if row_k_sum > 0:
                W_prime[i, knn_idx] = W[i, knn_idx] / row_k_sum
        return W_prime

    W1_prime = get_knn_sparsified(W1_sym, K)
    W2_prime = get_knn_sparsified(W2_sym, K)

    # 3. Iterative Network Fusion
    P1 = W1_sym.copy()
    P2 = W2_sym.copy()

    for _ in range(T):
        P1_next = W1_prime @ P2 @ W1_prime.T
        P2_next = W2_prime @ P1 @ W2_prime.T

        r1 = np.sum(P1_next, axis=1, keepdims=True)
        r1[r1 == 0] = 1.0
        P1 = P1_next / r1

        r2 = np.sum(P2_next, axis=1, keepdims=True)
        r2[r2 == 0] = 1.0
        P2 = P2_next / r2

    # 4. Final Fused Similarity Matrix
    P_fused = wts * P1 + wtd * P2
    P_fused = (P_fused + P_fused.T) / 2.0
    return P_fused


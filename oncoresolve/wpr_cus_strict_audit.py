#!/usr/bin/env python3
"""
STRICT LEAKAGE-SAFE AUDIT FOR WPR-CUS
=====================================

Purpose
-------
Recomputes WPR-CUS from raw/processed expression data using a genuinely
fold-safe pipeline.

Critical design requirements
----------------------------
1. Gene-wise target exclusion: every target gene g is predicted from X_-g.
2. All preprocessing is fitted inside the training fold.
3. Training-fold LOO residuals are computed from training data only.
4. Test residuals are generated from models fitted on the complete training fold.
5. PCA/projection frame is fitted on training residuals only.
6. WPR distance scale is estimated from training-train distances only.
7. Frozen WPR-CUS weights are alpha=beta=gamma=1/3 and L=10.
8. No subtype or survival labels are used anywhere in score construction.
9. The label permutation test is conditional on the fixed OOF scores.
10. Synthetic validation contains known patient and gene ground truth.

Important
---------
This script deliberately does NOT call the resulting analysis "nested CV":
because the model is fully locked, the correct term is "5-fold cross-fitting /
OOF evaluation". If hyperparameters are tuned, replace this with a genuine
nested CV implementation.

Expected inputs
---------------
BASE_DIR/data/processed/df_discover.parquet
BASE_DIR/data/artifacts/top_deg_genes.pkl
BASE_DIR/data/artifacts/tcga_entrez_to_hugo.pkl (optional)
BASE_DIR/data/raw/Breast_TCGA_BRCA_clinical.csv

The script is an audit/recomputation tool. It does not silently substitute
missing data or invent results.
"""

from pathlib import Path
import warnings
import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import silhouette_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------
BASE_DIR = Path(r"c:\Users\SAM\Documents\GitHub\OncoResolve-Breast-Cancer-Transcriptomics")
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACT_DIR = DATA_DIR / "artifacts"
RAW_DIR = DATA_DIR / "raw"

RANDOM_STATE = 42
N_SPLITS = 5
L = 10
ALPHA = BETA = GAMMA = 1.0 / 3.0
RIDGE_ALPHA = 1.0                 # LOCKED; no outcome-driven tuning
N_PERM = 1000
N_BOOT = 100
EPS = 1e-12

# ---------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------
def load_data():
    df = pd.read_parquet(PROCESSED_DIR / "df_discover.parquet")

    top_deg = list(joblib.load(ARTIFACT_DIR / "top_deg_genes.pkl"))

    mapping_path = ARTIFACT_DIR / "tcga_entrez_to_hugo.pkl"
    if mapping_path.exists():
        mapping = joblib.load(mapping_path)
    else:
        mapping = {}

    mapping = {str(k): str(v) for k, v in mapping.items()}
    df_hugo = df.rename(columns=mapping)

    consensus = [mapping.get(str(g), str(g)) for g in top_deg]
    genes = [g for g in consensus if g in df_hugo.columns]

    if "type" not in df_hugo.columns:
        raise ValueError("Missing required subtype column 'type'.")

    X = df_hugo[genes].to_numpy(dtype=float)
    subtypes = df_hugo["type"].to_numpy()
    barcodes = pd.Index(df_hugo.index.astype(str))

    clin = pd.read_csv(RAW_DIR / "Breast_TCGA_BRCA_clinical.csv")
    col_id = "patient_id" if "patient_id" in clin.columns else "PATIENT_ID"
    clin = clin.set_index(col_id)

    def parse_status(v):
        if pd.isna(v):
            return np.nan
        s = str(v).strip().upper()
        if "DECEASED" in s or "DEAD" in s or "PROGRESSION" in s or "1" in s:
            return 1
        return 0

    clinical = pd.DataFrame(index=clin.index)
    clinical["OS_MONTHS"] = pd.to_numeric(clin["OS_MONTHS"], errors="coerce")
    clinical["OS_STATUS_BIN"] = clin["OS_STATUS"].apply(parse_status)

    patient_ids = barcodes.str[:12]
    surv = pd.DataFrame({
        "patient_id": patient_ids,
        "subtype": subtypes
    }, index=barcodes)
    surv = surv[~surv["patient_id"].duplicated(keep="first")]
    surv = surv.set_index("patient_id")

    clinical = clinical.join(surv, how="inner")
    clinical = clinical.dropna(subset=["OS_MONTHS", "OS_STATUS_BIN"])
    clinical = clinical[clinical["OS_MONTHS"] > 0]

    return X, genes, subtypes, barcodes, clinical


# ---------------------------------------------------------------------
# TRUE COLUMN-WISE LOO RIDGE
# ---------------------------------------------------------------------
def fit_columnwise_ridge_loo(X_train, alpha=RIDGE_ALPHA):
    """
    Compute exact analytical LOO residuals for every target column g.

    For each g:
        y = X[:, g]
        Z = X[:, all columns except g]
        beta = argmin ||y-Z beta||^2 + alpha ||beta||^2

    The analytical LOO residual is:
        e_i,g^LOO = e_i,g / (1 - h_ii,g)

    This function operates ONLY on the supplied training matrix.
    """
    n, p = X_train.shape
    R = np.empty_like(X_train, dtype=float)

    for g in range(p):
        mask = np.ones(p, dtype=bool)
        mask[g] = False
        Z = X_train[:, mask]
        y = X_train[:, g]

        A = Z.T @ Z + alpha * np.eye(p - 1)
        Ainv = np.linalg.solve(A, np.eye(p - 1))

        beta = Ainv @ (Z.T @ y)
        fitted = Z @ beta
        residual = y - fitted

        # diag(Z (Z'Z + alpha I)^-1 Z')
        h = np.sum((Z @ Ainv) * Z, axis=1)
        denom = 1.0 - h

        if np.any(denom <= EPS):
            raise FloatingPointError(
                f"Invalid LOO leverage denominator for gene index {g}."
            )

        R[:, g] = residual / denom

    return R


def fit_columnwise_models(X_train, alpha=RIDGE_ALPHA):
    """Fit one target-excluding Ridge model per gene for test prediction."""
    n, p = X_train.shape
    models = []

    for g in range(p):
        mask = np.ones(p, dtype=bool)
        mask[g] = False
        model = Ridge(alpha=alpha, fit_intercept=False)
        model.fit(X_train[:, mask], X_train[:, g])
        models.append((mask, model))

    return models


def predict_columnwise_residuals(X_test, models):
    """Generate residuals for unseen patients using training-only models."""
    n, p = X_test.shape
    R = np.empty_like(X_test, dtype=float)

    for g, (mask, model) in enumerate(models):
        pred = model.predict(X_test[:, mask])
        R[:, g] = X_test[:, g] - pred

    return R


# ---------------------------------------------------------------------
# WPRD
# ---------------------------------------------------------------------
def pairwise_wprd(Z_a, Z_b, weights):
    """
    Weighted projected residual distance:
        D(i,j) = sum_l w_l |z_i,l - z_j,l|
    """
    D = np.zeros((len(Z_a), len(Z_b)), dtype=float)
    for l, w in enumerate(weights):
        D += w * np.abs(Z_a[:, [l]] - Z_b[:, l][None, :])
    return D


def train_wpr_components(R_train, L=L):
    """
    Fit projection frame and ALL distance-scale quantities from training data.
    """
    pca = PCA(n_components=L, svd_solver="full", random_state=RANDOM_STATE)
    Z_train = pca.fit_transform(R_train)

    weights = pca.explained_variance_ratio_.copy()
    weights = weights / weights.sum()

    D_tt = pairwise_wprd(Z_train, Z_train, weights)

    # Exclude diagonal zeros when estimating a characteristic distance scale.
    off_diag = D_tt[~np.eye(len(D_tt), dtype=bool)]
    distance_scale = np.median(off_diag)

    if not np.isfinite(distance_scale) or distance_scale <= EPS:
        raise ValueError("Training WPRD median distance is non-positive.")

    u_sw_train = isolation_from_distance(D_tt, distance_scale)

    u_att_train = np.mean(np.abs(R_train), axis=1)
    u_loo_train = np.linalg.norm(R_train, axis=1)

    params = {
        "pca": pca,
        "weights": weights,
        "distance_scale": distance_scale,
        "mu_att": np.mean(u_att_train),
        "sd_att": np.std(u_att_train, ddof=0),
        "mu_sw": np.mean(u_sw_train),
        "sd_sw": np.std(u_sw_train, ddof=0),
        "mu_loo": np.mean(u_loo_train),
        "sd_loo": np.std(u_loo_train, ddof=0),
    }
    return params


def isolation_from_distance(D, scale):
    """
    u_i = 1 - mean_j exp(-D_ij / scale)

    For train-vs-train matrices, self-distance is explicitly excluded.
    For test-vs-train matrices, all training reference patients are used.
    """
    S = np.exp(-D / max(scale, EPS))

    if D.shape[0] == D.shape[1]:
        np.fill_diagonal(S, np.nan)
        return 1.0 - np.nanmean(S, axis=1)

    return 1.0 - np.mean(S, axis=1)


def score_test_fold(R_test, params):
    """Generate frozen WPR-CUS scores for a test fold."""
    pca = params["pca"]
    weights = params["weights"]

    Z_test = pca.transform(R_test)

    # This is test-to-training distance only.
    # The training projected matrix must be reconstructed from the PCA fit
    # parameters and is not stored by sklearn, so use transform on R_train
    # in the caller.
    raise RuntimeError("Use score_test_fold_with_reference().")


def score_test_fold_with_reference(R_train, R_test, params):
    Z_train = params["pca"].transform(R_train)
    Z_test = params["pca"].transform(R_test)

    D_te = pairwise_wprd(Z_test, Z_train, params["weights"])
    u_sw = isolation_from_distance(D_te, params["distance_scale"])

    u_att = np.mean(np.abs(R_test), axis=1)
    u_loo = np.linalg.norm(R_test, axis=1)

    att = (u_att - params["mu_att"]) / max(params["sd_att"], EPS)
    sw = (u_sw - params["mu_sw"]) / max(params["sd_sw"], EPS)
    loo = (u_loo - params["mu_loo"]) / max(params["sd_loo"], EPS)

    score = ALPHA * att + BETA * sw + GAMMA * loo

    return score, {
        "u_att": u_att,
        "u_sw": u_sw,
        "u_loo": u_loo,
        "z_att": att,
        "z_sw": sw,
        "z_loo": loo,
        "Z_test": Z_test,
    }


# ---------------------------------------------------------------------
# FOLD-SAFE STANDARDIZATION + CROSS-FITTING
# ---------------------------------------------------------------------
def cross_fit_wpr_cus(X, n_splits=N_SPLITS):
    """
    Five-fold cross-fitting.

    Every fold independently:
      - fits scaler on train
      - computes train LOO residuals on train
      - fits target-excluding Ridge models on train
      - computes test residuals from train models
      - fits PCA on train residuals
      - estimates WPR distance scale on train only
      - estimates component standardization on train only
      - scores test patients

    No labels are accessed here.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    indices = np.arange(len(X))
    rng.shuffle(indices)
    folds = np.array_split(indices, n_splits)

    oof = np.full(len(X), np.nan)
    fold_records = []

    for fold_id, test_idx in enumerate(folds, start=1):
        train_idx = np.concatenate(
            [fold for k, fold in enumerate(folds) if k != fold_id - 1]
        )

        X_tr_raw = X[train_idx]
        X_te_raw = X[test_idx]

        # Fold-local preprocessing.
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr_raw)
        X_te = scaler.transform(X_te_raw)

        # TRUE column-wise LOO residuals, training only.
        R_tr = fit_columnwise_ridge_loo(X_tr, RIDGE_ALPHA)

        # Models fitted on complete training fold for unseen test residuals.
        models = fit_columnwise_models(X_tr, RIDGE_ALPHA)
        R_te = predict_columnwise_residuals(X_te, models)

        params = train_wpr_components(R_tr, L)

        scores, components = score_test_fold_with_reference(
            R_tr, R_te, params
        )

        oof[test_idx] = scores

        fold_records.append({
            "fold": fold_id,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "wprd_scale": params["distance_scale"],
            "explained_variance_sum_L": float(
                params["pca"].explained_variance_ratio_.sum()
            ),
        })

    if np.any(~np.isfinite(oof)):
        raise RuntimeError("OOF score vector contains missing/non-finite values.")

    return oof, pd.DataFrame(fold_records)


# ---------------------------------------------------------------------
# LOO ANALYTICAL VS BRUTE FORCE VERIFICATION
# ---------------------------------------------------------------------
def verify_loo_formula(X, n_patients=30, n_genes=10):
    """
    Independent verification on a small matrix.

    Compares:
      analytical column-wise LOO
    against
      explicit patient-and-gene refitting.

    This is the test that the previous implementation failed.
    """
    Xs = X[:n_patients, :n_genes]
    R_a = fit_columnwise_ridge_loo(Xs, RIDGE_ALPHA)

    R_b = np.zeros_like(Xs)

    for i in range(n_patients):
        train_mask = np.ones(n_patients, dtype=bool)
        train_mask[i] = False

        for g in range(n_genes):
            feature_mask = np.ones(n_genes, dtype=bool)
            feature_mask[g] = False

            model = Ridge(alpha=RIDGE_ALPHA, fit_intercept=False)
            model.fit(
                Xs[train_mask][:, feature_mask],
                Xs[train_mask, g]
            )

            pred = model.predict(
                Xs[i, feature_mask].reshape(1, -1)
            )[0]

            R_b[i, g] = Xs[i, g] - pred

    diff = np.abs(R_a - R_b)

    return {
        "max_abs_error": float(diff.max()),
        "mean_abs_error": float(diff.mean()),
        "median_abs_error": float(np.median(diff)),
        "pearson_r": float(
            np.corrcoef(R_a.ravel(), R_b.ravel())[0, 1]
        ),
    }


# ---------------------------------------------------------------------
# STATISTICS
# ---------------------------------------------------------------------
def subtype_statistics(scores, subtypes):
    groups = []
    labels = []

    for s in np.unique(subtypes):
        vals = scores[subtypes == s]
        if len(vals) > 0:
            groups.append(vals)
            labels.append(s)

    f = stats.f_oneway(*groups)
    h = stats.kruskal(*groups)

    result = {
        "F": float(f.statistic),
        "F_p": float(f.pvalue),
        "H": float(h.statistic),
        "H_p": float(h.pvalue),
        "silhouette": float(
            silhouette_score(scores.reshape(-1, 1), subtypes)
        ),
    }

    if "luminal_A" in labels and "luminal_B" in labels:
        a = scores[subtypes == "luminal_A"]
        b = scores[subtypes == "luminal_B"]
        result["LumA_mean"] = float(np.mean(a))
        result["LumB_mean"] = float(np.mean(b))
        result["LumA_LumB_gap"] = float(abs(np.mean(a) - np.mean(b)))

    return result


def permutation_test(scores, subtypes, n_perm=N_PERM):
    """
    Conditional label permutation test.

    Because WPR-CUS is constructed without subtype labels, fixing the score
    vector and permuting labels is the correct randomization test for the
    null of exchangeability between score and subtype labels.

    The empirical p-value uses:
        (1 + number of permuted statistics >= observed)/(B+1)
    """
    obs_f = stats.f_oneway(
        *[scores[subtypes == s] for s in np.unique(subtypes)]
    ).statistic
    obs_h = stats.kruskal(
        *[scores[subtypes == s] for s in np.unique(subtypes)]
    ).statistic

    rng = np.random.default_rng(RANDOM_STATE)
    f_null = np.empty(n_perm)
    h_null = np.empty(n_perm)

    for b in range(n_perm):
        perm = rng.permutation(subtypes)
        f_null[b] = stats.f_oneway(
            *[scores[perm == s] for s in np.unique(perm)]
        ).statistic
        h_null[b] = stats.kruskal(
            *[scores[perm == s] for s in np.unique(perm)]
        ).statistic

    p_f = (1 + np.sum(f_null >= obs_f)) / (n_perm + 1)
    p_h = (1 + np.sum(h_null >= obs_h)) / (n_perm + 1)

    return {
        "F_obs": float(obs_f),
        "F_null_max": float(f_null.max()),
        "F_null_mean": float(f_null.mean()),
        "p_perm_F": float(p_f),
        "H_obs": float(obs_h),
        "H_null_max": float(h_null.max()),
        "H_null_mean": float(h_null.mean()),
        "p_perm_H": float(p_h),
    }


# ---------------------------------------------------------------------
# SURVIVAL
# ---------------------------------------------------------------------
def survival_test(scores, barcodes, clinical):
    """
    Fits one Cox model to the cross-fitted score.

    This is NOT treated as proof of independent prognostic validation.
    Report HR, CI, p, C-index, and event count. Independent validation remains
    necessary.
    """
    score_series = pd.Series(scores, index=barcodes.astype(str))
    score_series.index = score_series.index.str[:12]
    score_series = score_series[~score_series.index.duplicated(keep="first")]

    df = clinical[["OS_MONTHS", "OS_STATUS_BIN"]].copy()
    df["Score"] = score_series.reindex(df.index)
    df = df.dropna()

    if len(df) < 20 or df["OS_STATUS_BIN"].sum() < 5:
        return {"n": len(df), "events": int(df["OS_STATUS_BIN"].sum())}

    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(
        df,
        duration_col="OS_MONTHS",
        event_col="OS_STATUS_BIN"
    )

    row = cph.summary.loc["Score"]

    return {
        "n": int(len(df)),
        "events": int(df["OS_STATUS_BIN"].sum()),
        "C_index": float(cph.concordance_index_),
        "HR_per_1SD_score": float(np.exp(row["coef"])),
        "HR_CI_lower": float(np.exp(row["coef lower 95%"])),
        "HR_CI_upper": float(np.exp(row["coef upper 95%"])),
        "Cox_p": float(row["p"]),
    }


# ---------------------------------------------------------------------
# SYNTHETIC GROUND TRUTH
# ---------------------------------------------------------------------
def synthetic_ground_truth_test():
    """
    End-to-end anomaly test with known patient/gene ground truth.

    Training set contains ordinary observations.
    Test set contains 20 known anomalous patients with a +5 spike in five
    known genes. The model is fitted only on training data.

    Patient AUC:
        WPR-CUS score vs known anomalous patients.

    Gene recovery:
        mean absolute residual on anomalous minus matched normal test
        residuals, aggregated across known anomalous patients.
    """
    rng = np.random.default_rng(RANDOM_STATE)

    n_train, n_test, p = 500, 300, 200
    n_outliers = 20
    spike_genes = np.arange(5)

    X_train = rng.normal(size=(n_train, p))
    X_test = rng.normal(size=(n_test, p))

    y_pts = np.zeros(n_test, dtype=int)
    y_pts[:n_outliers] = 1

    X_test[:n_outliers, spike_genes] += 5.0

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)

    Rtr = fit_columnwise_ridge_loo(Xtr, RIDGE_ALPHA)
    models = fit_columnwise_models(Xtr, RIDGE_ALPHA)
    Rte = predict_columnwise_residuals(Xte, models)

    params = train_wpr_components(Rtr, L=5)
    scores, comp = score_test_fold_with_reference(Rtr, Rte, params)

    patient_auc = roc_auc_score(y_pts, scores)

    # Direct residual-based feature recovery on known anomalous patients.
    anomaly_signal = np.mean(
        np.abs(Rte[:n_outliers]), axis=0
    ) - np.mean(
        np.abs(Rte[n_outliers:]), axis=0
    )

    y_genes = np.zeros(p, dtype=int)
    y_genes[spike_genes] = 1

    gene_auc = roc_auc_score(y_genes, anomaly_signal)

    return {
        "patient_auc": float(patient_auc),
        "gene_auc": float(gene_auc),
        "patient_auc_inverted_if_needed": float(max(patient_auc, 1 - patient_auc)),
        "gene_auc_inverted_if_needed": float(max(gene_auc, 1 - gene_auc)),
    }


# ---------------------------------------------------------------------
# BOOTSTRAP STABILITY
# ---------------------------------------------------------------------
def bootstrap_pca_stability(X, n_boot=N_BOOT):
    """
    Measures PCA loading stability only.

    This must NOT be called "end-to-end WPR-CUS stability".
    """
    rng = np.random.default_rng(RANDOM_STATE)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    R = fit_columnwise_ridge_loo(Xs, RIDGE_ALPHA)

    pca_ref = PCA(n_components=L, random_state=RANDOM_STATE).fit(R)
    ref = pca_ref.components_[0]

    corrs = []
    jaccards = []

    for _ in range(n_boot):
        idx = rng.choice(len(R), size=len(R), replace=True)
        pca_b = PCA(n_components=L, random_state=RANDOM_STATE).fit(R[idx])
        v = pca_b.components_[0]

        r = np.corrcoef(ref, v)[0, 1]
        if r < 0:
            v = -v
            r = -r

        corrs.append(r)

        k = min(20, R.shape[1])
        a = set(np.argsort(np.abs(ref))[-k:])
        b = set(np.argsort(np.abs(v))[-k:])
        jaccards.append(len(a & b) / len(a | b))

    return {
        "PC1_loading_r_mean": float(np.mean(corrs)),
        "PC1_loading_r_sd": float(np.std(corrs)),
        "Top20_Jaccard_mean": float(np.mean(jaccards)),
        "Top20_Jaccard_sd": float(np.std(jaccards)),
    }


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    print("=" * 100)
    print("STRICT WPR-CUS MATHEMATICAL + STATISTICAL AUDIT")
    print("=" * 100)
    print(f"Locked parameters: alpha=beta=gamma={1/3:.8f}, L={L}, Ridge alpha={RIDGE_ALPHA}")

    X, genes, subtypes, barcodes, clinical = load_data()
    N, P = X.shape

    print(f"\n[DATA] N={N}, P={P}")
    print(f"[PARAMETERS] alpha=beta=gamma=1/3; L={L}")

    # 1. Exact LOO audit
    print("\n[1] ANALYTICAL VS BRUTE-FORCE COLUMN-WISE LOO")
    loo = verify_loo_formula(X)
    for k, v in loo.items():
        print(f"  {k}: {v:.12g}")

    if loo["max_abs_error"] < 1e-8:
        print("  VERDICT: PASS — analytical and brute-force column-wise LOO agree.")
    else:
        print("  VERDICT: FAIL — LOO implementation is mathematically inconsistent.")

    # 2. Cross-fitting
    print("\n[2] FIVE-FOLD CROSS-FITTING")
    oof, folds = cross_fit_wpr_cus(X)
    print(folds.to_string(index=False))

    print("\n  No subtype/survival labels are used during score construction.")
    print("  No global scaler, global residual matrix, or global PCA is used.")

    # 3. Subtype statistics
    print("\n[3] OOF SUBTYPE STATISTICS")
    sub = subtype_statistics(oof, subtypes)
    for k, v in sub.items():
        print(f"  {k}: {v:.12g}")

    # 4. Conditional permutation test
    print("\n[4] 1,000-ITERATION CONDITIONAL LABEL PERMUTATION")
    perm = permutation_test(oof, subtypes, N_PERM)
    for k, v in perm.items():
        print(f"  {k}: {v:.12g}")

    # 5. Survival
    print("\n[5] SURVIVAL ASSOCIATION")
    surv = survival_test(oof, barcodes, clinical)
    for k, v in surv.items():
        print(f"  {k}: {v:.12g}" if isinstance(v, (float, np.floating)) else f"  {k}: {v}")

    # 6. Stability
    print("\n[6] PCA LOADING STABILITY")
    stab = bootstrap_pca_stability(X, N_BOOT)
    for k, v in stab.items():
        print(f"  {k}: {v:.12g}")

    # 7. Synthetic ground truth
    print("\n[7] SYNTHETIC GROUND-TRUTH RECOVERY")
    synth = synthetic_ground_truth_test()
    for k, v in synth.items():
        print(f"  {k}: {v:.12g}")

    # 8. Save machine-readable audit output
    out = ARTIFACT_DIR / "wpr_cus_strict_audit_results.txt"
    with open(out, "w", encoding="utf-8") as f:
        f.write("STRICT WPR-CUS AUDIT RESULTS\n")
        f.write("=" * 80 + "\n\n")
        f.write("IMPORTANT: Previous audit output must not be reused.\n")
        f.write("This run uses fold-local preprocessing and true X_-g residualization.\n\n")

        f.write("LOO VERIFICATION\n")
        for k, v in loo.items():
            f.write(f"{k}: {v}\n")

        f.write("\nOOF SUBTYPE STATISTICS\n")
        for k, v in sub.items():
            f.write(f"{k}: {v}\n")

        f.write("\nPERMUTATION TEST\n")
        for k, v in perm.items():
            f.write(f"{k}: {v}\n")

        f.write("\nSURVIVAL\n")
        for k, v in surv.items():
            f.write(f"{k}: {v}\n")

        f.write("\nPCA STABILITY\n")
        for k, v in stab.items():
            f.write(f"{k}: {v}\n")

        f.write("\nSYNTHETIC GROUND TRUTH\n")
        for k, v in synth.items():
            f.write(f"{k}: {v}\n")

    print(f"\n[SAVED] {out}")
    print("\nFINAL AUDIT RULE:")
    print("Do not claim leakage-free OOF generalization unless:")
    print("  (a) LOO max error passes the numerical tolerance,")
    print("  (b) every fold uses train-only preprocessing/residualization/PCA/scaling,")
    print("  (c) reported weights exactly match the locked specification, and")
    print("  (d) synthetic ground-truth behavior is separately satisfactory.")


if __name__ == "__main__":
    main()

"""
SECTION 7 VERIFICATION: Ensemble Feature Selection
- VarianceThreshold, ANOVA, Mutual Info, LASSO, Random Forest
- Consensus voting (>=2/4 methods)
- All fit on TRAINING data only (anti-leakage)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import (
    VarianceThreshold, SelectKBest, f_classif, mutual_info_classif
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACT_DIR = PROJECT_ROOT / "data" / "artifacts"

print("=" * 60)
print("SECTION 7: ENSEMBLE FEATURE SELECTION")
print("=" * 60)

# ── Load data ─────────────────────────────────────────────────
df = pd.read_parquet(PROCESSED_DATA_DIR / "breast_cancer_qn.parquet")
feat_cols = np.array([c for c in df.columns if c != 'type'])
X = df[feat_cols].values.astype(np.float32)
y = df['type'].values

le = LabelEncoder()
y_enc = le.fit_transform(y)
print(f"\n[7.0] Data: {X.shape}, Classes: {list(le.classes_)}")

# ── 7.1 Train-test split (BEFORE any feature selection) ───────
print("\n[7.1] Stratified 80/20 train-test split...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, stratify=y_enc, random_state=42
)
print(f"  X_train: {X_train.shape}  X_test: {X_test.shape}")
print(f"  Train class distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")
print(f"  Test  class distribution: {dict(zip(*np.unique(y_test, return_counts=True)))}")
joblib.dump(y_train, ARTIFACT_DIR / "y_train.pkl")
joblib.dump(y_test,  ARTIFACT_DIR / "y_test.pkl")
joblib.dump(le,      ARTIFACT_DIR / "label_encoder.pkl")

# ── 7.2 Step 1: VarianceThreshold (fit on train only) ────────
print("\n[7.2] Step 1: VarianceThreshold (threshold=0.1, train-only fit)...")
vt = VarianceThreshold(threshold=0.1)
vt.fit(X_train)
mask_var = vt.get_support()
X_train_vt = X_train[:, mask_var]
X_test_vt  = X_test[:, mask_var]
feat_var    = feat_cols[mask_var]
print(f"  Genes after variance filter: {feat_var.shape[0]:,}  (removed {(~mask_var).sum():,})")
joblib.dump(vt,       ARTIFACT_DIR / "variance_selector.pkl")
joblib.dump(feat_var, ARTIFACT_DIR / "selected_features.pkl")

# ── 7.3 StandardScaler (fit on train_vt only) ────────────────
print("\n[7.3] StandardScaler (fit on variance-filtered training data)...")
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train_vt)
X_test_sc  = scaler.transform(X_test_vt)
joblib.dump(scaler,        ARTIFACT_DIR / "scaler.pkl")
joblib.dump(X_train_sc,    ARTIFACT_DIR / "X_train_scaled.pkl")
joblib.dump(X_test_sc,     ARTIFACT_DIR / "X_test_scaled.pkl")
print(f"  X_train_scaled: {X_train_sc.shape}  mean~0: {X_train_sc.mean():.6f}  std~1: {X_train_sc.std():.6f}")

# ── 7.4 Method 1: ANOVA F-test ───────────────────────────────
K = 2000
print(f"\n[7.4] Method 1: ANOVA F-test (top K={K})...")
anova = SelectKBest(f_classif, k=K)
anova.fit(X_train_sc, y_train)
mask_anova = anova.get_support()
genes_anova = set(feat_var[mask_anova])
print(f"  Genes selected by ANOVA: {len(genes_anova):,}")
print(f"  Top ANOVA F-scores: {sorted(anova.scores_[mask_anova])[::-1][:5]}")

# ── 7.5 Method 2: Mutual Information ─────────────────────────
print(f"\n[7.5] Method 2: Mutual Information (top K={K})...")
from functools import partial
mi_scorer = partial(mutual_info_classif, random_state=42)
mi = SelectKBest(mi_scorer, k=K)
mi.fit(X_train_sc, y_train)
mask_mi = mi.get_support()
genes_mi = set(feat_var[mask_mi])
print(f"  Genes selected by MI: {len(genes_mi):,}")
print(f"  Overlap with ANOVA: {len(genes_anova & genes_mi):,} genes")

# ── 7.6 Method 3: LASSO on ANOVA-top-2000 pre-filtered genes ─
# Running LASSO on all 34K genes is impractical (~30min).
# Best practice: pre-filter to ANOVA top-2000 then apply LASSO
# (still independent from MI and RF — LASSO penalizes redundancy).
print(f"\n[7.6] Method 3: LASSO L1 Logistic Regression (on ANOVA top-{K})...")
X_train_anova = X_train_sc[:, mask_anova]  # (109, 2000)
X_test_anova  = X_test_sc[:,  mask_anova]
lasso_lr = LogisticRegression(
    penalty='l1', C=0.05, solver='liblinear',
    max_iter=1000, random_state=42
)
# Fit one-vs-rest per class, collect non-zero genes union
from sklearn.multiclass import OneVsRestClassifier
lasso_ovr = OneVsRestClassifier(lasso_lr, n_jobs=-1)
lasso_ovr.fit(X_train_anova, y_train)
# Union of non-zero features across all classifiers
feat_var_anova = feat_var[mask_anova]  # 2000 gene names
lasso_nonzero = np.zeros(K, dtype=bool)
for est in lasso_ovr.estimators_:
    lasso_nonzero |= (np.abs(est.coef_[0]) > 0)
genes_lasso = set(feat_var_anova[lasso_nonzero])
print(f"  LASSO non-zero genes (from ANOVA-2000): {len(genes_lasso):,}")
print(f"  Overlap with ANOVA: {len(genes_anova & genes_lasso):,}")
print(f"  Overlap with MI   : {len(genes_mi & genes_lasso):,}")

# ── 7.7 Method 4: Random Forest Importance ───────────────────
print(f"\n[7.7] Method 4: Random Forest importance (top K={K}, 100 trees)...")
rf_sel = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_sel.fit(X_train_sc, y_train)
importances = rf_sel.feature_importances_
top_rf_idx = np.argsort(importances)[::-1][:K]
mask_rf = np.zeros(len(feat_var), dtype=bool)
mask_rf[top_rf_idx] = True
genes_rf = set(feat_var[mask_rf])
print(f"  Genes selected by RF: {len(genes_rf):,}")
print(f"  Top RF importance scores: {sorted(importances[top_rf_idx])[::-1][:5]}")

# ── 7.8 Consensus: genes in >= 2 of 4 methods ────────────────
print(f"\n[7.8] Consensus feature selection (>= 2 of 4 methods)...")
gene_votes = {}
for gene in feat_var:
    votes = sum([
        gene in genes_anova,
        gene in genes_mi,
        gene in genes_lasso,
        gene in genes_rf,
    ])
    gene_votes[gene] = votes

votes_arr = np.array([gene_votes[g] for g in feat_var])
print(f"  Genes with 1 vote : {(votes_arr == 1).sum():,}")
print(f"  Genes with 2 votes: {(votes_arr == 2).sum():,}")
print(f"  Genes with 3 votes: {(votes_arr == 3).sum():,}")
print(f"  Genes with 4 votes: {(votes_arr == 4).sum():,}")

consensus_mask = votes_arr >= 2
consensus_genes = feat_var[consensus_mask]
print(f"\n  CONSENSUS (>= 2 votes): {len(consensus_genes):,} genes")

# Get consensus indices in the scaled arrays
feat_var_list = list(feat_var)
consensus_indices = [feat_var_list.index(g) for g in consensus_genes]

X_train_consensus = X_train_sc[:, consensus_indices]
X_test_consensus  = X_test_sc[:,  consensus_indices]
print(f"  X_train_consensus shape: {X_train_consensus.shape}")
print(f"  X_test_consensus  shape: {X_test_consensus.shape}")

# Save
joblib.dump(consensus_genes,    ARTIFACT_DIR / "top_consensus_genes.pkl")
joblib.dump(consensus_indices,  ARTIFACT_DIR / "consensus_indices.pkl")
joblib.dump(X_train_consensus,  ARTIFACT_DIR / "X_train_consensus.pkl")
joblib.dump(X_test_consensus,   ARTIFACT_DIR / "X_test_consensus.pkl")
print(f"\n[7.9] Saved all feature selection artifacts.")

print("\n" + "=" * 60)
print("SECTION 7 COMPLETE [OK]")
print("=" * 60)

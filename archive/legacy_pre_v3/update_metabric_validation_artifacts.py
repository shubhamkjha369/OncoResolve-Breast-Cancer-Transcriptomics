"""
Update external_validation_results.pkl artifact with 176/178 feature mapping metrics.
"""

import joblib
import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

repo_root = Path(__file__).resolve().parent.parent
data_dir = repo_root / "data"
artifacts_dir = data_dir / "artifacts"
ext_dir = data_dir / "external_cohort"
processed_dir = data_dir / "processed"

# Load external validation results if exists, or create new dict
art_path = artifacts_dir / "external_validation_results.pkl"
if art_path.exists():
    ext_results = joblib.load(art_path)
else:
    ext_results = {}

# Load models and encoders
pipeline_lgbm = joblib.load(artifacts_dir / 'lgbm_model.pkl')
pipeline_linear_svm = joblib.load(artifacts_dir / 'linear_svm_model.pkl')
lgbm_classifier = getattr(pipeline_lgbm, 'named_steps', {}).get('clf', pipeline_lgbm)
linear_svm_classifier = getattr(pipeline_linear_svm, 'named_steps', {}).get('clf', pipeline_linear_svm)

le_cohort = joblib.load(artifacts_dir / 'label_encoder_cohort.pkl')
known_classes = set(le_cohort.classes_)
top_deg_genes = list(joblib.load(artifacts_dir / 'top_deg_genes.pkl'))
n_features = len(top_deg_genes)
correct_genes_order = sorted([str(g) for g in top_deg_genes])

# Load gene map
with open(artifacts_dir / 'gene_map.json', 'r', encoding='utf-8') as f:
    gene_map = json.load(f)
entrez_to_hugo = {str(k): str(v) for k, v in gene_map['symbols'].items()}

# Evaluate METABRIC
df_clin = pd.read_csv(ext_dir / "METABRIC_clinical.csv")
if "patient_id" in df_clin.columns:
    df_clin = df_clin.set_index("patient_id")

subtype_map = {
    "Basal": "basal", "Her2": "her2", "LumA": "luminal_A", 
    "LumB": "luminal_B", "Normal": "normal", "claudin-low": "basal"
}

ext_df = pd.read_parquet(processed_dir / "METABRIC_expression_clean.parquet")
y_raw = ext_df.index.map(df_clin["CLAUDIN_SUBTYPE"])
y_mapped = y_raw.map(subtype_map)
valid_mask = y_mapped.isin(known_classes)
ext_df = ext_df[valid_mask]
y_ext = y_mapped[valid_mask].values

X_ext_raw = np.zeros((ext_df.shape[0], n_features))
found_count = 0

for idx, feature in enumerate(correct_genes_order):
    hugo_symbol = entrez_to_hugo.get(str(feature), None)
    if hugo_symbol is not None and hugo_symbol in ext_df.columns:
        median_val = ext_df[hugo_symbol].median()
        val = ext_df[hugo_symbol].astype(float).fillna(median_val if not np.isnan(median_val) else 0.0).values
        if len(val) > 0 and val.max() > 50:
            val = np.log2(np.clip(val, 0, None) + 1)
        X_ext_raw[:, idx] = val
        found_count += 1
    else:
        X_ext_raw[:, idx] = 0.0

scaler_ext = StandardScaler()
X_ext_aligned = np.nan_to_num(scaler_ext.fit_transform(X_ext_raw), nan=0.0)

y_pred_lgbm = lgbm_classifier.predict(X_ext_aligned)
y_pred_linear_svm = linear_svm_classifier.predict(X_ext_aligned)

if np.issubdtype(y_pred_lgbm.dtype, np.number):
    y_pred_lgbm = le_cohort.inverse_transform(y_pred_lgbm.astype(int))
if np.issubdtype(y_pred_linear_svm.dtype, np.number):
    y_pred_linear_svm = le_cohort.inverse_transform(y_pred_linear_svm.astype(int))

acc_lgbm = accuracy_score(y_ext, y_pred_lgbm)
bal_acc_lgbm = balanced_accuracy_score(y_ext, y_pred_lgbm)
f1_lgbm_weighted = f1_score(y_ext, y_pred_lgbm, average="weighted")
f1_lgbm_macro = f1_score(y_ext, y_pred_lgbm, average="macro")

acc_svm = accuracy_score(y_ext, y_pred_linear_svm)
bal_acc_svm = balanced_accuracy_score(y_ext, y_pred_linear_svm)
f1_svm_weighted = f1_score(y_ext, y_pred_linear_svm, average="weighted")
f1_svm_macro = f1_score(y_ext, y_pred_linear_svm, average="macro")

ext_results["METABRIC"] = {
    "svm": {
        "acc": acc_lgbm,
        "bal_acc": bal_acc_lgbm,
        "f1": f1_lgbm_weighted,
        "f1_weighted": f1_lgbm_weighted,
        "f1_macro": f1_lgbm_macro,
        "y_pred": y_pred_lgbm
    },
    "lr": {
        "acc": acc_svm,
        "bal_acc": bal_acc_svm,
        "f1": f1_svm_weighted,
        "f1_weighted": f1_svm_weighted,
        "f1_macro": f1_svm_macro,
        "y_pred": y_pred_linear_svm
    },
    "y_true": y_ext,
    "n_shared": found_count,
    "n_samples": len(y_ext),
    "platform": "Illumina HT-12 v3 microarray"
}

joblib.dump(ext_results, art_path)
print(f"[SUCCESS] Updated {art_path.name} artifact with METABRIC {found_count}/{n_features} mapped genes.")

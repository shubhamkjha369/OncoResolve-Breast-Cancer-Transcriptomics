import pandas as pd
import gseapy as gp
import joblib
from pathlib import Path

ARTIFACT_DIR = Path("data/artifacts")
best_rf = joblib.load(ARTIFACT_DIR / "tuned_rf.pkl")
best_lr = joblib.load(ARTIFACT_DIR / "tuned_lr.pkl")
X_train_c = joblib.load(ARTIFACT_DIR / "X_train_consensus.pkl")
y_train = joblib.load(ARTIFACT_DIR / "y_train.pkl")

# Load annotated shap df
shap_df_annot = pd.read_parquet(ARTIFACT_DIR / "shap_importance_annotated.parquet")

# Collapse probes to genes
gene_level_df = (
    shap_df_annot
    .groupby("gene_symbol", as_index=False)
    .agg({
        "ensemble_shap": "max",
        "rf_mean_abs_shap": "max",
        "lr_mean_abs_shap": "max"
    })
)

# Use top 300 genes
top_genes_shap = (
    gene_level_df
    .sort_values("ensemble_shap", ascending=False)
    .head(300)
    ["gene_symbol"]
    .tolist()
)

print("Running Enrichr on KEGG...")
enr_kegg = gp.enrichr(
    gene_list=top_genes_shap, gene_sets='KEGG_2021_Human', organism='human', outdir=None, verbose=False
)
raw_results = enr_kegg.results

print("\nBreast Cancer row:")
print(raw_results[raw_results["Term"].str.contains("Breast cancer", case=False, na=False)])

print("\nProstate Cancer row:")
print(raw_results[raw_results["Term"].str.contains("Prostate cancer", case=False, na=False)])

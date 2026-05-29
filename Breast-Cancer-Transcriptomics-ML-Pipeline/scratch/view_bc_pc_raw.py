import pandas as pd
import gseapy as gp
from pathlib import Path

# Let's run a simple script that loads top_genes_shap from the parquet we just generated and prints the raw rows
ARTIFACT_DIR = Path("data/artifacts")
shap_df_annot = pd.read_parquet(ARTIFACT_DIR / "shap_importance_annotated.parquet")
gene_level_df = shap_df_annot.groupby("gene_symbol", as_index=False).agg({"ensemble_shap": "max"})
top_genes_shap = gene_level_df.sort_values("ensemble_shap", ascending=False).head(300)["gene_symbol"].tolist()

enr_kegg = gp.enrichr(gene_list=top_genes_shap, gene_sets='KEGG_2021_Human', organism='human', outdir=None, verbose=False)
raw = enr_kegg.results

bc = raw[raw["Term"].str.contains("Breast cancer", case=False, na=False)]
pc = raw[raw["Term"].str.contains("Prostate cancer", case=False, na=False)]

print("--- Breast Cancer Row ---")
for col in bc.columns:
    print(f"{col}: {bc.iloc[0][col]}")
print("\n--- Prostate Cancer Row ---")
for col in pc.columns:
    print(f"{col}: {pc.iloc[0][col]}")

import joblib
import pandas as pd
import numpy as np
import os

print("=== OncoResolve Artifact Verification Script ===")

artifacts_dir = "data/artifacts"

# 1. Load top consensus genes using joblib
genes_path = os.path.join(artifacts_dir, "top_consensus_genes.pkl")
try:
    top_consensus_genes = joblib.load(genes_path)
    print(f"Top consensus genes type: {type(top_consensus_genes)}")
    print(f"Count of top consensus genes: {len(top_consensus_genes)}")
    affx_genes = [g for g in top_consensus_genes if "AFFX" in str(g)]
    print(f"Number of AFFX probes in top consensus genes: {len(affx_genes)}")
    if len(affx_genes) > 0:
        print(f"WARNING: Found AFFX probes: {affx_genes}")
except Exception as e:
    print(f"Error loading consensus genes: {e}")

# 2. Check uniqueness residuals parquet
residuals_path = os.path.join(artifacts_dir, "uniqueness_residuals.parquet")
try:
    df_residuals = pd.read_parquet(residuals_path)
    print(f"Uniqueness residuals shape: {df_residuals.shape}")
    affx_cols = [c for c in df_residuals.columns if "AFFX" in str(c)]
    print(f"Number of AFFX probes in uniqueness residuals columns: {len(affx_cols)}")
    if len(affx_cols) > 0:
        print(f"WARNING: Found AFFX probes in residuals: {affx_cols}")
except Exception as e:
    print(f"Error loading residuals: {e}")

# 3. Check other parquets/npy files using joblib
for filename in ["pca_50.pkl", "final_audited_pipeline.pkl", "scaler.pkl"]:
    p = os.path.join(artifacts_dir, filename)
    if os.path.exists(p):
        try:
            obj = joblib.load(p)
            print(f"Successfully loaded {filename} using joblib. Type: {type(obj)}")
        except Exception as e:
            print(f"Error loading {filename} using joblib: {e}")
    else:
        print(f"Checked existence of {filename}: False")

# Check PCA 2D parquet
dr_path = os.path.join(artifacts_dir, "dr_coordinates.parquet")
if os.path.exists(dr_path):
    df_dr = pd.read_parquet(dr_path)
    print(f"Dimensional reduction coordinates shape: {df_dr.shape}")
    print(df_dr.head(2))

print("=== Verification Complete ===")

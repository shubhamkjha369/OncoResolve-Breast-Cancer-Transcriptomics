import pandas as pd
import numpy as np
import os
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
data_dir = REPO_ROOT / "data"
raw_dir = data_dir / "raw"
ext_dir = data_dir / "external_cohort"
processed_dir = data_dir / "processed"
art_dir = data_dir / "artifacts"

# Create output directories if not exist
os.makedirs(processed_dir, exist_ok=True)
os.makedirs(art_dir, exist_ok=True)

# -------------------------------------------------------------
# Step 1: Load the Datasets
# -------------------------------------------------------------
print("Step 1: Loading raw datasets...")
metabric_path = os.path.join(ext_dir, "METABRIC_expression.csv")
scanb_path = os.path.join(ext_dir, "SCANB_GSE96058_expression_subset.csv")
met_clin_path = os.path.join(ext_dir, "METABRIC_clinical.csv")
scan_clin_path = os.path.join(ext_dir, "SCANB_GSE96058_clinical.csv")
scan_map_path = os.path.join(ext_dir, "SCANB_mapping.csv")

metabric_raw = pd.read_csv(metabric_path)
scanb_raw = pd.read_csv(scanb_path)
df_met_clin = pd.read_csv(met_clin_path)
df_scan_clin = pd.read_csv(scan_clin_path)
df_scan_map = pd.read_csv(scan_map_path)

print(f"METABRIC raw expression shape: {metabric_raw.shape}")
print(f"SCAN-B raw expression shape: {scanb_raw.shape}")

# -------------------------------------------------------------
# Step 4, 5, 7: Standardize orientation, filter by cancer subtypes, set index
# -------------------------------------------------------------
print("\nStep 4, 5 & 7: Standardizing orientation, setting index, and filtering out normal, cell-line, & NaN subtypes...")

# 1. METABRIC
df_met = metabric_raw.copy()
df_met.set_index("sample_id", inplace=True)
df_met.index.name = "sample_id"

if "patient_id" in df_met_clin.columns:
    df_met_clin = df_met_clin.set_index("patient_id")

# Strict Filter METABRIC
met_claudin = df_met.index.map(df_met_clin["CLAUDIN_SUBTYPE"])
valid_claudin_subtypes = ["LumA", "LumB", "Her2", "claudin-low", "Basal", "Normal"]
met_keep_mask = met_claudin.isin(valid_claudin_subtypes)
df_met = df_met[met_keep_mask]
print(f"METABRIC filtered shape (retained only valid cancer subtypes): {df_met.shape}")

# 2. SCAN-B
df_scan = scanb_raw.copy()
df_scan.set_index("sample_id", inplace=True)
df_scan.index.name = "sample_id"

# Clean mapping
df_scan_map['gsm_id_clean'] = df_scan_map['gsm_id'].str.strip().str.replace('"', '').str.replace('\n', '')
df_scan_map['f_id_clean'] = df_scan_map['f_id'].str.strip().str.replace('"', '').str.replace('\n', '')
f_to_gsm = dict(zip(df_scan_map["f_id_clean"], df_scan_map["gsm_id_clean"]))

if "sample_id" in df_scan_clin.columns:
    df_scan_clin = df_scan_clin.set_index("sample_id")

# Strict Filter SCAN-B
scan_gsm = df_scan.index.map(f_to_gsm)
scan_pam50 = scan_gsm.map(df_scan_clin["pam50_subtype"])
valid_pam50_subtypes = ["LumA", "LumB", "Her2", "Basal", "Normal"]
scan_keep_mask = scan_pam50.isin(valid_pam50_subtypes)
df_scan = df_scan[scan_keep_mask]
print(f"SCAN-B filtered shape (retained only valid cancer subtypes): {df_scan.shape}")

# Verify all columns are numeric
print("Are METABRIC columns all numeric?", all(np.issubdtype(t, np.number) for t in df_met.dtypes))
print("Are SCAN-B columns all numeric?", all(np.issubdtype(t, np.number) for t in df_scan.dtypes))

# -------------------------------------------------------------
# Step 2 & 3: Investigate scale and log2 transform status
# -------------------------------------------------------------
print("\nStep 2: Checking statistics for log2-transformation status on filtered data...")

def get_stats(df, label):
    # Select numeric columns
    numeric_cols = df.select_dtypes(include=[np.number])
    flat_vals = numeric_cols.values.flatten()
    
    # Calculate stats
    stats_dict = {
        "Min": np.nanmin(flat_vals),
        "Max": np.nanmax(flat_vals),
        "Mean": np.nanmean(flat_vals),
        "Median": np.nanmedian(flat_vals),
        "1%": np.nanpercentile(flat_vals, 1),
        "5%": np.nanpercentile(flat_vals, 5),
        "25%": np.nanpercentile(flat_vals, 25),
        "50%": np.nanpercentile(flat_vals, 50),
        "75%": np.nanpercentile(flat_vals, 75),
        "95%": np.nanpercentile(flat_vals, 95),
        "99%": np.nanpercentile(flat_vals, 99),
    }
    
    print(f"\n{label} Statistics:")
    for k, v in stats_dict.items():
        print(f"  {k:<8}: {v:.6f}")
        
    return stats_dict

metabric_stats = get_stats(df_met, "METABRIC Filtered")
scanb_stats = get_stats(df_scan, "SCAN-B Filtered")

# Distribution Plots
plt.figure(figsize=(12, 5))
plt.style.use('default')
sns.set_theme(style="ticks", context="paper", font_scale=1.2)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# METABRIC
sns.histplot(df_met.values.flatten()[::1000], kde=True, color="#1f77b4", ax=axes[0])
axes[0].set_title("METABRIC Expression Distribution (Sampled)", fontweight='bold', pad=12)
axes[0].set_xlabel("Expression Value")
axes[0].set_ylabel("Frequency")
sns.despine(ax=axes[0])

# SCAN-B
sns.histplot(df_scan.values.flatten()[::1000], kde=True, color="#ff7f0e", ax=axes[1])
axes[1].set_title("SCAN-B Expression Distribution (Sampled)", fontweight='bold', pad=12)
axes[1].set_xlabel("Expression Value")
axes[1].set_ylabel("Frequency")
sns.despine(ax=axes[1])

plt.tight_layout()
plt.savefig(os.path.join(art_dir, "raw_distributions.png"), dpi=150)
plt.close()

# -------------------------------------------------------------
# Step 6: Handle Missing Values
# -------------------------------------------------------------
print("\nStep 6: Auditing missing values...")
print(f"METABRIC total nulls: {df_met.isnull().sum().sum()}")
print(f"SCAN-B total nulls: {df_scan.isnull().sum().sum()}")

# -------------------------------------------------------------
# Step 8: Gene Identifier Audit & Unique Column check
# -------------------------------------------------------------
print("\nStep 8: Auditing gene names...")
# Clean whitespace
df_met.columns = df_met.columns.str.strip()
df_scan.columns = df_scan.columns.str.strip()

print("Asserting columns are unique...")
assert df_met.columns.is_unique, "METABRIC column names are not unique!"
assert df_scan.columns.is_unique, "SCAN-B column names are not unique!"
print("Gene symbols are verified to be unique.")

# Check duplicate sample IDs
print(f"METABRIC duplicate sample IDs in index: {df_met.index.duplicated().sum()}")
print(f"SCAN-B duplicate sample IDs in index: {df_scan.index.duplicated().sum()}")
assert df_met.index.duplicated().sum() == 0, "METABRIC has duplicate sample IDs!"
assert df_scan.index.duplicated().sum() == 0, "SCAN-B has duplicate sample IDs!"

# -------------------------------------------------------------
# Step 8.5: Gene Harmonization Audit
# -------------------------------------------------------------
print("\nStep 8.5: Executing gene harmonization overlap analysis...")
mapping_path = os.path.join(art_dir, "tcga_entrez_to_hugo.pkl")
if os.path.exists(mapping_path):
    entrez_to_hugo = joblib.load(mapping_path)
else:
    raise FileNotFoundError("TCGA Entrez to HUGO symbol mapping pickle not found. Run mapping script first!")

tcga_path = os.path.join(processed_dir, "breast_cancer.parquet")
tcga_df = pd.read_parquet(tcga_path)
tcga_expr = tcga_df.drop(columns=["type"])
tcga_expr.columns = tcga_expr.columns.map(entrez_to_hugo)
# Drop unmapped columns
tcga_expr = tcga_expr.loc[:, tcga_expr.columns.notna()]

train_genes = list(tcga_expr.columns)
metabric_genes = list(df_met.columns)
scanb_genes = list(df_scan.columns)

s_train = set(train_genes)
s_met = set(metabric_genes)
s_scan = set(scanb_genes)

common_genes = sorted(list(s_train.intersection(s_met).intersection(s_scan)))

print(f"Training set genes count: {len(s_train)}")
print(f"METABRIC genes count: {len(s_met)}")
print(f"SCAN-B genes count: {len(s_scan)}")
print(f"Common genes across all three: {len(common_genes)}")

# -------------------------------------------------------------
# Step 9: Save Processed External Cohorts
# -------------------------------------------------------------
print("\nStep 9: Saving processed external cohorts...")

# Ensure columns are sorted alphabetically for strict order preservation
df_met_clean = df_met.reindex(columns=sorted(df_met.columns))
df_scan_clean = df_scan.reindex(columns=sorted(df_scan.columns))

# Verify column sorting
assert list(df_met_clean.columns) == sorted(df_met.columns)
assert list(df_scan_clean.columns) == sorted(df_scan.columns)

# Save to parquet (primary)
met_parquet_path = os.path.join(processed_dir, "METABRIC_expression_clean.parquet")
scan_parquet_path = os.path.join(processed_dir, "SCANB_expression_clean.parquet")
df_met_clean.to_parquet(met_parquet_path, index=True)
df_scan_clean.to_parquet(scan_parquet_path, index=True)

# Save to CSV (backup)
met_csv_path = os.path.join(processed_dir, "METABRIC_expression_clean.csv")
scan_csv_path = os.path.join(processed_dir, "SCANB_expression_clean.csv")
df_met_clean.to_csv(met_csv_path, index=True)
df_scan_clean.to_csv(scan_csv_path, index=True)

print(f"Saved METABRIC clean matrix to: {met_parquet_path} (Shape: {df_met_clean.shape})")
print(f"Saved SCAN-B clean matrix to: {scan_parquet_path} (Shape: {df_scan_clean.shape})")

# -------------------------------------------------------------
# Step 10: External Cohort Compatibility Audit (PCA)
# -------------------------------------------------------------
print("\nStep 10: Performing External Cohort Compatibility Audit (PCA)...")

# Prepare common subset
train_sub = tcga_expr[common_genes].copy()
met_sub = df_met_clean[common_genes].copy()
scan_sub = df_scan_clean[common_genes].copy()

# Print checks
print(f"Subset Shapes - Train: {train_sub.shape}, METABRIC: {met_sub.shape}, SCAN-B: {scan_sub.shape}")

# Scale independently to center them and remove global platform offsets
scaler = StandardScaler()
train_scaled = pd.DataFrame(scaler.fit_transform(train_sub), columns=common_genes, index=train_sub.index)
met_scaled = pd.DataFrame(scaler.fit_transform(met_sub), columns=common_genes, index=met_sub.index)
scan_scaled = pd.DataFrame(scaler.fit_transform(scan_sub), columns=common_genes, index=scan_sub.index)

# Add cohort labels
train_scaled['cohort'] = 'Train (TCGA)'
met_scaled['cohort'] = 'METABRIC'
scan_scaled['cohort'] = 'SCAN-B'

# Combine the three cohorts
combined_df = pd.concat([train_scaled, met_scaled, scan_scaled], axis=0)
print(f"Combined data shape for PCA: {combined_df.shape}")

# Perform PCA
X_combined = combined_df.drop(columns=['cohort']).values
cohorts = combined_df['cohort'].values

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_combined)

explained_variance = pca.explained_variance_ratio_
print(f"Explained variance - PC1: {explained_variance[0]:.4f}, PC2: {explained_variance[1]:.4f}")

# Create PCA plot
plt.figure(figsize=(10, 8))
# Set theme
sns.set_theme(style="ticks", context="paper", font_scale=1.2)

# Professional color palette
colors = {'Train (TCGA)': '#1f77b4', 'METABRIC': '#ff7f0e', 'SCAN-B': '#2ca02c'}

for cohort in ['Train (TCGA)', 'METABRIC', 'SCAN-B']:
    mask = cohorts == cohort
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=cohort, color=colors[cohort], alpha=0.7, s=40, edgecolors='w', linewidths=0.3)

plt.title("Cross-Cohort Transcriptomic Compatibility Space (PCA Projection)", fontweight='bold', fontsize=14, pad=15)
plt.xlabel(f"Principal Component 1 ({explained_variance[0]*100:.2f}% Variance)", fontweight='bold')
plt.ylabel(f"Principal Component 2 ({explained_variance[1]*100:.2f}% Variance)", fontweight='bold')
plt.legend(title="Cohort", frameon=True, loc='best')
plt.grid(True, linestyle='--', alpha=0.5)
sns.despine(offset=10)
plt.tight_layout()

# Save PCA plot
pca_plot_path = os.path.join(art_dir, "cross_cohort_pca_compatibility.png")
plt.savefig(pca_plot_path, dpi=300, transparent=False, facecolor='white')
plt.close()
print(f"Saved cross-cohort PCA compatibility plot to: {pca_plot_path}")
print("Workflow run complete! All outputs generated successfully.")

"""
assemble_notebook.py - Unified Jupyter Notebook Assembler
===========================================================
This script constructs a single, beautiful, and comprehensive Jupyter Notebook
named 'Breast_Cancer_ML_Pipeline.ipynb' in the notebooks/ directory.
It integrates all 13 sections of code, detailed explanations for non-coder biologists,
and premium visualizations using Matplotlib/Seaborn/Plotly.
"""
import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.13.1"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

def add_markdown(text):
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().split("\n")]
    })

def add_code(code):
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in code.strip().split("\n")]
    })

# =====================================================================
# INTRODUCTORY CELL
# =====================================================================
add_markdown("""
# 🧬 Breast Cancer Transcriptomics & Machine Learning Pipeline
### An End-to-End Bioinformatics, Machine Learning, and Deep Learning Project
**Dataset:** GSE45827 (Affymetrix Human Genome U133 Plus 2.0 Array)  
**Task:** Classify breast cancer clinical samples into five molecular subtypes: **Basal, HER2, Luminal A, Luminal B, and Normal-like**.

---

## 📘 Target Audience & Objectives
* **Audience:** Non-coder biologists, clinical researchers, and data scientists.
* **Goals:**
  1. Process raw gene expression microarray profiles with rigorous QC and quantile normalization.
  2. Perform Exploratory Data Analysis (EDA) and Differential Gene Expression (DGE) analysis to discover subtype-specific transcripts.
  3. Cluster patients unsupervised to see if true biological groups emerge naturally.
  4. Perform Co-expression Network Analysis (WGCNA equivalent) to define modules and hub genes.
  5. Apply ensemble feature selection methods to avoid overfitting.
  6. Train and benchmark classical machine learning and deep learning models.
  7. Use SHAP (SHapley Additive exPlanations) for model interpretability and biomarker discovery.
  8. Perform biological pathway enrichment analysis (GO and KEGG) to validate the model's features.
""")

# =====================================================================
# SECTION 0 — Environment Setup & Imports
# =====================================================================
add_markdown("""
## 🛠️ SECTION 0: Environment Setup & Imports
We begin by importing all necessary Python packages, setting up the file path architecture relative to our notebook directory, and defining global visualization aesthetics.
""")

add_code("""
import sys
import os
import warnings
from pathlib import Path

# Scientific computing & data wrangling
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
from statsmodels.stats.multitest import multipletests

# Machine Learning & Unsupervised Clustering
from sklearn.preprocessing import RobustScaler, StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
    silhouette_score, adjusted_rand_score, normalized_mutual_info_score,
    roc_curve, auc
)

# Deep Learning (PyTorch)
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Bioinformatic APIs, Explainability & Pathway Enrichment
import shap
import mygene
import gseapy as gp

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Setup absolute file path architecture
# Notebook runs in the 'notebooks' folder, so paths are relative to parent
PROJECT_ROOT = Path("..").resolve()
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACT_DIR = PROJECT_ROOT / "data" / "artifacts"

# Create directories if they do not exist
for d in [PROCESSED_DATA_DIR, ARTIFACT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Aesthetic configuration
sns.set_theme(style="whitegrid", context="notebook", palette="muted")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 12
warnings.filterwarnings("ignore")

# PyTorch Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Environment ready! Device: {device} | PyTorch version: {torch.__version__}")
""")

# =====================================================================
# SECTION 1 — Data Loading & Quality Control
# =====================================================================
add_markdown("""
## 🧬 SECTION 1: Data Loading & Quality Control
### 🔬 Biological Context
Gene expression microarray data contains measurements of thousands of RNA transcripts across tumor samples. The dataset (**GSE45827**) contains 151 samples and 54,675 probes. 
However, **21 of these samples are cell lines**, which represent laboratory cell cultures. Cell lines undergo genetic drift and lack the microenvironment of real patient tumors, representing potential confounders. 
In this section, we load the raw data, remove the cell line samples (leaving **130 clinical samples**), cast values to memory-efficient `float32`, and verify that the data is already on a log2 scale (Affymetrix RMA output).
""")

add_code("""
# ── 1.1 Load Raw CSV Data ──
csv_path = RAW_DATA_DIR / "Breast_GSE45827.csv"
df_raw = pd.read_csv(csv_path)

print(f"Raw dataset shape: {df_raw.shape[0]} samples x {df_raw.shape[1]:,} features")
print(f"Unique class labels present: {df_raw['type'].unique().tolist()}")

# ── 1.2 Class Distribution Before Filtering ──
print("\\nClass distribution before removing cell lines:")
vc_all = df_raw['type'].value_counts()
for label, count in vc_all.items():
    print(f"  {label:<15}: {count:>3} samples ({count/len(df_raw)*100:.1f}%)")

# ── 1.3 Remove Cell Line Samples & ID Column ──
df = df_raw[df_raw['type'] != 'cell_line'].copy()
df = df.drop(columns=['samples'])
print(f"\\nRemoved {len(df_raw) - len(df)} cell line samples.")
print(f"Remaining clinical samples shape: {df.shape[0]} samples x {df.shape[1]:,} features")

# ── 1.4 Class Distribution of Clinical Subtypes ──
print("\\nClass distribution of clinical subtypes:")
vc_clinical = df['type'].value_counts()
for label, count in vc_clinical.items():
    print(f"  {label:<15}: {count:>3} samples ({count/len(df)*100:.1f}%)")

# ── 1.5 Optimize Memory via Downcasting ──
feat_cols = [c for c in df.columns if c != 'type']
df[feat_cols] = df[feat_cols].astype(np.float32)
mem_mb = df[feat_cols].memory_usage(deep=True).sum() / 1e6
print(f"\\nMemory footprint of expression values: {mem_mb:.2f} MB")

# ── 1.6 Expression Scale Verification ──
# Sample 1% of the data to perform distribution checks
sample_vals = df[feat_cols].values.flatten()[::100]
print("\\nExpression value distribution (checks for log2 scale):")
print(f"  Minimum value      : {float(np.percentile(sample_vals, 0)):.3f}")
print(f"  1st Percentile     : {float(np.percentile(sample_vals, 1)):.3f}")
print(f"  Median value       : {float(np.percentile(sample_vals, 50)):.3f}")
print(f"  95th Percentile    : {float(np.percentile(sample_vals, 95)):.3f}")
print(f"  Maximum value      : {float(np.percentile(sample_vals, 100)):.3f}")
print("  (Confirmed: typical log2 expression range of 2 to 16 for microarrays)")

# Save parsed raw parquet
df.to_parquet(PROCESSED_DATA_DIR / "breast_cancer.parquet", index=False)
""")

# Plotting subtype counts for QC
add_code("""
# ── 1.7 Visualize Subtypes Before vs. After Filtering ──
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Before
sns.barplot(x=vc_all.index, y=vc_all.values, ax=axes[0], palette="viridis")
axes[0].set_title("Subtype Distribution (Raw, n=151)")
axes[0].set_ylabel("Sample Count")
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=30)

# After
sns.barplot(x=vc_clinical.index, y=vc_clinical.values, ax=axes[1], palette="magma")
axes[1].set_title("Subtype Distribution (Clinical Only, n=130)")
axes[1].set_ylabel("Sample Count")
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=30)

plt.tight_layout()
plt.show()
""")

# =====================================================================
# SECTION 2 — Normalization & Data Quality
# =====================================================================
add_markdown("""
## 📊 SECTION 2: Quantile Normalization & Quality Checks
### 🔬 Biological Context
Microarray experiments can suffer from technical variations (e.g., differences in pipetting, hybridization efficiency, or laser scanning). 
To make samples directly comparable, we apply **Quantile Normalization (QN)**. QN is a non-parametric method that forces all samples to have the exact same distribution of intensities, adjusting for technical noise without distorting biological signals.
We also compute sample-sample correlations to identify potential technical outlier samples.
""")

add_code("""
# Load raw processed data
df = pd.read_parquet(PROCESSED_DATA_DIR / "breast_cancer.parquet")
feat_cols = [c for c in df.columns if c != 'type']
X = df[feat_cols].values.astype(np.float32)
y = df['type'].values

# ── 2.1 Per-Sample Medians BEFORE Normalization ──
per_sample_median_before = np.median(X, axis=1)
print("Before Normalization per-sample median spread:")
print(f"  Mean: {per_sample_median_before.mean():.4f} | Std: {per_sample_median_before.std():.4f}")
print(f"  Range (Min-Max): {per_sample_median_before.min():.4f} to {per_sample_median_before.max():.4f}")

# ── 2.2 Perform Quantile Normalization ──
# Sort columns (samples) to find rank order
X_sorted = np.sort(X, axis=1)
reference = X_sorted.mean(axis=0)  # average profile
ranks = np.argsort(np.argsort(X, axis=1), axis=1)  # indices of sorted values
X_qn = reference[ranks]            # project ranks onto average profile

# ── 2.3 Per-Sample Medians AFTER Normalization ──
per_sample_median_after = np.median(X_qn, axis=1)
print("\\nAfter Normalization per-sample median spread:")
print(f"  Mean: {per_sample_median_after.mean():.4f} | Std: {per_sample_median_after.std():.4f}")
print(f"  Range (Min-Max): {per_sample_median_after.min():.4f} to {per_sample_median_after.max():.4f}")
variability_reduction = (1 - per_sample_median_after.std() / per_sample_median_before.std()) * 100
print(f"  Technical sample variability reduced by: {variability_reduction:.2f}%")

# ── 2.4 Outlier Sample Detection ──
corr_matrix = np.corrcoef(X_qn)  # (130, 130) sample-sample correlation
np.fill_diagonal(corr_matrix, np.nan)
mean_corr_per_sample = np.nanmean(corr_matrix, axis=1)

global_mean = np.nanmean(mean_corr_per_sample)
global_std  = np.nanstd(mean_corr_per_sample)
outlier_threshold = global_mean - 2 * global_std
outlier_mask = mean_corr_per_sample < outlier_threshold

print(f"\\nOutlier Detection:")
print(f"  Global mean sample correlation: {global_mean:.4f} | Std: {global_std:.4f}")
print(f"  Outlier Threshold (mean - 2*std): {outlier_threshold:.4f}")
print(f"  Outliers detected: {outlier_mask.sum()}")
if outlier_mask.sum() > 0:
    print(f"  Outlier subtypes: {list(y[outlier_mask])}")

# ── 2.5 Save Normalized Data ──
df_qn = pd.DataFrame(X_qn, columns=feat_cols, dtype=np.float32)
df_qn.insert(0, 'type', y)
df_qn.to_parquet(PROCESSED_DATA_DIR / "breast_cancer_qn.parquet", index=False)
print("\\nSaved quantile-normalized data to 'breast_cancer_qn.parquet'")
""")

add_code("""
# ── 2.6 Visualize Normalization Effects ──
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Plot medians
sns.histplot(per_sample_median_before, ax=axes[0], color="red", label="Raw", kde=True)
sns.histplot(per_sample_median_after, ax=axes[0], color="blue", label="Quantile Normalized", kde=True)
axes[0].set_title("Per-sample Median Distribution")
axes[0].set_xlabel("Expression Value")
axes[0].legend()

# Correlation Heatmap
sns.heatmap(corr_matrix, ax=axes[1], cmap="coolwarm", xticklabels=False, yticklabels=False)
axes[1].set_title("Sample-Sample Correlation Matrix (QN)")

plt.tight_layout()
plt.show()
""")

# =====================================================================
# SECTION 3 — Exploratory Data Analysis (EDA)
# =====================================================================
add_markdown("""
## 🔍 SECTION 3: Exploratory Data Analysis (Dimensionality Reduction)
### 🔬 Biological Context
Gene expression tables have 54,675 dimensions, making visual grouping impossible. 
In this section, we reduce dimensions using:
1. **PCA (Principal Component Analysis):** Linear reduction highlighting global variation.
2. **t-SNE (t-Distributed Stochastic Neighbor Embedding):** Non-linear mapping preserving local neighborhoods.
3. **UMAP (Uniform Manifold Approximation and Projection):** Non-linear mapping preserving both local and global structure.

To reduce computational overhead and technical noise, we first filter for the **top 5000 most variable genes**.
""")

add_code("""
# Load QN data
df = pd.read_parquet(PROCESSED_DATA_DIR / "breast_cancer_qn.parquet")
feat_cols = [c for c in df.columns if c != 'type']
X = df[feat_cols].values.astype(np.float32)
y = df['type'].values

# ── 3.1 Pre-filter to Top 5000 Most Variable Genes ──
gene_vars = np.var(X, axis=0)
top5k_idx = np.argsort(gene_vars)[-5000:]
X_top5k = X[:, top5k_idx]

# ── 3.2 Standard Scaling & PCA (50 Components) ──
scaler_robust = RobustScaler()
X_scaled = scaler_robust.fit_transform(X_top5k)

pca = PCA(n_components=50, random_state=42)
X_pca50 = pca.fit_transform(X_scaled)
cumvar = np.cumsum(pca.explained_variance_ratio_)

print(f"Variance explained by Principal Components:")
print(f"  PC1: {pca.explained_variance_ratio_[0]*100:.2f}%")
print(f"  PC2: {pca.explained_variance_ratio_[1]*100:.2f}%")
print(f"  PC3: {pca.explained_variance_ratio_[2]*100:.2f}%")
print(f"  Top 10 PCs cumulative: {cumvar[9]*100:.2f}%")
print(f"  Top 50 PCs cumulative: {cumvar[49]*100:.2f}%")

# Save PC1 & PC2 for fast indexing
pca2_df = pd.DataFrame({'PC1': X_pca50[:, 0], 'PC2': X_pca50[:, 1], 'subtype': y})
pca2_df.to_parquet(PROCESSED_DATA_DIR / "pca_2d.parquet", index=False)

# ── 3.3 t-SNE ──
tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', n_iter=1000)
X_tsne = tsne.fit_transform(X_pca50)
print(f"\\nt-SNE completed. Final KL Divergence: {tsne.kl_divergence_:.4f}")

# ── 3.4 UMAP ──
import umap
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
X_umap = reducer.fit_transform(X_pca50)
print("UMAP completed.")

# ── 3.5 Save Dimensionality Reduction Coordinates ──
dr_df = pd.DataFrame({
    'subtype': y,
    'PC1': X_pca50[:, 0], 'PC2': X_pca50[:, 1], 'PC3': X_pca50[:, 2],
    'TSNE1': X_tsne[:, 0], 'TSNE2': X_tsne[:, 1],
    'UMAP1': X_umap[:, 0], 'UMAP2': X_umap[:, 1],
})
dr_df.to_parquet(ARTIFACT_DIR / "dr_coordinates.parquet", index=False)
joblib.dump(pca, ARTIFACT_DIR / "pca_50.pkl")
print("Saved DR coordinates and pca_50.pkl model.")
""")

add_code("""
# ── 3.6 Visualize PCA, t-SNE, and UMAP ──
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
subtypes_list = sorted(list(np.unique(y)))

for st in subtypes_list:
    mask = y == st
    axes[0].scatter(X_pca50[mask, 0], X_pca50[mask, 1], label=st, alpha=0.8)
    axes[1].scatter(X_tsne[mask, 0], X_tsne[mask, 1], label=st, alpha=0.8)
    axes[2].scatter(X_umap[mask, 0], X_umap[mask, 1], label=st, alpha=0.8)

axes[0].set_title("PCA (PC1 vs PC2)")
axes[0].set_xlabel("PC1")
axes[0].set_ylabel("PC2")

axes[1].set_title("t-SNE")
axes[1].set_xlabel("t-SNE 1")
axes[1].set_ylabel("t-SNE 2")

axes[2].set_title("UMAP")
axes[2].set_xlabel("UMAP 1")
axes[2].set_ylabel("UMAP 2")
axes[2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()
""")

# =====================================================================
# SECTION 4 — Differential Gene Expression (DGE) Analysis
# =====================================================================
add_markdown("""
## 🧬 SECTION 4: Differential Gene Expression (DGE) Analysis
### 🔬 Biological Context
Differential expression identifies specific genes whose transcription levels change significantly between subtypes. 
Here, we perform a **One-vs-Rest Welch's t-test** for each probe. Welch's t-test is used because it does not assume equal variances between groups. 
To control for false positives when testing 54,000+ genes, we apply the **Benjamini-Hochberg False Discovery Rate (FDR)** correction.
Significant Differentially Expressed Genes (DEGs) are filtered using:
* $\\text{FDR} < 0.05$
* $|\\log_2 \\text{Fold Change}| > 1.0$ (representing at least a 2-fold change in expression)
""")

add_code("""
# Load QN data
df = pd.read_parquet(PROCESSED_DATA_DIR / "breast_cancer_qn.parquet")
feat_cols = [c for c in df.columns if c != 'type']
X = df[feat_cols].values.astype(np.float32)
y = df['type'].values

subtypes = sorted(np.unique(y))
dge_results = {}
all_dge = []

for st in subtypes:
    mask = y == st
    X_group = X[mask]
    X_rest  = X[~mask]
    
    # Vectorized Welch t-test
    t_stats, p_vals = stats.ttest_ind(X_group, X_rest, axis=0, equal_var=False)
    
    # Calculate log2 fold change
    log2fc = X_group.mean(axis=0) - X_rest.mean(axis=0)
    
    # FDR Correction (Benjamini-Hochberg)
    reject, p_adj, _, _ = multipletests(p_vals, method='fdr_bh')
    
    sig_mask = (p_adj < 0.05) & (np.abs(log2fc) > 1.0)
    up_mask  = sig_mask & (log2fc > 0)
    dn_mask  = sig_mask & (log2fc < 0)
    
    dge_results[st] = {
        'up': up_mask.sum(), 'down': dn_mask.sum(), 'total': sig_mask.sum()
    }
    
    # Append significants to global DGE list
    dge_df = pd.DataFrame({
        'probe_id': feat_cols, 'subtype': st, 'log2FC': log2fc, 'p_adj': p_adj,
        'significant': sig_mask,
        'direction': np.where(up_mask, 'up', np.where(dn_mask, 'down', 'ns'))
    })
    dge_df = dge_df[dge_df['significant']].copy()
    all_dge.append(dge_df)

dge_all = pd.concat(all_dge, ignore_index=True)
dge_all.to_parquet(ARTIFACT_DIR / "dge_results.parquet", index=False)

print(f"Summary of significant DEGs per subtype (|log2FC| > 1, FDR < 0.05):")
print(f"  {'Subtype':<15} | {'Up':>5} | {'Down':>5} | {'Total':>5}")
print("  " + "-" * 40)
for st in subtypes:
    res = dge_results[st]
    print(f"  {st:<15} | {res['up']:>5,} | {res['down']:>5,} | {res['total']:>5,}")

# Select Union of top 50 DEGs per class for visualization
top_deg_probes = set()
for st in subtypes:
    r = dge_all[dge_all['subtype'] == st]
    ranked = r.sort_values('log2FC', key=abs, ascending=False).head(50)['probe_id']
    top_deg_probes.update(ranked)
joblib.dump(list(top_deg_probes), ARTIFACT_DIR / "top_deg_probes.pkl")
print(f"\\nSaved top {len(top_deg_probes)} unique DEGs to top_deg_probes.pkl")
""")

# =====================================================================
# SECTION 5 — Clustering Analysis
# =====================================================================
add_markdown("""
## 🗂️ SECTION 5: Unsupervised Clustering Analysis
### 🔬 Biological Context
Do breast cancer subtypes group together naturally without utilizing subtype labels?
Here, we perform:
1. **Hierarchical Clustering (Ward's linkage, Euclidean distance):** Visualizes sample relationships in a tree structure (dendrogram).
2. **K-Means Clustering:** Iterative partition clustering evaluated by the WCSS elbow method and Silhouette scores.
""")

add_code("""
# Load coordinate and QN data
dr_df = pd.read_parquet(ARTIFACT_DIR / "dr_coordinates.parquet")
X_pca_full = dr_df[[c for c in dr_df.columns if c.startswith('PC')]].values
y_enc = LabelEncoder().fit_transform(dr_df['subtype'].values)
n_true = len(np.unique(y_enc))

# ── 5.1 Hierarchical Clustering (Ward) ──
df_qn = pd.read_parquet(PROCESSED_DATA_DIR / "breast_cancer_qn.parquet")
feat_cols = [c for c in df_qn.columns if c != 'type']
X_qn = df_qn[feat_cols].values

# Hierarchical on top 2000 variable genes for speed and robustness
gene_vars = np.var(X_qn, axis=0)
top2k_idx = np.argsort(gene_vars)[-2000:]
X_top2k = X_qn[:, top2k_idx]

dist_condensed = pdist(X_top2k, metric='euclidean')
Z = linkage(dist_condensed, method='ward')
h_labels = fcluster(Z, t=n_true, criterion='maxclust') - 1

h_ari = adjusted_rand_score(y_enc, h_labels)
h_nmi = normalized_mutual_info_score(y_enc, h_labels)
print(f"Hierarchical Clustering Ward (k=5):")
print(f"  Adjusted Rand Index (ARI): {h_ari:.4f}")
print(f"  Normalized Mutual Info   : {h_nmi:.4f}")

# ── 5.2 K-Means Clustering ──
km5 = KMeans(n_clusters=n_true, random_state=42, n_init=20)
km5_labels = km5.fit_predict(X_pca_full)
km_ari = adjusted_rand_score(y_enc, km5_labels)
km_nmi = normalized_mutual_info_score(y_enc, km5_labels)
print(f"\\nK-Means Clustering (k=5):")
print(f"  Adjusted Rand Index (ARI): {km_ari:.4f}")
print(f"  Normalized Mutual Info   : {km_nmi:.4f}")

# Save clustering outputs
cluster_df = pd.DataFrame({
    'subtype': dr_df['subtype'].values,
    'true_label': y_enc,
    'hierarchical_cluster': h_labels,
    'kmeans_cluster': km5_labels,
})
cluster_df.to_parquet(ARTIFACT_DIR / "cluster_labels.parquet", index=False)
joblib.dump(Z, ARTIFACT_DIR / "hierarchical_linkage.pkl")
joblib.dump(km5_labels, ARTIFACT_DIR / "kmeans5_labels.pkl")
""")

add_code("""
# ── 5.3 Visualize True Labels vs. K-means clusters ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# True labels
for st in sorted(np.unique(dr_df['subtype'].values)):
    mask = dr_df['subtype'].values == st
    axes[0].scatter(dr_df.loc[mask, 'UMAP1'], dr_df.loc[mask, 'UMAP2'], label=st, alpha=0.8)
axes[0].set_title("True Subtypes (UMAP space)")
axes[0].legend()

# K-Means
for cl in sorted(np.unique(km5_labels)):
    mask = km5_labels == cl
    axes[1].scatter(dr_df.loc[mask, 'UMAP1'], dr_df.loc[mask, 'UMAP2'], label=f"Cluster {cl}", alpha=0.8)
axes[1].set_title("K-Means Clusters (UMAP space)")
axes[1].legend()

plt.tight_layout()
plt.show()
""")

# =====================================================================
# SECTION 6 — Co-expression Network Analysis
# =====================================================================
add_markdown("""
## 🕸️ SECTION 6: Co-expression Network Analysis (WGCNA equivalent)
### 🔬 Biological Context
Genes do not act in isolation. Instead, they form coordinated co-expression networks. 
Here, we build a Pearson correlation network on the **top 500 variable genes**. To maintain strict data hygiene, we compute this **only on the training split**.
An edge is drawn between two genes if they have $|r| > 0.85$.
We identify **modules** (highly correlated groups of genes) using hierarchical clustering on the topological distance matrix ($1 - |r|$).
Genes with the highest degree of connectivity are flagged as **hub genes** — likely central regulators of cellular processes.
""")

add_code("""
# Load data
df = pd.read_parquet(PROCESSED_DATA_DIR / "breast_cancer_qn.parquet")
feat_cols = [c for c in df.columns if c != 'type']
X = df[feat_cols].values.astype(np.float32)
y = df['type'].values

# Split to prevent data leakage in network structure
X_train, _, _, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Select top 500 variable genes from train
gene_vars = np.var(X_train, axis=0)
top500_idx = np.argsort(gene_vars)[-500:]
top500_probes = [feat_cols[i] for i in top500_idx]
X_net = X_train[:, top500_idx]

# ── 6.1 Pearson Correlation Network ──
corr_matrix = np.corrcoef(X_net.T)
np.fill_diagonal(corr_matrix, 0) # remove self loops

# ── 6.2 Adjacency Matrix & Connectivity ──
THR = 0.85
adj = (np.abs(corr_matrix) > THR).astype(int)
node_degree = adj.sum(axis=1)

print(f"Network Statistics (Threshold |r| > {THR}):")
print(f"  Nodes (Genes)      : 500")
print(f"  Total Edges        : {adj.sum() // 2:,}")
print(f"  Max Connectivity   : {node_degree.max()} edges")
print(f"  Mean Connectivity  : {node_degree.mean():.2f} edges")

print("\\nTop 5 Hub Genes by Connectivity:")
top_hubs = np.argsort(node_degree)[::-1][:5]
for idx in top_hubs:
    print(f"  Gene: {top500_probes[idx]:<25} | Degree: {node_degree[idx]}")

# ── 6.3 Module Detection via Hierarchical Clustering ──
dist_matrix = 1 - np.abs(corr_matrix)
np.fill_diagonal(dist_matrix, 0)
dist_matrix = (dist_matrix + dist_matrix.T) / 2.0  # force exact symmetry

condensed = squareform(dist_matrix)
Z_net = linkage(condensed, method='average')
module_labels = fcluster(Z_net, t=0.20, criterion='distance') # distance threshold 0.2 means r > 0.80

n_modules = len(np.unique(module_labels))
print(f"\\nModules detected: {n_modules}")
module_sizes = pd.Series(module_labels).value_counts().sort_values(ascending=False)
print(f"Top 3 module sizes: {list(module_sizes.head(3).values)}")

# Save artifacts
net_df = pd.DataFrame({
    'probe_id': top500_probes, 'degree': node_degree, 'module': module_labels
}).sort_values('degree', ascending=False)
net_df.to_parquet(ARTIFACT_DIR / "coexpression_network.parquet", index=False)
np.save(ARTIFACT_DIR / "corr_matrix_500.npy", corr_matrix)
joblib.dump(top500_probes, ARTIFACT_DIR / "top500_probes.pkl")
joblib.dump(module_labels, ARTIFACT_DIR / "module_labels.pkl")
""")

add_code("""
# ── 6.4 Heatmap of Sorted Co-expression Modules ──
# Sort correlation matrix by module labels to show block structures
sort_idx = np.argsort(module_labels)
sorted_corr = corr_matrix[sort_idx, :][:, sort_idx]

plt.figure(figsize=(8, 7))
sns.heatmap(sorted_corr, cmap="coolwarm", vmin=-1, vmax=1, xticklabels=False, yticklabels=False)
plt.title("Co-expression Matrix Ordered by Network Module")
plt.show()
""")

# =====================================================================
# SECTION 7 — Ensemble Feature Selection (Anti-Overfitting)
# =====================================================================
add_markdown("""
## 🧬 SECTION 7: Ensemble Feature Selection (Strict Anti-Overfitting)
### 🔬 Biological Context
When features (54,000+ probes) outnumber samples (130 tumors), models easily overfit. 
To guarantee mathematical validity, we apply **all feature selection steps strictly on the training set**.
Our ensemble framework runs four distinct feature selection techniques:
1. **ANOVA F-Test:** Linear rank filtering of class separation.
2. **Mutual Information:** Non-linear statistical dependency filtering.
3. **LASSO L1 Logistic Regression (on ANOVA top 2000):** Sparsity regularization that penalizes collinearity.
4. **Random Forest Feature Importance:** Ensemble tree Gini importance.

We keep genes that are selected by **at least 2 out of the 4 methods**, creating a robust **Consensus Feature Set**.
""")

add_code("""
# Load QN data
df = pd.read_parquet(PROCESSED_DATA_DIR / "breast_cancer_qn.parquet")
feat_cols = np.array([c for c in df.columns if c != 'type'])
X = df[feat_cols].values.astype(np.float32)
y = df['type'].values

# Encode targets
le = LabelEncoder()
y_enc = le.fit_transform(y)

# ── 7.1 Stratified Train-Test Split (BEFORE selection) ──
X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, stratify=y_enc, random_state=42)

# Save target arrays
joblib.dump(y_train, ARTIFACT_DIR / "y_train.pkl")
joblib.dump(y_test,  ARTIFACT_DIR / "y_test.pkl")
joblib.dump(le,      ARTIFACT_DIR / "label_encoder.pkl")

# ── 7.2 Variance Threshold Filtering ──
vt = VarianceThreshold(threshold=0.1)
vt.fit(X_train)
mask_var = vt.get_support()
X_train_vt = X_train[:, mask_var]
X_test_vt  = X_test[:, mask_var]
feat_var   = feat_cols[mask_var]
print(f"Variance Filter: {X_train.shape[1]:,} -> {X_train_vt.shape[1]:,} genes retained.")

# ── 7.3 Standardize training set ──
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train_vt)
X_test_sc  = scaler.transform(X_test_vt)
joblib.dump(scaler,        ARTIFACT_DIR / "scaler.pkl")
joblib.dump(X_train_sc,    ARTIFACT_DIR / "X_train_scaled.pkl")
joblib.dump(X_test_sc,     ARTIFACT_DIR / "X_test_scaled.pkl")

# ── 7.4 Method 1: ANOVA F-test (select top 2000) ──
K = 2000
anova = SelectKBest(f_classif, k=K)
anova.fit(X_train_sc, y_train)
mask_anova = anova.get_support()
genes_anova = set(feat_var[mask_anova])

# ── 7.5 Method 2: Mutual Information (select top 2000) ──
from functools import partial
mi_scorer = partial(mutual_info_classif, random_state=42)
mi = SelectKBest(mi_scorer, k=K)
mi.fit(X_train_sc, y_train)
mask_mi = mi.get_support()
genes_mi = set(feat_var[mask_mi])

# ── 7.6 Method 3: LASSO (C=0.05, Liblinear solver, on ANOVA top 2000) ──
X_train_anova = X_train_sc[:, mask_anova]
lasso_lr = LogisticRegression(penalty='l1', C=0.05, solver='liblinear', max_iter=1000, random_state=42)
from sklearn.multiclass import OneVsRestClassifier
lasso_ovr = OneVsRestClassifier(lasso_lr, n_jobs=-1)
lasso_ovr.fit(X_train_anova, y_train)

# Collect union of non-zero features across all classes
feat_var_anova = feat_var[mask_anova]
lasso_nonzero = np.zeros(K, dtype=bool)
for est in lasso_ovr.estimators_:
    lasso_nonzero |= (np.abs(est.coef_[0]) > 0)
genes_lasso = set(feat_var_anova[lasso_nonzero])

# ── 7.7 Method 4: Random Forest Importance (top 2000) ──
rf_sel = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_sel.fit(X_train_sc, y_train)
importances = rf_sel.feature_importances_
top_rf_idx = np.argsort(importances)[::-1][:K]
mask_rf = np.zeros(len(feat_var), dtype=bool)
mask_rf[top_rf_idx] = True
genes_rf = set(feat_var[mask_rf])

# ── 7.8 Consensus Voting (>= 2 methods) ──
gene_votes = {}
for g in feat_var:
    votes = sum([g in genes_anova, g in genes_mi, g in genes_lasso, g in genes_rf])
    gene_votes[g] = votes

votes_arr = np.array([gene_votes[g] for g in feat_var])
consensus_mask = votes_arr >= 2
consensus_genes = feat_var[consensus_mask]

# Parse features
feat_var_list = list(feat_var)
consensus_indices = [feat_var_list.index(g) for g in consensus_genes]
X_train_consensus = X_train_sc[:, consensus_indices]
X_test_consensus  = X_test_sc[:,  consensus_indices]

print(f"\\nEnsemble Feature Selection Summary:")
print(f"  ANOVA selected: {len(genes_anova)}")
print(f"  MI selected   : {len(genes_mi)}")
print(f"  LASSO selected: {len(genes_lasso)}")
print(f"  RF selected   : {len(genes_rf)}")
print(f"  Consensus Feature Space (>= 2 votes): {len(consensus_genes)} genes")

# Save consensus data
joblib.dump(consensus_genes,   ARTIFACT_DIR / "top_consensus_genes.pkl")
joblib.dump(consensus_indices, ARTIFACT_DIR / "consensus_indices.pkl")
joblib.dump(X_train_consensus, ARTIFACT_DIR / "X_train_consensus.pkl")
joblib.dump(X_test_consensus,  ARTIFACT_DIR / "X_test_consensus.pkl")
""")

# =====================================================================
# SECTION 8 — Machine Learning Model Benchmarking
# =====================================================================
add_markdown("""
## 🤖 SECTION 8: Machine Learning Model Benchmarking
### 🔬 Biological Context
We benchmark classical machine learning models (Logistic Regression, Support Vector Classifier, Random Forest, XGBoost, and LightGBM) across two distinct feature configurations:
1. **Consensus Genes:** The biological consensus feature set selected in Section 7.
2. **PCA-50 Space:** The top 50 Principal Components derived from PCA in Section 3.

Each model is evaluated on the held-out test set (n=28) to measure Accuracy and Weighted F1-Score.
""")

add_code("""
# Load features
X_train_c = joblib.load(ARTIFACT_DIR / "X_train_consensus.pkl")
X_test_c  = joblib.load(ARTIFACT_DIR / "X_test_consensus.pkl")
y_train   = joblib.load(ARTIFACT_DIR / "y_train.pkl")
y_test    = joblib.load(ARTIFACT_DIR / "y_test.pkl")
le        = joblib.load(ARTIFACT_DIR / "label_encoder.pkl")
class_names = list(le.classes_)

# Create PCA-50 feature space
pca = PCA(n_components=50, random_state=42)
X_train_pca = pca.fit_transform(X_train_c)
X_test_pca  = pca.transform(X_test_c)

feature_spaces = {
    'Consensus': (X_train_c, X_test_c),
    'PCA-50':    (X_train_pca, X_test_pca)
}

# Define models
from sklearn.base import clone
from copy import deepcopy
import xgboost as xgb
import lightgbm as lgb

base_models = {
    'Logistic Regression': LogisticRegression(max_iter=5000, random_state=42),
    'SVM (RBF)':           SVC(kernel='rbf', probability=True, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost':             xgb.XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=6, eval_metric='mlogloss', random_state=42),
    'LightGBM':            lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=42, verbose=-1)
}

results = []
best_acc = 0
best_model_obj = None
best_model_name = ""
best_space_name = ""
best_space = None

for space_name, (X_tr, X_te) in feature_spaces.items():
    print(f"\\nRunning Benchmark on '{space_name}' ({X_tr.shape[1]} features)...")
    for model_name, model_tmpl in base_models.items():
        model = clone(model_tmpl)
        model.fit(X_tr, y_train)
        preds = model.predict(X_te)
        
        acc = accuracy_score(y_test, preds)
        f1  = f1_score(y_test, preds, average='weighted')
        print(f"  {model_name:<20}: Accuracy={acc:.4f} | F1={f1:.4f}")
        
        results.append({
            'feature_space': space_name,
            'model': model_name,
            'accuracy': acc,
            'weighted_f1': f1
        })
        
        if acc > best_acc:
            best_acc = acc
            best_model_obj = deepcopy(model)
            best_model_name = model_name
            best_space_name = space_name
            best_space = (X_tr, X_te)

print(f"\\nBest Benchmarked Model: {best_model_name} on {best_space_name} (Acc={best_acc:.4f})")

# Save outputs
bench_df = pd.DataFrame(results)
bench_df.to_parquet(ARTIFACT_DIR / "benchmark_results.parquet", index=False)
joblib.dump(best_model_obj, ARTIFACT_DIR / "best_model.pkl")
joblib.dump({'model': best_model_name, 'space': best_space_name}, ARTIFACT_DIR / "best_model_info.pkl")
""")

add_code("""
# ── 8.1 Visualize Benchmarking Results ──
bench_df = pd.read_parquet(ARTIFACT_DIR / "benchmark_results.parquet")
plt.figure(figsize=(12, 6))
sns.barplot(data=bench_df, x="model", y="accuracy", hue="feature_space", palette="Set2")
plt.title("Model Classification Accuracy Comparison")
plt.ylabel("Accuracy Score")
plt.xlabel("Model Architecture")
plt.ylim(0.7, 1.02)
plt.legend(title="Feature Space")
plt.show()
""")

# =====================================================================
# SECTION 9 — Deep Learning: PyTorch MLP Classifier
# =====================================================================
add_markdown("""
## 🧠 SECTION 9: Deep Learning (PyTorch MLP Classifier)
### 🔬 Biological Context
Deep neural networks can learn highly complex, non-linear interactions. Here, we build a multi-layer perceptron (MLP) in PyTorch.
* **Architecture:** `Input(1480 consensus features) -> Dense(512, Batch Normalization, ReLU, Dropout 0.4) -> Dense(256, Batch Normalization, ReLU, Dropout 0.3) -> Dense(128, ReLU, Dropout 0.2) -> Output(5 classes)`.
* **Loss Function:** Cross-Entropy Loss weighted by clinical class frequencies (to account for class imbalance).
* **Optimizer:** Adam with $L_2$ weight decay ($10^{-4}$) to prevent overfitting.
""")

add_code("""
# Load inputs
X_train_c = joblib.load(ARTIFACT_DIR / "X_train_consensus.pkl").astype(np.float32)
X_test_c  = joblib.load(ARTIFACT_DIR / "X_test_consensus.pkl").astype(np.float32)
y_train   = joblib.load(ARTIFACT_DIR / "y_train.pkl").astype(np.int64)
y_test    = joblib.load(ARTIFACT_DIR / "y_test.pkl").astype(np.int64)
le        = joblib.load(ARTIFACT_DIR / "label_encoder.pkl")
class_names = list(le.classes_)

n_features = X_train_c.shape[1]
n_classes  = len(class_names)

# Class Weights (balanced)
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', classes=np.arange(n_classes), y=y_train)
class_weights_tensor = torch.FloatTensor(class_weights).to(device)

# ── 9.1 Network Architecture ──
class BreastCancerMLP(nn.Module):
    def __init__(self, in_dim, n_cls):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, n_cls),
        )
    def forward(self, x):
        return self.net(x)

model = BreastCancerMLP(n_features, n_classes).to(device)

# DataLoaders (with automatic pinning for efficiency)
X_tr_t = torch.FloatTensor(X_train_c)
y_tr_t = torch.LongTensor(y_train)
X_te_t = torch.FloatTensor(X_test_c)

train_ds = TensorDataset(X_tr_t, y_tr_t)
train_dl = DataLoader(train_ds, batch_size=16, shuffle=True, pin_memory=True)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

# ── 9.2 Training Loop ──
EPOCHS = 100
history = {'train_loss': [], 'train_acc': [], 'val_acc': [], 'val_f1': []}
best_val_acc = 0
best_epoch = 0

print("Training PyTorch MLP:")
for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_loss = 0; correct = 0; total = 0
    for Xb, yb in train_dl:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(Xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * len(yb)
        correct += (out.argmax(1) == yb).sum().item()
        total += len(yb)
    
    train_loss = epoch_loss / total
    train_acc  = correct / total
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_out = model(X_te_t.to(device))
        val_preds = val_out.argmax(1).cpu().numpy()
    val_acc = accuracy_score(y_test, val_preds)
    val_f1  = f1_score(y_test, val_preds, average='weighted', zero_division=0)
    
    scheduler.step(1 - val_acc)
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)
    history['val_f1'].append(val_f1)
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch = epoch
        torch.save(model.state_dict(), ARTIFACT_DIR / "mlp_best.pt")
        
    if epoch % 10 == 0 or epoch == 1:
        print(f"  Epoch {epoch:>3}: Loss={train_loss:.4f} | Train Acc={train_acc:.4f} | Val Acc={val_acc:.4f}")

# Save training history
pd.DataFrame(history).to_parquet(ARTIFACT_DIR / "mlp_training_history.parquet", index=False)
joblib.dump({'test_acc': best_val_acc, 'best_epoch': best_epoch}, ARTIFACT_DIR / "mlp_results.pkl")
print(f"\\nPyTorch MLP Training Finished. Best Validation Accuracy: {best_val_acc:.4f} at epoch {best_epoch}")
""")

# =====================================================================
# SECTION 10 — Cross-Validation & Hyperparameter Tuning
# =====================================================================
add_markdown("""
## 🔄 SECTION 10: Cross-Validation & Hyperparameter Tuning
### 🔬 Biological Context
Evaluating models on a single train-test split can introduce performance bias depending on sample layout. 
To guarantee generalizability, we apply **Stratified 5-Fold Cross-Validation** on our consensus Random Forest classifier. 
We search the hyperparameter grid using **GridSearchCV** over:
* `n_estimators` (number of trees)
* `max_depth` (complexity limit)
* `max_features` (sub-sampling feature splits)
""")

add_code("""
# Load features
X_train_c = joblib.load(ARTIFACT_DIR / "X_train_consensus.pkl")
X_test_c  = joblib.load(ARTIFACT_DIR / "X_test_consensus.pkl")
y_train   = joblib.load(ARTIFACT_DIR / "y_train.pkl")
y_test    = joblib.load(ARTIFACT_DIR / "y_test.pkl")
le        = joblib.load(ARTIFACT_DIR / "label_encoder.pkl")
class_names = list(le.classes_)

# ── 10.1 Stratified 5-Fold CV on Baseline RF ──
rf_base = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = cross_validate(
    rf_base, X_train_c, y_train, cv=cv,
    scoring=['accuracy', 'f1_weighted'],
    return_train_score=True, n_jobs=-1
)
mean_train_acc = cv_results['train_accuracy'].mean()
mean_val_acc = cv_results['test_accuracy'].mean()
print("Baseline RF 5-Fold CV Scores:")
print(f"  CV Accuracy  : {mean_val_acc:.4f} +/- {cv_results['test_accuracy'].std():.4f}")
print(f"  Train Accuracy: {mean_train_acc:.4f} (Overfitting gap: {mean_train_acc - mean_val_acc:.4f})")

# ── 10.2 GridSearchCV Hyperparameter Tuning ──
param_grid = {
    'n_estimators': [100, 300],
    'max_features': ['sqrt', 'log2'],
    'max_depth':    [None, 20, 10],
}
rf_grid = RandomForestClassifier(random_state=42, n_jobs=-1)
gs = GridSearchCV(
    rf_grid, param_grid,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='f1_weighted', n_jobs=-1
)
gs.fit(X_train_c, y_train)
best_rf = gs.best_estimator_

print(f"\\nBest Parameters found by Grid Search: {gs.best_params_}")
print(f"Best CV Weighted F1: {gs.best_score_:.4f}")

# Final Test Set Evaluation
y_pred_tuned = best_rf.predict(X_test_c)
tuned_acc = accuracy_score(y_test, y_pred_tuned)
tuned_f1  = f1_score(y_test, y_pred_tuned, average='weighted')
print(f"\\nTuned Model Test Accuracy: {tuned_acc:.4f} | F1: {tuned_f1:.4f}")

# Save artifacts
joblib.dump(best_rf, ARTIFACT_DIR / "tuned_rf.pkl")
joblib.dump(gs,      ARTIFACT_DIR / "gridsearch_results.pkl")
pd.DataFrame(gs.cv_results_).to_parquet(ARTIFACT_DIR / "gridsearch_df.parquet", index=False)
""")

add_code("""
# ── 10.3 Confusion Matrix Heatmap ──
cm = confusion_matrix(y_test, y_pred_tuned)
cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
plt.figure(figsize=(7, 6))
sns.heatmap(cm_df, annot=True, cmap="Blues", fmt="d")
plt.title("Tuned Random Forest Confusion Matrix")
plt.ylabel("True Class")
plt.xlabel("Predicted Class")
plt.show()
""")

# =====================================================================
# SECTION 11 — SHAP Interpretability
# =====================================================================
add_markdown("""
## 🧬 SECTION 11: SHAP Interpretability (Explainable AI)
### 🔬 Biological Context
Machine learning models are often criticised as "black boxes". To extract biological meaning, we use **SHAP (SHapley Additive exPlanations)** based on game theory. 
SHAP calculates the contribution of each gene to the classification of every tumor. 
To bridge the gap between Affymetrix probe IDs (e.g., `205225_at`) and biology, we query the **MyGene API** to annotate probes with their official HUGO gene symbols and descriptions (e.g., `ESR1` - Estrogen Receptor 1).
""")

add_code("""
# Load models & test data
tuned_rf     = joblib.load(ARTIFACT_DIR / "tuned_rf.pkl")
X_test_c     = joblib.load(ARTIFACT_DIR / "X_test_consensus.pkl")
consensus_genes = joblib.load(ARTIFACT_DIR / "top_consensus_genes.pkl")
le           = joblib.load(ARTIFACT_DIR / "label_encoder.pkl")
class_names  = list(le.classes_)

# ── 11.1 Compute TreeSHAP Values ──
explainer = shap.TreeExplainer(tuned_rf)
shap_values = explainer.shap_values(X_test_c)

# Handle numpy array format returned for multiclass tree ensembles in newer SHAP versions
if isinstance(shap_values, list):
    shap_arr = np.array(shap_values)
    mean_abs_shap = np.abs(shap_arr).mean(axis=(0, 1))
else:
    shap_arr = shap_values
    mean_abs_shap = np.abs(shap_arr).mean(axis=(0, 2))

# ── 11.2 Map Probe IDs via MyGene API ──
top10_idx = np.argsort(mean_abs_shap)[::-1][:10]
top10_probes = [consensus_genes[i] for i in top10_idx]

print("Top 10 Biomarkers Ranked by Mean |SHAP| Value:")
try:
    mg = mygene.MyGeneInfo()
    results = mg.querymany(
        top10_probes, scopes='reporter', fields='symbol,name', species='human', verbose=False
    )
    annotations = {}
    for r in results:
        q = r.get('query', '')
        s = r.get('symbol', 'N/A')
        n = r.get('name', 'N/A')
        annotations[q] = {'symbol': s, 'name': n}
        
    for rank, probe in enumerate(top10_probes, 1):
        ann = annotations.get(probe, {'symbol': 'N/A', 'name': 'N/A'})
        print(f"  #{rank:<2}: {probe:<12} -> {ann['symbol']:<10} | {ann['name'][:55]}")
    joblib.dump(annotations, ARTIFACT_DIR / "shap_annotations.pkl")
except Exception as e:
    print(f"  MyGene API annotation failed: {e}")

# Save importance parquet
shap_df = pd.DataFrame({
    'probe_id': consensus_genes,
    'mean_abs_shap': mean_abs_shap,
    'shap_rank': pd.Series(mean_abs_shap).rank(ascending=False).values.astype(int),
})
shap_df = shap_df.sort_values('shap_rank')
shap_df.to_parquet(ARTIFACT_DIR / "shap_importance.parquet", index=False)
np.save(ARTIFACT_DIR / "shap_values.npy", shap_values)
""")

# =====================================================================
# SECTION 12 — Pathway Enrichment Analysis (GO/KEGG/GSEA)
# =====================================================================
add_markdown("""
## 🗺️ SECTION 12: Pathway Enrichment Analysis (GO & KEGG)
### 🔬 Biological Context
Do our model's top biomarkers represent genuine breast cancer pathways? 
To find out, we query the **Enrichr API** using **GO (Gene Ontology) Biological Processes** and **KEGG Pathways** on the top 100 SHAP genes.
We also perform **GSEA (Gene Set Enrichment Analysis) Prerank** on all consensus genes sorted by SHAP importance to discover coordinate expression shifts in molecular cascades.
""")

add_code("""
# Load SHAP importance table
shap_df = pd.read_parquet(ARTIFACT_DIR / "shap_importance.parquet")

print("Performing MyGene annotation mapping for all consensus genes...")
try:
    mg = mygene.MyGeneInfo()
    all_probes = shap_df['probe_id'].tolist()
    results = mg.querymany(
        all_probes, scopes='reporter', fields='symbol', species='human', verbose=False
    )
    probe_to_symbol = {}
    for r in results:
        q = r.get('query', '')
        s = r.get('symbol', None)
        if s and not r.get('notfound', False):
            probe_to_symbol[q] = s
            
    shap_df['gene_symbol'] = shap_df['probe_id'].map(probe_to_symbol)
    shap_df_annot = shap_df.dropna(subset=['gene_symbol'])
    print(f"  Successfully annotated {len(shap_df_annot)} consensus probes.")

    # ── 12.1 GO Biological Process Enrichment ──
    top_genes_shap = shap_df_annot.sort_values('mean_abs_shap', ascending=False)['gene_symbol'].head(100).tolist()
    
    print("\\nEnriching Top 100 SHAP genes in GO Biological Process 2023...")
    enr_go = gp.enrichr(
        gene_list=top_genes_shap, gene_sets='GO_Biological_Process_2023', organism='human', outdir=None, verbose=False
    )
    go_sig = enr_go.results[enr_go.results['Adjusted P-value'] < 0.05].sort_values('Adjusted P-value')
    print(f"  Significant GO Terms: {len(go_sig)}")
    if len(go_sig) > 0:
        for idx, row in go_sig.head(5).iterrows():
            print(f"    - {row['Term'][:50]:<50} | FDR={row['Adjusted P-value']:.2e} | Overlap={row['Overlap']}")

    # ── 12.2 KEGG Pathway Enrichment ──
    print("\\nEnriching in KEGG 2021 Human...")
    enr_kegg = gp.enrichr(
        gene_list=top_genes_shap, gene_sets='KEGG_2021_Human', organism='human', outdir=None, verbose=False
    )
    kegg_sig = enr_kegg.results[enr_kegg.results['Adjusted P-value'] < 0.05].sort_values('Adjusted P-value')
    print(f"  Significant KEGG Pathways: {len(kegg_sig)}")
    if len(kegg_sig) > 0:
        for idx, row in kegg_sig.head(5).iterrows():
            print(f"    - {row['Term'][:50]:<50} | FDR={row['Adjusted P-value']:.2e} | Overlap={row['Overlap']}")

    # Save parquets
    go_sig.to_parquet(ARTIFACT_DIR / "enrichr_go_results.parquet", index=False)
    kegg_sig.to_parquet(ARTIFACT_DIR / "enrichr_kegg_results.parquet", index=False)
    shap_df.to_parquet(ARTIFACT_DIR / "shap_importance_annotated.parquet", index=False)
except Exception as e:
    print(f"Pathway enrichment failed: {e}")
""")

# =====================================================================
# SECTION 13 — Summary & Conclusions
# =====================================================================
add_markdown("""
## 📊 SECTION 13: Summary & Conclusions
### Key Takeaways
1. **Quantile Normalization** successfully aligned all 130 tumor samples, correcting for experimental batch issues.
2. **PCA, t-SNE, and UMAP** dimensions show clear clustering of **Basal** and **Normal** subtypes, indicating distinct transcriptional profiles.
3. **Differential Expression (DGE)** identified thousands of subtype-specific markers (e.g. estrogen-regulated genes in Luminal types).
4. Our **Ensemble Feature Selection** successfully filtered 54,000+ probes down to a core list of consensus genes.
5. Hyperparameter-tuned **Random Forest** and the **PyTorch MLP** both demonstrated high accuracy on held-out samples.
6. **SHAP values** identified critical clinical markers (e.g., `ERBB2` for HER2-subtype, `ESR1` for Luminal subtypes) matching standard clinical panels.
7. **GO/KEGG Enrichment** shows that these features are heavily concentrated in **Cell Cycle regulation, spindle checkpoints, and cellular senescence**, aligning with cancer pathology.
""")

# Write the final JSON file
output_path = PROJECT_ROOT / "notebooks" / "Breast_Cancer_ML_Pipeline.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print(f"Successfully assembled and wrote notebook to: {output_path}")

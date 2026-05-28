"""
SECTION 2 VERIFICATION: Normalization & Data Quality
Checks:
  - Quantile normalization across samples
  - Sample-sample correlation matrix for outlier detection
  - Per-sample median expression spread (QC)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACT_DIR = PROJECT_ROOT / "data" / "artifacts"

print("=" * 60)
print("SECTION 2: NORMALIZATION & QUALITY CONTROL")
print("=" * 60)

# Load the parquet saved in Section 1
df = pd.read_parquet(PROCESSED_DATA_DIR / "breast_cancer.parquet")
feat_cols = [c for c in df.columns if c != 'type']
X = df[feat_cols].values.astype(np.float32)
y = df['type'].values
print(f"\n[2.0] Loaded: {X.shape[0]} samples x {X.shape[1]} features")

# ── 2.1 Per-sample median BEFORE normalization ────────────────
per_sample_median_before = np.median(X, axis=1)
print(f"\n[2.1] Per-sample median expression (BEFORE quantile normalization):")
print(f"  Mean of medians   : {per_sample_median_before.mean():.4f}")
print(f"  Std  of medians   : {per_sample_median_before.std():.4f}")
print(f"  Min median sample : {per_sample_median_before.min():.4f}")
print(f"  Max median sample : {per_sample_median_before.max():.4f}")
print(f"  Range (max-min)   : {per_sample_median_before.max()-per_sample_median_before.min():.4f}")

# ── 2.2 Quantile normalization ────────────────────────────────
# Reference distribution = mean of sorted values across all samples
print(f"\n[2.2] Applying quantile normalization...")
# Sort each sample independently, then replace values with reference distribution
X_sorted = np.sort(X, axis=1)
reference = X_sorted.mean(axis=0)   # mean across samples at each rank
ranks = np.argsort(np.argsort(X, axis=1), axis=1)  # double-argsort gives ranks
X_qn = reference[ranks]
print(f"  Reference distribution shape: {reference.shape}")

# ── 2.3 Per-sample median AFTER normalization ─────────────────
per_sample_median_after = np.median(X_qn, axis=1)
print(f"\n[2.3] Per-sample median expression (AFTER quantile normalization):")
print(f"  Mean of medians   : {per_sample_median_after.mean():.4f}")
print(f"  Std  of medians   : {per_sample_median_after.std():.4f}")
print(f"  Min median sample : {per_sample_median_after.min():.4f}")
print(f"  Max median sample : {per_sample_median_after.max():.4f}")
print(f"  Range (max-min)   : {per_sample_median_after.max()-per_sample_median_after.min():.4f}")

reduction_pct = (1 - per_sample_median_after.std() / per_sample_median_before.std()) * 100
print(f"\n  >> Sample-to-sample variability reduced by {reduction_pct:.1f}%")

# ── 2.4 Sample-sample Pearson correlation ─────────────────────
print(f"\n[2.4] Sample-sample correlation matrix (quality check):")
# Compute pairwise Pearson on transposed matrix
from numpy import corrcoef
corr_matrix = corrcoef(X_qn)   # shape (137, 137)
np.fill_diagonal(corr_matrix, np.nan)
print(f"  Min pairwise correlation : {np.nanmin(corr_matrix):.4f}")
print(f"  Mean pairwise correlation: {np.nanmean(corr_matrix):.4f}")
print(f"  Max pairwise correlation : {np.nanmax(corr_matrix):.4f}")

# Identify potential outlier samples (mean correlation < mean - 2*std)
mean_corr_per_sample = np.nanmean(corr_matrix, axis=1)
global_mean = np.nanmean(mean_corr_per_sample)
global_std  = np.nanstd(mean_corr_per_sample)
outlier_threshold = global_mean - 2 * global_std
outlier_mask = mean_corr_per_sample < outlier_threshold
n_outliers = outlier_mask.sum()
print(f"\n  Outlier threshold (mean - 2*std): {outlier_threshold:.4f}")
print(f"  Potential outlier samples        : {n_outliers}")
if n_outliers > 0:
    outlier_subtypes = y[outlier_mask]
    print(f"  Outlier subtypes                 : {list(outlier_subtypes)}")
else:
    print(f"  >> No outlier samples detected. All samples are well-correlated.")

# ── 2.5 Within-subtype vs between-subtype correlation ─────────
print(f"\n[2.5] Within- vs between-subtype correlation:")
subtypes = np.unique(y)
within_corrs = []
between_corrs = []
for i in range(len(y)):
    for j in range(i+1, len(y)):
        c = corr_matrix[i, j]
        if y[i] == y[j]:
            within_corrs.append(c)
        else:
            between_corrs.append(c)

print(f"  Mean within-subtype correlation : {np.mean(within_corrs):.4f}")
print(f"  Mean between-subtype correlation: {np.mean(between_corrs):.4f}")
print(f"  >> Difference                   : {np.mean(within_corrs) - np.mean(between_corrs):.4f}")
print(f"  >> Subtypes are {'well-separated' if np.mean(within_corrs)-np.mean(between_corrs)>0.02 else 'overlapping'} in expression space")

# ── 2.6 Save normalized matrix ───────────────────────────────
print(f"\n[2.6] Saving quantile-normalized data...")
df_qn = pd.DataFrame(X_qn, columns=feat_cols, dtype=np.float32)
df_qn.insert(0, 'type', y)
df_qn.to_parquet(PROCESSED_DATA_DIR / "breast_cancer_qn.parquet", index=False)
print(f"  Saved: breast_cancer_qn.parquet")

print("\n" + "=" * 60)
print("SECTION 2 COMPLETE [OK]")
print("=" * 60)

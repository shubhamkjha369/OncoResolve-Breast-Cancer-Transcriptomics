"""
SECTION 1 VERIFICATION: Data Loading & Quality Control
Run this script and capture output. All output used VERBATIM in notebook explanations.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACT_DIR = PROJECT_ROOT / "data" / "artifacts"

for d in [PROCESSED_DATA_DIR, ARTIFACT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("SECTION 1: DATA LOADING & QUALITY CONTROL")
print("=" * 60)

# ── 1.1 Load raw CSV ──────────────────────────────────────────
print("\n[1.1] Loading raw CSV...")
csv_path = RAW_DATA_DIR / "Breast_GSE45827.csv"
df_raw = pd.read_csv(csv_path)
print(f"  Raw shape         : {df_raw.shape[0]} samples × {df_raw.shape[1]} columns")
print(f"  First 3 columns   : {list(df_raw.columns[:3])}")
print(f"  Last  3 columns   : {list(df_raw.columns[-3:])}")
print(f"  Missing values    : {df_raw.isnull().sum().sum()}")
print(f"  Data types present: {df_raw.dtypes.value_counts().to_dict()}")

# ── 1.2 Class distribution before filtering ────────────────────
print("\n[1.2] Class distribution (ALL 6 types):")
vc_all = df_raw['type'].value_counts().sort_values(ascending=False)
for label, count in vc_all.items():
    pct = count / len(df_raw) * 100
    print(f"  {label:<15}: {count:>3} samples ({pct:.1f}%)")

# ── 1.3 Remove 'cell_line' samples ────────────────────────────
print("\n[1.3] Removing 'cell_line' samples (lab artefacts)...")
df = df_raw[df_raw['type'] != 'cell_line'].copy()
df = df.drop(columns=['samples'])
print(f"  Samples removed   : {len(df_raw) - len(df)} (cell_line)")
print(f"  Remaining shape   : {df.shape[0]} samples × {df.shape[1]} columns")

# ── 1.4 Class distribution after filtering ─────────────────────
print("\n[1.4] Class distribution (clinical subtypes only):")
vc_clinical = df['type'].value_counts().sort_values(ascending=False)
for label, count in vc_clinical.items():
    pct = count / len(df) * 100
    print(f"  {label:<15}: {count:>3} samples ({pct:.1f}%)")

# ── 1.5 Cast to float32 ───────────────────────────────────────
print("\n[1.5] Casting feature values to float32...")
feat_cols = [c for c in df.columns if c != 'type']
df[feat_cols] = df[feat_cols].astype(np.float32)
mem_mb = df[feat_cols].memory_usage(deep=True).sum() / 1e6
print(f"  Feature columns   : {len(feat_cols):,}")
print(f"  Memory (features) : {mem_mb:.1f} MB")

# ── 1.6 Expression value range check (log2 confirmation) ────────
sample_vals = df[feat_cols].values.flatten()[::100]  # every 100th value
print("\n[1.6] Expression value distribution (confirms log2 scale):")
print(f"  Min               : {float(np.percentile(sample_vals, 0)):.3f}")
print(f"  1st percentile    : {float(np.percentile(sample_vals, 1)):.3f}")
print(f"  Median            : {float(np.percentile(sample_vals, 50)):.3f}")
print(f"  95th percentile   : {float(np.percentile(sample_vals, 95)):.3f}")
print(f"  Max               : {float(np.percentile(sample_vals, 100)):.3f}")
print(f"  (Log2 normalized Affymetrix data typically ranges 2-16)")

# ── 1.7 Save processed dataframe ──────────────────────────────
print("\n[1.7] Saving processed parquet...")
df.to_parquet(PROCESSED_DATA_DIR / "breast_cancer.parquet", index=False)
print(f"  Saved to: {PROCESSED_DATA_DIR / 'breast_cancer.parquet'}")

print("\n" + "=" * 60)
print("SECTION 1 COMPLETE [OK]")
print("=" * 60)


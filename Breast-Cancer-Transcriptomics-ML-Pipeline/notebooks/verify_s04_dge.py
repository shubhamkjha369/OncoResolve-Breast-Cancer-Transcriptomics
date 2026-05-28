"""
SECTION 4 VERIFICATION: Differential Gene Expression Analysis
- One-vs-rest Welch t-test per subtype
- Benjamini-Hochberg FDR correction
- Thresholds: FDR < 0.05, |log2FC| > 1.0
All outputs used verbatim in notebook explanations.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

from scipy import stats
from statsmodels.stats.multitest import multipletests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACT_DIR = PROJECT_ROOT / "data" / "artifacts"

print("=" * 60)
print("SECTION 4: DIFFERENTIAL GENE EXPRESSION ANALYSIS")
print("=" * 60)

# Load data
df = pd.read_parquet(PROCESSED_DATA_DIR / "breast_cancer_qn.parquet")
feat_cols = [c for c in df.columns if c != 'type']
X = df[feat_cols].values.astype(np.float32)
y = df['type'].values
subtypes = sorted(np.unique(y))
n_genes = len(feat_cols)

print(f"\n[4.0] Data: {X.shape[0]} samples x {n_genes:,} genes")
print(f"      Subtypes: {subtypes}")
print(f"      Test: Welch t-test (one-vs-rest per subtype)")
print(f"      FDR correction: Benjamini-Hochberg")
print(f"      Thresholds: FDR < 0.05, |log2FC| > 1.0")

dge_results = {}

for subtype in subtypes:
    mask = y == subtype
    X_group = X[mask]
    X_rest  = X[~mask]

    n_in  = mask.sum()
    n_out = (~mask).sum()

    # ── Welch t-test (vectorized across all genes) ────────────
    t_stats, p_vals = stats.ttest_ind(X_group, X_rest, axis=0, equal_var=False)

    # ── Log2 fold change (group mean / rest mean, in log2 space) ──
    # Data is already log2-normalized, so difference = log2FC
    log2fc = X_group.mean(axis=0) - X_rest.mean(axis=0)

    # ── BH FDR correction ─────────────────────────────────────
    reject, p_adj, _, _ = multipletests(p_vals, method='fdr_bh')

    # ── Apply thresholds ──────────────────────────────────────
    sig_mask = (p_adj < 0.05) & (np.abs(log2fc) > 1.0)
    up_mask  = sig_mask & (log2fc > 0)
    dn_mask  = sig_mask & (log2fc < 0)

    dge_results[subtype] = {
        'log2fc': log2fc,
        'p_adj':  p_adj,
        'sig':    sig_mask,
        'up':     up_mask,
        'down':   dn_mask,
        'n_in':   n_in,
        'n_out':  n_out,
    }

    print(f"\n[4.{subtypes.index(subtype)+1}] {subtype} vs rest (n={n_in} vs {n_out}):")
    print(f"  Total sig. genes (FDR<0.05 & |FC|>1): {sig_mask.sum():>5,}")
    print(f"    UP-regulated in {subtype:<12}        : {up_mask.sum():>5,}")
    print(f"    DOWN-regulated in {subtype:<12}      : {dn_mask.sum():>5,}")

    # Top 5 upregulated
    up_idx = np.where(up_mask)[0]
    if len(up_idx) > 0:
        top_up = up_idx[np.argsort(log2fc[up_idx])[::-1][:5]]
        print(f"  Top 5 upregulated probes:")
        for idx in top_up:
            print(f"    {feat_cols[idx]:<30} log2FC={log2fc[idx]:+.3f} FDR={p_adj[idx]:.2e}")

# ── 4.6 Summary table ─────────────────────────────────────────
print("\n" + "-" * 60)
print("[4.6] SUMMARY: Significant DEGs per subtype")
print(f"  {'Subtype':<15} {'Up':>6} {'Down':>6} {'Total':>7}")
print(f"  {'-'*15} {'-'*6} {'-'*6} {'-'*7}")
total_up = total_dn = 0
for st in subtypes:
    n_up = dge_results[st]['up'].sum()
    n_dn = dge_results[st]['down'].sum()
    n_tot = dge_results[st]['sig'].sum()
    total_up += n_up; total_dn += n_dn
    print(f"  {st:<15} {n_up:>6,} {n_dn:>6,} {n_tot:>7,}")

# ── 4.7 Save DEG results ──────────────────────────────────────
print("\n[4.7] Saving DEG results...")
all_dge = []
for st in subtypes:
    r = dge_results[st]
    dge_df = pd.DataFrame({
        'probe_id': feat_cols,
        'subtype':  st,
        'log2FC':   r['log2fc'],
        'p_adj':    r['p_adj'],
        'significant': r['sig'],
        'direction': np.where(r['up'], 'up', np.where(r['down'], 'down', 'ns'))
    })
    # Keep only significant genes
    dge_df = dge_df[dge_df['significant']].copy()
    all_dge.append(dge_df)

dge_all = pd.concat(all_dge, ignore_index=True)
dge_all.to_parquet(ARTIFACT_DIR / "dge_results.parquet", index=False)
print(f"  Saved {len(dge_all):,} significant gene-subtype pairs -> dge_results.parquet")

# ── 4.8 Top DEGs for heatmap (union of top 50 per subtype) ────
print("\n[4.8] Selecting top DEGs for heatmap visualization...")
top_deg_probes = set()
for st in subtypes:
    r = dge_results[st]
    sig_idx = np.where(r['sig'])[0]
    if len(sig_idx) > 0:
        ranked = sig_idx[np.argsort(np.abs(r['log2fc'][sig_idx]))[::-1][:50]]
        top_deg_probes.update([feat_cols[i] for i in ranked])
print(f"  Union of top 50 DEGs per subtype: {len(top_deg_probes)} unique probes")
joblib.dump(list(top_deg_probes), ARTIFACT_DIR / "top_deg_probes.pkl")
print(f"  Saved: top_deg_probes.pkl")

print("\n" + "=" * 60)
print("SECTION 4 COMPLETE [OK]")
print("=" * 60)

"""
SECTION 6 VERIFICATION: Co-expression Network Analysis
- Pearson correlation on top 500 variable probes (training-only)
- Adjacency matrix (|r| > 0.85 threshold)
- Module detection via hierarchical clustering
- Hub gene identification
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
from sklearn.preprocessing import LabelEncoder
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.stats import pearsonr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACT_DIR = PROJECT_ROOT / "data" / "artifacts"

print("=" * 60)
print("SECTION 6: CO-EXPRESSION NETWORK ANALYSIS")
print("=" * 60)

# ── Load data and create training split ──────────────────────
df = pd.read_parquet(PROCESSED_DATA_DIR / "breast_cancer_qn.parquet")
feat_cols = [c for c in df.columns if c != 'type']
X = df[feat_cols].values.astype(np.float32)
y = df['type'].values

le = LabelEncoder()
y_enc = le.fit_transform(y)

# Use only training data for network (anti-leakage)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, stratify=y_enc, random_state=42
)
print(f"\n[6.0] Training set only: {X_train.shape[0]} samples")
print(f"      (Network built on training data only — anti-leakage)")

# ── 6.1 Select top 500 most variable probes ───────────────────
print("\n[6.1] Selecting top 500 most variable probes (from training set)...")
gene_vars = np.var(X_train, axis=0)
top500_idx = np.argsort(gene_vars)[-500:]
top500_probes = [feat_cols[i] for i in top500_idx]
X_net = X_train[:, top500_idx]
print(f"  Variance range (top 500): {gene_vars[top500_idx].min():.4f} - {gene_vars[top500_idx].max():.4f}")

# ── 6.2 Pearson correlation matrix ────────────────────────────
print("\n[6.2] Computing Pearson correlation matrix (500 x 500)...")
corr_matrix = np.corrcoef(X_net.T)  # (500, 500)
np.fill_diagonal(corr_matrix, 0)     # remove self-correlations
print(f"  Correlation matrix shape: {corr_matrix.shape}")
print(f"  Min correlation: {corr_matrix.min():.4f}")
print(f"  Max correlation: {corr_matrix.max():.4f}")
print(f"  Mean |correlation|: {np.abs(corr_matrix).mean():.4f}")

# Distribution of correlation values
thresholds = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
print(f"\n  Edges at different |r| thresholds:")
n_possible = 500 * 499 // 2  # undirected pairs
for thr in thresholds:
    n_edges = (np.abs(corr_matrix) > thr).sum() // 2  # symmetric matrix
    pct = n_edges / n_possible * 100
    print(f"    |r| > {thr}: {n_edges:>6,} edges ({pct:.1f}% of possible)")

# ── 6.3 Build adjacency at threshold 0.85 ────────────────────
THR = 0.85
adj = (np.abs(corr_matrix) > THR).astype(int)
n_edges = adj.sum() // 2
node_degree = adj.sum(axis=1)
print(f"\n[6.3] Network at threshold |r| > {THR}:")
print(f"  Nodes (probes)    : 500")
print(f"  Edges             : {n_edges:,}")
print(f"  Isolated nodes    : {(node_degree == 0).sum()}")
print(f"  Max node degree   : {node_degree.max()}")
print(f"  Mean node degree  : {node_degree.mean():.2f}")
print(f"  Median node degree: {np.median(node_degree):.1f}")

# ── 6.4 Hub genes (top-degree nodes) ─────────────────────────
print(f"\n[6.4] Top 10 hub genes (highest connectivity):")
top_hub_idx = np.argsort(node_degree)[::-1][:10]
for rank, idx in enumerate(top_hub_idx, 1):
    print(f"  #{rank:<2}: {top500_probes[idx]:<30} degree={node_degree[idx]}")

# ── 6.5 Module detection via hierarchical clustering ──────────
print(f"\n[6.5] Module detection (hierarchical clustering on 1-|corr|)...")
dist_matrix = 1 - np.abs(corr_matrix)
np.fill_diagonal(dist_matrix, 0)
# Force exact symmetry to fix floating-point rounding issues
dist_matrix = (dist_matrix + dist_matrix.T) / 2.0
np.fill_diagonal(dist_matrix, 0)

from scipy.spatial.distance import squareform
condensed = squareform(dist_matrix)
Z_net = linkage(condensed, method='average')
module_labels = fcluster(Z_net, t=0.20, criterion='distance')  # 1-|r|<0.20 → |r|>0.80
n_modules = len(np.unique(module_labels))
print(f"  Number of modules detected: {n_modules}")

module_sizes = pd.Series(module_labels).value_counts().sort_values(ascending=False)
print(f"  Top 5 module sizes: {list(module_sizes.head(5).values)}")
print(f"  Singleton modules : {(module_sizes == 1).sum()}")
print(f"  Modules with >10 genes: {(module_sizes > 10).sum()}")

# ── 6.6 Save network artifacts ───────────────────────────────
print(f"\n[6.6] Saving network artifacts...")
# Save adjacency info
net_df = pd.DataFrame({
    'probe_id': top500_probes,
    'degree': node_degree,
    'module': module_labels,
})
net_df = net_df.sort_values('degree', ascending=False)
net_df.to_parquet(ARTIFACT_DIR / "coexpression_network.parquet", index=False)

# Save correlation matrix for heatmap
np.fill_diagonal(corr_matrix, 1.0)  # restore diagonal for visualization
np.save(ARTIFACT_DIR / "corr_matrix_500.npy", corr_matrix.astype(np.float32))
joblib.dump(top500_probes, ARTIFACT_DIR / "top500_probes.pkl")
joblib.dump(module_labels, ARTIFACT_DIR / "module_labels.pkl")
print(f"  Saved: coexpression_network.parquet, corr_matrix_500.npy, module_labels.pkl")

print("\n" + "=" * 60)
print("SECTION 6 COMPLETE [OK]")
print("=" * 60)

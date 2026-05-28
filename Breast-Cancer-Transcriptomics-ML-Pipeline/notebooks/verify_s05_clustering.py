"""
SECTION 5 VERIFICATION: Clustering Analysis
- Hierarchical clustering (Ward linkage)
- K-means (elbow method + silhouette)
- Cluster vs true label concordance
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import (
    silhouette_score, adjusted_rand_score,
    normalized_mutual_info_score, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACT_DIR = PROJECT_ROOT / "data" / "artifacts"

print("=" * 60)
print("SECTION 5: CLUSTERING ANALYSIS")
print("=" * 60)

# ── Load data ─────────────────────────────────────────────────
df = pd.read_parquet(PROCESSED_DATA_DIR / "breast_cancer_qn.parquet")
feat_cols = [c for c in df.columns if c != 'type']
X = df[feat_cols].values.astype(np.float32)
y = df['type'].values
subtypes = sorted(np.unique(y))

# Use PCA-50 coordinates for clustering (reduces noise, speeds up computation)
dr_df = pd.read_parquet(ARTIFACT_DIR / "dr_coordinates.parquet")
X_pca = np.column_stack([dr_df[f'PC{i}'].values for i in range(1, 4)])
# Use top 2000 most variable probes for hierarchical clustering
gene_vars = np.var(X, axis=0)
top2k_idx = np.argsort(gene_vars)[-2000:]
X_top2k = X[:, top2k_idx]

le = LabelEncoder()
y_enc = le.fit_transform(y)
n_true = len(subtypes)

print(f"\n[5.0] Data: {X.shape[0]} samples, {n_true} true subtypes")
print(f"      Using top 2000 variable probes for hierarchical clustering")
print(f"      Using PCA-50 coordinates for k-means / spectral")

# ── 5.1 Hierarchical clustering (Ward) ───────────────────────
print("\n[5.1] Hierarchical clustering (Ward linkage, Euclidean distance)...")
dist_condensed = pdist(X_top2k, metric='euclidean')
Z = linkage(dist_condensed, method='ward')
print(f"  Linkage matrix shape: {Z.shape}")

# Cut tree at k=5 (matching number of subtypes)
h_labels = fcluster(Z, t=n_true, criterion='maxclust') - 1  # 0-indexed
h_ari = adjusted_rand_score(y_enc, h_labels)
h_nmi = normalized_mutual_info_score(y_enc, h_labels)
h_sil = silhouette_score(X_top2k, h_labels, metric='euclidean', sample_size=137)
print(f"  Adjusted Rand Index (ARI)          : {h_ari:.4f}")
print(f"  Normalized Mutual Info (NMI)       : {h_nmi:.4f}")
print(f"  Silhouette Score                   : {h_sil:.4f}")
print(f"  (ARI=1.0 = perfect match with true labels, 0=random)")
joblib.dump(Z, ARTIFACT_DIR / "hierarchical_linkage.pkl")
joblib.dump(h_labels, ARTIFACT_DIR / "hierarchical_labels.pkl")

# ── 5.2 K-means elbow method ─────────────────────────────────
print("\n[5.2] K-means elbow method (k=2 to 10, PCA-50 space)...")
X_pca50 = np.column_stack([
    dr_df['PC1'].values, dr_df['PC2'].values, dr_df['PC3'].values
])
# Use full PCA-50 from file
pca_cols = [c for c in dr_df.columns if c.startswith('PC')]
X_pca_full = dr_df[pca_cols].values

wcss_list = []
sil_list  = []
k_range   = range(2, 11)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km_labels = km.fit_predict(X_pca_full)
    wcss_list.append(km.inertia_)
    sil = silhouette_score(X_pca_full, km_labels)
    sil_list.append(sil)
    print(f"  k={k}: WCSS={km.inertia_:,.1f}  Silhouette={sil:.4f}")

best_sil_k = k_range.start + np.argmax(sil_list)
print(f"\n  >> Best k by silhouette score: k={best_sil_k} (score={max(sil_list):.4f})")

# ── 5.3 K-means with k=5 (matching true labels) ─────────────
print(f"\n[5.3] K-means with k={n_true} (matching number of subtypes)...")
km5 = KMeans(n_clusters=n_true, random_state=42, n_init=20)
km5_labels = km5.fit_predict(X_pca_full)
km_ari = adjusted_rand_score(y_enc, km5_labels)
km_nmi = normalized_mutual_info_score(y_enc, km5_labels)
km_sil = silhouette_score(X_pca_full, km5_labels)
print(f"  Adjusted Rand Index (ARI): {km_ari:.4f}")
print(f"  Normalized Mutual Info   : {km_nmi:.4f}")
print(f"  Silhouette Score         : {km_sil:.4f}")
joblib.dump(km5_labels, ARTIFACT_DIR / "kmeans5_labels.pkl")

# ── 5.4 Cluster composition (which subtype lands in which cluster) ──
print(f"\n[5.4] K-means cluster composition (k=5):")
for cluster_id in range(n_true):
    mask = km5_labels == cluster_id
    subtypes_in_cluster = y[mask]
    vc = pd.Series(subtypes_in_cluster).value_counts()
    dominant = vc.index[0] if len(vc) > 0 else "empty"
    purity = vc.iloc[0] / mask.sum() if len(vc) > 0 else 0
    print(f"  Cluster {cluster_id}: n={mask.sum():>3}  dominant={dominant:<12} purity={purity:.1%}  | {dict(vc)}")

# ── 5.5 Method comparison ─────────────────────────────────────
print(f"\n[5.5] CLUSTERING METHOD COMPARISON:")
print(f"  {'Method':<25} {'ARI':>7} {'NMI':>7} {'Silhouette':>12}")
print(f"  {'-'*25} {'-'*7} {'-'*7} {'-'*12}")
print(f"  {'Hierarchical (Ward)':<25} {h_ari:>7.4f} {h_nmi:>7.4f} {h_sil:>12.4f}")
print(f"  {'K-means (k=5)':<25} {km_ari:>7.4f} {km_nmi:>7.4f} {km_sil:>12.4f}")

# Save cluster labels for visualization
cluster_df = pd.DataFrame({
    'subtype': y,
    'true_label': y_enc,
    'hierarchical_cluster': h_labels,
    'kmeans_cluster': km5_labels,
})
cluster_df.to_parquet(ARTIFACT_DIR / "cluster_labels.parquet", index=False)
print(f"\n[5.6] Saved: cluster_labels.parquet")

print("\n" + "=" * 60)
print("SECTION 5 COMPLETE [OK]")
print("=" * 60)

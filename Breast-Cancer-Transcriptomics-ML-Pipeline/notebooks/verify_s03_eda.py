"""
SECTION 3 VERIFICATION: EDA — PCA, t-SNE, UMAP
Run and capture ALL numeric outputs. Explanations written only from these real numbers.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACT_DIR = PROJECT_ROOT / "data" / "artifacts"

print("=" * 60)
print("SECTION 3: EDA -- PCA, t-SNE, UMAP")
print("=" * 60)

# ── Load quantile-normalized data ────────────────────────────
df = pd.read_parquet(PROCESSED_DATA_DIR / "breast_cancer_qn.parquet")
feat_cols = [c for c in df.columns if c != 'type']
X = df[feat_cols].values.astype(np.float32)
y = df['type'].values
print(f"\n[3.0] Loaded QN data: {X.shape}")

# ── 3.1 Top variable genes for visualization ─────────────────
print("\n[3.1] Selecting top 5000 most variable probes for visualization...")
gene_vars = np.var(X, axis=0)
top5k_idx = np.argsort(gene_vars)[-5000:]
X_top5k = X[:, top5k_idx]
print(f"  Variance range of selected probes: {gene_vars[top5k_idx].min():.4f} - {gene_vars[top5k_idx].max():.4f}")
print(f"  Variance of discarded probes (median): {np.median(gene_vars[np.argsort(gene_vars)[:-5000]]):.4f}")

# ── 3.2 RobustScaler + PCA ───────────────────────────────────
print("\n[3.2] RobustScaler + PCA (50 components)...")
scaler_robust = RobustScaler()
X_scaled = scaler_robust.fit_transform(X_top5k)

pca = PCA(n_components=50, random_state=42)
X_pca50 = pca.fit_transform(X_scaled)
cumvar = np.cumsum(pca.explained_variance_ratio_)

print(f"  PC1 explained variance : {pca.explained_variance_ratio_[0]*100:.2f}%")
print(f"  PC2 explained variance : {pca.explained_variance_ratio_[1]*100:.2f}%")
print(f"  PC3 explained variance : {pca.explained_variance_ratio_[2]*100:.2f}%")
print(f"  Top 5 PCs combined     : {cumvar[4]*100:.2f}%")
print(f"  Top 10 PCs combined    : {cumvar[9]*100:.2f}%")
print(f"  Top 50 PCs combined    : {cumvar[49]*100:.2f}%")

# PCA 2D for saving (first 2 components)
X_pca2 = X_pca50[:, :2]
pca2_df = pd.DataFrame({'PC1': X_pca2[:, 0], 'PC2': X_pca2[:, 1], 'subtype': y})
pca2_df.to_parquet(PROCESSED_DATA_DIR / "pca_2d.parquet", index=False)
print(f"  PCA 2D coordinates saved.")

# ── 3.3 Per-subtype PCA cluster analysis ────────────────────
print("\n[3.3] PCA cluster separation per subtype (PC1 vs PC2 centroids):")
for st in sorted(np.unique(y)):
    mask = y == st
    centroid = X_pca2[mask].mean(axis=0)
    spread = X_pca2[mask].std(axis=0).mean()
    print(f"  {st:<12}: centroid=({centroid[0]:+.2f}, {centroid[1]:+.2f}), spread={spread:.2f}")

# ── 3.4 t-SNE ────────────────────────────────────────────────
print("\n[3.4] Running t-SNE (perplexity=30, on PCA-50)...")
tsne = TSNE(n_components=2, perplexity=30, random_state=42,
            n_iter=1000, learning_rate='auto', init='pca')
X_tsne = tsne.fit_transform(X_pca50)
print(f"  t-SNE shape: {X_tsne.shape}")
print(f"  KL divergence (final): {tsne.kl_divergence_:.4f}")
print(f"  Value range X: [{X_tsne[:,0].min():.2f}, {X_tsne[:,0].max():.2f}]")
print(f"  Value range Y: [{X_tsne[:,1].min():.2f}, {X_tsne[:,1].max():.2f}]")

# Per-subtype t-SNE centroids
print("\n  t-SNE subtype centroids:")
for st in sorted(np.unique(y)):
    mask = y == st
    centroid = X_tsne[mask].mean(axis=0)
    spread = X_tsne[mask].std(axis=0).mean()
    print(f"  {st:<12}: centroid=({centroid[0]:+.2f}, {centroid[1]:+.2f}), spread={spread:.2f}")

# ── 3.5 UMAP ─────────────────────────────────────────────────
print("\n[3.5] Running UMAP (n_neighbors=15, min_dist=0.1, on PCA-50)...")
import umap
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2,
                    random_state=42, verbose=False)
X_umap = reducer.fit_transform(X_pca50)
print(f"  UMAP shape: {X_umap.shape}")
print(f"  Value range X: [{X_umap[:,0].min():.2f}, {X_umap[:,0].max():.2f}]")
print(f"  Value range Y: [{X_umap[:,1].min():.2f}, {X_umap[:,1].max():.2f}]")

print("\n  UMAP subtype centroids:")
for st in sorted(np.unique(y)):
    mask = y == st
    centroid = X_umap[mask].mean(axis=0)
    spread = X_umap[mask].std(axis=0).mean()
    print(f"  {st:<12}: centroid=({centroid[0]:+.2f}, {centroid[1]:+.2f}), spread={spread:.2f}")

# ── 3.6 Save DR coordinates ──────────────────────────────────
print("\n[3.6] Saving dimensionality reduction coordinates...")
dr_df = pd.DataFrame({
    'subtype': y,
    'PC1': X_pca50[:, 0], 'PC2': X_pca50[:, 1], 'PC3': X_pca50[:, 2],
    'TSNE1': X_tsne[:, 0], 'TSNE2': X_tsne[:, 1],
    'UMAP1': X_umap[:, 0], 'UMAP2': X_umap[:, 1],
})
dr_df.to_parquet(ARTIFACT_DIR / "dr_coordinates.parquet", index=False)
joblib.dump(pca, ARTIFACT_DIR / "pca_50.pkl")
print(f"  Saved: dr_coordinates.parquet, pca_50.pkl")

# ── 3.7 Top variable genes stats ────────────────────────────
print("\n[3.7] Top 20 highest-variance probes:")
top20_idx = np.argsort(gene_vars)[-20:][::-1]
top20_probes = [feat_cols[i] for i in top20_idx]
top20_vars = gene_vars[top20_idx]
for probe, var in zip(top20_probes[:10], top20_vars[:10]):
    print(f"  {probe:<30} var={var:.4f}")
print(f"  ... (top 10 of 20 shown)")

print("\n" + "=" * 60)
print("SECTION 3 COMPLETE [OK]")
print("=" * 60)

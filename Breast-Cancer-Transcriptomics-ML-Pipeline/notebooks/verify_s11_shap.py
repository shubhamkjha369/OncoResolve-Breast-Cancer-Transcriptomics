"""
SECTION 11 VERIFICATION: SHAP Interpretability
- TreeExplainer on best tuned RF model
- Evaluated on held-out test set (not training data)
- Top probes annotated via MyGene API
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

import shap

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = PROJECT_ROOT / "data" / "artifacts"

print("=" * 60)
print("SECTION 11: SHAP INTERPRETABILITY")
print("=" * 60)

# ── Load ──────────────────────────────────────────────────────
tuned_rf     = joblib.load(ARTIFACT_DIR / "tuned_rf.pkl")
X_train_c    = joblib.load(ARTIFACT_DIR / "X_train_consensus.pkl")
X_test_c     = joblib.load(ARTIFACT_DIR / "X_test_consensus.pkl")
y_test       = joblib.load(ARTIFACT_DIR / "y_test.pkl")
le           = joblib.load(ARTIFACT_DIR / "label_encoder.pkl")
consensus_genes = joblib.load(ARTIFACT_DIR / "top_consensus_genes.pkl")
class_names  = list(le.classes_)

print(f"\n[11.0] Model: tuned Random Forest")
print(f"       Features: {X_test_c.shape[1]} consensus genes")
print(f"       Test samples for SHAP: {X_test_c.shape[0]}")

# ── 11.1 SHAP TreeExplainer ───────────────────────────────────
print(f"\n[11.1] Computing SHAP values (TreeExplainer)...")
explainer = shap.TreeExplainer(tuned_rf)
shap_values = explainer.shap_values(X_test_c)

# shap_values is a list of arrays [n_classes] each (n_test, n_features)
print(f"  SHAP values type   : {type(shap_values)}")
if isinstance(shap_values, list):
    print(f"  Number of classes  : {len(shap_values)}")
    print(f"  Shape per class    : {shap_values[0].shape}")
    # Mean |SHAP| across all classes and samples
    shap_arr = np.array(shap_values)  # (n_classes, n_test, n_features)
    mean_abs_shap = np.abs(shap_arr).mean(axis=(0, 1))  # (n_features,)
else:
    print(f"  SHAP array shape   : {shap_values.shape}")
    shap_arr = shap_values
    mean_abs_shap = np.abs(shap_arr).mean(axis=(0, 2))  # (n_features,)

print(f"  Mean |SHAP| computed over all {X_test_c.shape[0]} test samples")
print(f"  Max mean |SHAP|  : {mean_abs_shap.max():.6f}")
print(f"  Min mean |SHAP|  : {mean_abs_shap.min():.6f}")

# ── 11.2 Top features by mean |SHAP| ─────────────────────────
print(f"\n[11.2] Top 20 features by mean |SHAP| value:")
top20_idx   = np.argsort(mean_abs_shap)[::-1][:20]
top20_genes = [consensus_genes[i] for i in top20_idx]
top20_shap  = mean_abs_shap[top20_idx]
for rank, (gene, sv) in enumerate(zip(top20_genes, top20_shap), 1):
    print(f"  #{rank:<2}: {gene:<30}  mean|SHAP|={sv:.6f}")

# ── 11.3 Per-class top features ──────────────────────────────
print(f"\n[11.3] Top 5 features per class:")
for cls_idx, cls_name in enumerate(class_names):
    if isinstance(shap_values, list):
        cls_shap = np.abs(shap_values[cls_idx]).mean(axis=0)
    else:
        cls_shap = np.abs(shap_values[:, :, cls_idx]).mean(axis=0)
    top5_cls = np.argsort(cls_shap)[::-1][:5]
    genes_cls = [consensus_genes[i] for i in top5_cls]
    shap_cls  = cls_shap[top5_cls]
    print(f"\n  Class: {cls_name}")
    for gene, sv in zip(genes_cls, shap_cls):
        print(f"    {gene:<30}  mean|SHAP|={sv:.6f}")

# ── 11.4 MyGene annotation ────────────────────────────────────
print(f"\n[11.4] Annotating top 10 probes via MyGene API...")
top10_probes = top20_genes[:10]
try:
    import mygene
    mg = mygene.MyGeneInfo()
    # Clean probe IDs to Affymetrix format (remove _at suffix for query)
    results = mg.querymany(
        top10_probes,
        scopes='reporter',
        fields='symbol,name,summary',
        species='human',
        verbose=False
    )
    annotations = {}
    for r in results:
        query = r.get('query', '')
        symbol = r.get('symbol', 'N/A')
        name   = r.get('name', 'N/A')
        annotations[query] = {'symbol': symbol, 'name': name}

    print(f"  Annotated {len(annotations)} probes")
    print(f"\n  Probe -> Gene Symbol -> Gene Name:")
    for probe in top10_probes:
        ann = annotations.get(probe, {'symbol': 'N/A', 'name': 'N/A'})
        print(f"    {probe:<30} -> {ann['symbol']:<12} -> {ann['name'][:60]}")
    joblib.dump(annotations, ARTIFACT_DIR / "shap_annotations.pkl")
except Exception as e:
    print(f"  MyGene API unavailable: {e}")
    print(f"  Top probes (unannotated): {top10_probes}")
    annotations = {}

# ── 11.5 Save SHAP artifacts ──────────────────────────────────
print(f"\n[11.5] Saving SHAP artifacts...")
shap_df = pd.DataFrame({
    'probe_id': consensus_genes,
    'mean_abs_shap': mean_abs_shap,
    'shap_rank': pd.Series(mean_abs_shap).rank(ascending=False).values.astype(int),
})
shap_df = shap_df.sort_values('shap_rank')
shap_df.to_parquet(ARTIFACT_DIR / "shap_importance.parquet", index=False)
np.save(ARTIFACT_DIR / "shap_values.npy",
        np.array(shap_values) if isinstance(shap_values, list) else shap_values)
print(f"  Saved: shap_importance.parquet, shap_values.npy")

print("\n" + "=" * 60)
print("SECTION 11 COMPLETE [OK]")
print("=" * 60)

"""
SECTION 8 VERIFICATION: ML Model Benchmarking
Depends on: X_train_consensus, X_test_consensus, y_train, y_test from Section 7
Tests: LR, SVM, RF, XGBoost, LightGBM on consensus features
Metrics: Accuracy, Weighted F1, per-class Precision/Recall
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import copy

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = PROJECT_ROOT / "data" / "artifacts"

print("=" * 60)
print("SECTION 8: ML MODEL BENCHMARKING")
print("=" * 60)

# ── Load artifacts ────────────────────────────────────────────
X_train_c = joblib.load(ARTIFACT_DIR / "X_train_consensus.pkl")
X_test_c  = joblib.load(ARTIFACT_DIR / "X_test_consensus.pkl")
X_train_s = joblib.load(ARTIFACT_DIR / "X_train_scaled.pkl")
X_test_s  = joblib.load(ARTIFACT_DIR / "X_test_scaled.pkl")
y_train   = joblib.load(ARTIFACT_DIR / "y_train.pkl")
y_test    = joblib.load(ARTIFACT_DIR / "y_test.pkl")
le        = joblib.load(ARTIFACT_DIR / "label_encoder.pkl")
class_names = list(le.classes_)

print(f"\n[8.0] Consensus features: {X_train_c.shape[1]}")
print(f"      Train: {X_train_c.shape[0]}  Test: {X_test_c.shape[0]}")
print(f"      Classes: {class_names}")

# ── PCA-50 feature space (from full scaled data) ─────────────
print(f"\n[8.1] Creating PCA-50 feature space...")
pca = PCA(n_components=50, random_state=42)
X_train_pca = pca.fit_transform(X_train_s)
X_test_pca  = pca.transform(X_test_s)
print(f"  PCA-50 variance captured: {pca.explained_variance_ratio_.sum()*100:.2f}%")

# ── Define models ─────────────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(max_iter=5000, C=1.0, random_state=42, n_jobs=-1),
    "SVM (RBF)":           SVC(kernel='rbf', probability=True, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
    "XGBoost":             XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                                          objective='multi:softprob', eval_metric='mlogloss',
                                          random_state=42, verbosity=0),
    "LightGBM":            LGBMClassifier(n_estimators=300, learning_rate=0.05,
                                           random_state=42, verbose=-1),
}

feature_spaces = {
    "Consensus": (X_train_c, X_test_c),
    "PCA-50":    (X_train_pca, X_test_pca),
}

results = []
best_acc = 0
best_model_obj = None
best_model_name = ""
best_space_name = ""
best_X_test = None  # store the correct test slice for best model

print(f"\n[8.2] Benchmark loop...")
for space_name, (X_tr, X_te) in feature_spaces.items():
    print(f"\n  --- Feature space: {space_name} ({X_tr.shape[1]} features) ---")
    for model_name, model_template in models.items():
        # Clone fresh model per space to avoid weight contamination
        model = clone(model_template)
        model.fit(X_tr, y_train)
        y_pred = model.predict(X_te)
        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred, average='weighted')
        results.append({
            'feature_space': space_name,
            'model': model_name,
            'accuracy': acc,
            'weighted_f1': f1,
        })
        print(f"  {model_name:<22} Acc={acc:.4f}  F1={f1:.4f}")
        if acc > best_acc or (acc == best_acc and model_name == "Random Forest"):
            best_acc = acc
            best_model_obj = copy.deepcopy(model)  # deepcopy preserves trained state
            best_model_name = model_name
            best_space_name = space_name
            best_X_test = X_te  # store corresponding test set

# ── Results summary ───────────────────────────────────────────
print(f"\n[8.3] FULL BENCHMARK RESULTS:")
df_results = pd.DataFrame(results).sort_values('accuracy', ascending=False)
print(df_results.to_string(index=False))

print(f"\n  >> Best model: {best_model_name} on {best_space_name}")
print(f"     Accuracy = {best_acc:.4f}")

# ── Detailed report for best model ───────────────────────
print(f"\n[8.4] Classification report for best model ({best_model_name} on {best_space_name}):")
y_pred_best = best_model_obj.predict(best_X_test)
print(classification_report(y_test, y_pred_best, target_names=class_names))

# ── Confusion matrix ──────────────────────────────────────────
print(f"\n[8.5] Confusion matrix (rows=true, cols=predicted):")
cm = confusion_matrix(y_test, y_pred_best)
cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
print(cm_df.to_string())

# ── Save ──────────────────────────────────────────────────────
print(f"\n[8.6] Saving results and best model...")
df_results.to_parquet(ARTIFACT_DIR / "benchmark_results.parquet", index=False)
joblib.dump(best_model_obj, ARTIFACT_DIR / "best_model.pkl")
joblib.dump({'name': best_model_name, 'space': best_space_name,
             'accuracy': best_acc}, ARTIFACT_DIR / "best_model_info.pkl")
print(f"  Saved: benchmark_results.parquet, best_model.pkl")

print("\n" + "=" * 60)
print("SECTION 8 COMPLETE [OK]")
print("=" * 60)

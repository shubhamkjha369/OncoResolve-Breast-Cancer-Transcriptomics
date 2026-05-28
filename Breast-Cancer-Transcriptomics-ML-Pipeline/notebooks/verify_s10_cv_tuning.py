"""
SECTION 10 VERIFICATION: Cross-Validation & Hyperparameter Tuning
- Stratified 5-fold CV on Random Forest (consensus features)
- GridSearchCV: n_estimators x max_features x max_depth
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    StratifiedKFold, cross_validate, GridSearchCV
)
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = PROJECT_ROOT / "data" / "artifacts"

print("=" * 60)
print("SECTION 10: CROSS-VALIDATION & HYPERPARAMETER TUNING")
print("=" * 60)

# ── Load artifacts ────────────────────────────────────────────
X_train_c = joblib.load(ARTIFACT_DIR / "X_train_consensus.pkl")
X_test_c  = joblib.load(ARTIFACT_DIR / "X_test_consensus.pkl")
y_train   = joblib.load(ARTIFACT_DIR / "y_train.pkl")
y_test    = joblib.load(ARTIFACT_DIR / "y_test.pkl")
le        = joblib.load(ARTIFACT_DIR / "label_encoder.pkl")
class_names = list(le.classes_)

print(f"\n[10.0] Data: {X_train_c.shape[1]} consensus features")

# ── 10.1 Stratified 5-fold CV on baseline RF ─────────────────
print("\n[10.1] Stratified 5-fold CV on Random Forest (baseline)...")
rf_base = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = cross_validate(
    rf_base, X_train_c, y_train, cv=cv,
    scoring=['accuracy', 'f1_weighted'],
    return_train_score=True, n_jobs=-1
)

print(f"  CV Accuracy  : {cv_results['test_accuracy'].mean():.4f} +/- {cv_results['test_accuracy'].std():.4f}")
print(f"  CV F1 weighted: {cv_results['test_f1_weighted'].mean():.4f} +/- {cv_results['test_f1_weighted'].std():.4f}")
print(f"  Per-fold accuracy: {[f'{s:.4f}' for s in cv_results['test_accuracy']]}")
print(f"  Train accuracy  : {cv_results['train_accuracy'].mean():.4f} +/- {cv_results['train_accuracy'].std():.4f}")
train_test_gap = cv_results['train_accuracy'].mean() - cv_results['test_accuracy'].mean()
print(f"  Train-Val gap (overfitting proxy): {train_test_gap:.4f}")
print(f"  >> {'Moderate overfitting' if train_test_gap > 0.1 else 'Well-generalizing model'}")

joblib.dump(cv_results, ARTIFACT_DIR / "cv_results_baseline_rf.pkl")

# ── 10.2 GridSearchCV ─────────────────────────────────────────
print(f"\n[10.2] GridSearchCV on Random Forest (consensus features)...")
param_grid = {
    'n_estimators': [100, 300],
    'max_features': ['sqrt', 'log2'],
    'max_depth':    [None, 20, 10],
}
n_combinations = 2 * 2 * 3
print(f"  Grid: {param_grid}")
print(f"  Total combinations: {n_combinations} x 5 folds = {n_combinations*5} fits")

rf_grid = RandomForestClassifier(random_state=42, n_jobs=-1)
gs = GridSearchCV(
    rf_grid, param_grid,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='f1_weighted', n_jobs=-1, verbose=0
)
gs.fit(X_train_c, y_train)

print(f"\n  Best parameters: {gs.best_params_}")
print(f"  Best CV F1     : {gs.best_score_:.4f}")

# Print full grid results
gs_df = pd.DataFrame(gs.cv_results_)
gs_df = gs_df[[
    'param_n_estimators', 'param_max_features', 'param_max_depth',
    'mean_test_score', 'std_test_score'
]].sort_values('mean_test_score', ascending=False)
print(f"\n  Grid search results (sorted by F1):")
print(gs_df.to_string(index=False))

# ── 10.3 Final evaluation on test set ────────────────────────
print(f"\n[10.3] Evaluating tuned model on held-out test set...")
best_rf = gs.best_estimator_
y_pred_tuned = best_rf.predict(X_test_c)
tuned_acc = accuracy_score(y_test, y_pred_tuned)
tuned_f1  = f1_score(y_test, y_pred_tuned, average='weighted')
print(f"  Tuned RF Test Accuracy: {tuned_acc:.4f}")
print(f"  Tuned RF Test F1      : {tuned_f1:.4f}")
print(f"\n  Classification Report:")
print(classification_report(y_test, y_pred_tuned, target_names=class_names, zero_division=0))

print(f"\n  Confusion matrix (rows=true, cols=predicted):")
cm = confusion_matrix(y_test, y_pred_tuned)
cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
print(cm_df.to_string())

# ── Save ──────────────────────────────────────────────────────
joblib.dump(best_rf, ARTIFACT_DIR / "tuned_rf.pkl")
joblib.dump(gs,      ARTIFACT_DIR / "gridsearch_results.pkl")
gs_df.to_parquet(ARTIFACT_DIR / "gridsearch_df.parquet", index=False)
print(f"\n[10.4] Saved: tuned_rf.pkl, gridsearch_results.pkl, gridsearch_df.parquet")

print("\n" + "=" * 60)
print("SECTION 10 COMPLETE [OK]")
print("=" * 60)

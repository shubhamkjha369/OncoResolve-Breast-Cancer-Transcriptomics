import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

ARTIFACT_DIR = Path("data/artifacts")

# Load assets
best_rf = joblib.load(ARTIFACT_DIR / "tuned_rf.pkl")
best_lr = joblib.load(ARTIFACT_DIR / "tuned_lr.pkl")
X_train_c = joblib.load(ARTIFACT_DIR / "X_train_consensus.pkl")
y_train = joblib.load(ARTIFACT_DIR / "y_train.pkl")

print(f"X_train_c shape: {X_train_c.shape}")
print(f"y_train shape: {y_train.shape}")

cv_repeated = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=10,
    random_state=42
)

# Evaluate Tuned Random Forest
print("Evaluating Tuned RF...")
rf_scores = cross_val_score(
    best_rf,
    X_train_c,
    y_train,
    scoring='f1_weighted',
    cv=cv_repeated,
    n_jobs=-1
)

# Evaluate Tuned Logistic Regression
print("Evaluating Tuned LR...")
lr_scores = cross_val_score(
    best_lr,
    X_train_c,
    y_train,
    scoring='f1_weighted',
    cv=cv_repeated,
    n_jobs=-1
)

print("Tuned Random Forest      — Mean F1: {:.4f} | Std F1: {:.4f}".format(rf_scores.mean(), rf_scores.std()))
print("Tuned Logistic Regress.  — Mean F1: {:.4f} | Std F1: {:.4f}".format(lr_scores.mean(), lr_scores.std()))

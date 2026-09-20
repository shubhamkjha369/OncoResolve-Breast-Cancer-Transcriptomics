import os
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

class OncoClassifier:
    """
    OncoResolve diagnostic classifier.
    Exposes APIs to load pre-trained models (LightGBM or Linear SVM)
    or train new models for PAM50 breast cancer subtyping.
    Includes inference-time HER2 decision threshold gating to suppress false positives.
    """
    def __init__(sef, model_type="svm", model_path=None, label_encoder_path=None, her2_conflidence_threshold=0.40):
        self.model_type = model_type.lower()
        self.model_ = None
        self.label_encoder_ = None
        self.her2_confidence_threshold = her2_confidence_threshold
        
        # Determine default paths if not specified
        if model_path is None or label_encoder_path is None:
            package_root = Path(__file__).resolve().parent.parent
            artifacts_dir = package_root / "data" / "artifacts"
            results_dir = package_root / "results"
            
            if model_path is None:
                if self.model_type in ["lgbm", "lightgbm"]:
                    candidate_paths = [
                        artifacts_dir / "lgbm_model.pkl",
                        artifacts_dir / "finalized_pam50_LGBM_model.pkl",
                        artifacts_dir / "final_lgbm_pipeline.pkl",
                        results_dir / "lgbm_model.pkl"
                    ]
                else:
                    candidate_paths = [
                        artifacts_dir / "linear_svm_model.pkl",
                        artifacts_dir / "finalized_pam50_linear_svm_model.pkl",
                        artifacts_dir / "final_svm_pipeline.pkl",
                        results_dir / "linear_svm_model.pkl"
                    ]
                
                for cand in candidate_paths:
                    if cand.exists():
                        model_path = cand
                        break

            if label_encoder_path is None:
                candidate_encoders = [
                    artifacts_dir / "label_encoder_cohort.pkl",
                    results_dir / "label_encoder_cohort.pkl"
                ]
                for cand in candidate_encoders:
                    if cand.exists():
                        label_encoder_path = cand
                        break

        # Attempt to load pre-trained if the files exist
        if model_path is not None and os.path.exists(model_path):
            self.model_ = joblib.load(model_path)
        if label_encoder_path is not None and os.path.exists(label_encoder_path):
            self.label_encoder_ = joblib.load(label_encoder_path)

    def load_pretrained(self, model_path, label_encoder_path=None):
        """
        Loads the pre-trained classifier and label encoder.
        """
        self.model_ = joblib.load(model_path)
        if label_encoder_path and os.path.exists(label_encoder_path):
            self.label_encoder_ = joblib.load(label_encoder_path)
        return self

    def fit(self, X, y, model_params=None):
        """
        Trains a new classification model from scratch.

        Parameters:
        -----------
        X : np.ndarray or pd.DataFrame
            Expression matrix.
        y : np.ndarray or pd.Series
            Subtype labels (raw names or encoded).
        model_params : dict, optional
            Hyperparameters for the classifier.
        """
        from sklearn.preprocessing import LabelEncoder
        from sklearn.svm import SVC

        y_vals = np.asarray(y)
        first_val = y_vals[0] if len(y_vals) > 0 else None
        if isinstance(first_val, (str, np.str_)):
            self.label_encoder_ = LabelEncoder()
            y_encoded = self.label_encoder_.fit_transform(y_vals)
        else:
            y_encoded = y_vals

        params = model_params or {}
        if self.model_type in ["lgbm", "lightgbm"]:
            from lightgbm import LGBMClassifier
            self.model_ = LGBMClassifier(class_weight="balanced", random_state=42, verbosity=-1, n_jobs=1, **params)
        elif self.model_type in ["svm", "linear_svm"]:
            self.model_ = SVC(kernel="linear", C=0.01, class_weight="balanced", probability=True, random_state=42, **params)
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}. Select 'lgbm', 'linear_svm', 'svm', or 'logistic_regression'.")

        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        self.model_.fit(X_arr, y_encoded)
        return self

    def predict(self, X, apply_her2_gate=True):
        """
        Predicts PAM50 intrinsic subtypes for input samples.
        Optionally applies HER2 decision threshold gating to suppress false positives.
        """
        if self.model_ is None:
            raise RuntimeError("Model is not loaded or trained. Call fit() or load_pretrained() first.")

        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)

        raw_preds = self.model_.predict(X_arr)
        
        # Extract probabilities if threshold gating is requested
        if apply_her2_gate and self.her2_confidence_threshold is not None:
            try:
                probs = self.predict_proba(X_arr)
                if isinstance(probs, pd.DataFrame):
                    probs_arr = probs.values
                    classes_list = list(probs.columns)
                else:
                    probs_arr = probs
                    classes_list = list(self.label_encoder_.classes_) if self.label_encoder_ is not None else []
                
                if "her2" in classes_list:
                    her2_idx = classes_list.index("her2")
                    adjusted_preds = []
                    for idx, p in enumerate(probs_arr):
                        curr_pred = raw_preds[idx]
                        curr_pred_idx = curr_pred if np.issubdtype(type(curr_pred), np.integer) else (classes_list.index(curr_pred) if curr_pred in classes_list else None)
                        
                        if p[her2_idx] < self.her2_confidence_threshold and curr_pred_idx == her2_idx:
                            p_copy = p.copy()
                            p_copy[her2_idx] = -1.0
                            fallback_idx = np.argmax(p_copy)
                            adjusted_preds.append(fallback_idx if self.label_encoder_ is not None and np.issubdtype(type(raw_preds[0]), np.integer) else classes_list[fallback_idx])
                        else:
                            adjusted_preds.append(curr_pred)
                    raw_preds = np.array(adjusted_preds)
            except Exception:
                pass  # Fall back to raw predictions if proba not available

        if self.label_encoder_ is not None and np.issubdtype(np.array(raw_preds).dtype, np.integer):
            return self.label_encoder_.inverse_transform(raw_preds)
        return raw_preds

    def predict_proba(self, X):
        """
        Predicts classification probabilities for each PAM50 subtype.
        """
        if self.model_ is None:
            raise RuntimeError("Model is not loaded or trained. Call fit() or load_pretrained() first.")
            
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
            
        if hasattr(self.model_, "predict_proba"):
            probs = self.model_.predict_proba(X_arr)
        elif hasattr(self.model_, "decision_function"):
            dec = self.model_.decision_function(X_arr)
            if dec.ndim == 1:
                dec = np.vstack([-dec, dec]).T
            exp_dec = np.exp(dec - np.max(dec, axis=1, keepdims=True))
            probs = exp_dec / np.sum(exp_dec, axis=1, keepdims=True)
        else:
            preds = self.model_.predict(X_arr)
            num_classes = len(self.label_encoder_.classes_) if self.label_encoder_ else len(np.unique(preds))
            probs = np.zeros((len(preds), num_classes))
            probs[np.arange(len(preds)), preds] = 1.0
        
        # Wrap as DataFrame if class names are known
        if self.label_encoder_ is not None:
            classes = list(self.label_encoder_.classes_)
            idx = X.index if isinstance(X, pd.DataFrame) else None
            return pd.DataFrame(probs, index=idx, columns=classes)
        return probs

import os
import copy
import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.model_selection import train_test_split


class DGEInFoldFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, top_k=500):
        self.top_k = top_k

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        n_samples, self.n_features_in_ = X.shape
        classes = np.unique(y)
        n_classes = len(classes)

        if n_classes < 2 or self.top_k >= self.n_features_in_:
            self.selected_indices_ = np.arange(self.n_features_in_)
            return self

        f_scores = np.zeros(self.n_features_in_)
        grand_mean = np.mean(X, axis=0)

        for cls in classes:
            mask = (y == cls)
            n_cls = np.sum(mask)
            if n_cls > 1:
                cls_mean = np.mean(X[mask], axis=0)
                f_scores += n_cls * ((cls_mean - grand_mean) ** 2)

        top_indices = np.argsort(f_scores)[::-1][:self.top_k]
        self.selected_indices_ = np.sort(top_indices)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float32)
        return X[:, self.selected_indices_]


class PyTorchMLP(nn.Module):
    def __init__(self, in_features, hidden_sizes, n_classes, dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = in_features
        for h_dim in hidden_sizes:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, n_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class PyTorchMLPClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        hidden_layer_sizes=(128, 64, 32),
        lr=1e-3,
        weight_decay=1e-3,
        dropout=0.30,
        batch_size=32,
        max_epochs=30,
        early_stopping=True,
        patience=5,
        min_delta=1e-4,
        validation_fraction=0.15,
        scheduler="reduce_on_plateau",
        scheduler_factor=0.5,
        scheduler_patience=5,
        min_lr=1e-6,
        device="cpu",
        random_state=42
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.early_stopping = early_stopping
        self.patience = patience
        self.min_delta = min_delta
        self.validation_fraction = validation_fraction
        self.scheduler = scheduler
        self.scheduler_factor = scheduler_factor
        self.scheduler_patience = scheduler_patience
        self.min_lr = min_lr
        self.device = device
        self.random_state = random_state
        self.model_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)

        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.n_features_in_ = X.shape[1]

        y_mapped = np.searchsorted(self.classes_, y)

        if self.validation_fraction > 0 and X.shape[0] > 20:
            X_tr, X_val, y_tr, y_val = train_test_split(
                X, y_mapped, test_size=self.validation_fraction,
                stratify=y_mapped, random_state=self.random_state
            )
        else:
            X_tr, y_tr = X, y_mapped
            X_val, y_val = None, None

        torch.manual_seed(self.random_state)
        self.model_ = PyTorchMLP(
            self.n_features_in_, self.hidden_layer_sizes,
            self.n_classes_, self.dropout
        ).to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=self.scheduler_factor,
            patience=self.scheduler_patience, min_lr=self.min_lr
        ) if self.scheduler == "reduce_on_plateau" else None

        train_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr).long())
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        best_loss = float('inf')
        patience_counter = 0
        best_weights = None

        for epoch in range(self.max_epochs):
            self.model_.train()
            for bx, by in train_loader:
                bx, by = bx.to(self.device), by.to(self.device)
                optimizer.zero_grad()
                out = self.model_(bx)
                loss = criterion(out, by)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model_.parameters(), 1.0)
                optimizer.step()

            if X_val is not None:
                self.model_.eval()
                with torch.no_grad():
                    vx = torch.from_numpy(X_val).to(self.device)
                    vy = torch.from_numpy(y_val).long().to(self.device)
                    val_out = self.model_(vx)
                    val_loss = criterion(val_out, vy).item()

                if lr_scheduler:
                    lr_scheduler.step(val_loss)

                if val_loss < best_loss - self.min_delta:
                    best_loss = val_loss
                    patience_counter = 0
                    best_weights = copy.deepcopy(self.model_.state_dict())
                else:
                    patience_counter += 1

                if self.early_stopping and patience_counter >= self.patience:
                    break

        if best_weights is not None:
            self.model_.load_state_dict(best_weights)
        self.model_.eval()
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        self.model_.eval()
        with torch.no_grad():
            tx = torch.from_numpy(X).to(self.device)
            logits = self.model_(tx)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        indices = np.argmax(probs, axis=1)
        return self.classes_[indices]

    def compute_gradient_attributions(self, X):
        """
        Computes Gradient x Input feature attributions for PyTorch MLP for each class.
        
        Returns
        -------
        np.ndarray of shape (n_samples, n_features, n_classes)
        """
        X_arr = np.asarray(X, dtype=np.float32)
        if self.model_ is None:
            raise RuntimeError("PyTorchMLPClassifier model is not fitted.")
        
        self.model_.eval()
        n_classes = self.n_classes_
        attributions = []
        for c_idx in range(n_classes):
            X_t = torch.tensor(X_arr, dtype=torch.float32, device=self.device, requires_grad=True)
            out = self.model_(X_t)[:, c_idx]
            out.backward(torch.ones_like(out))
            attr = (X_t.grad * X_t).detach().cpu().numpy()
            attributions.append(attr)
        
        return np.stack(attributions, axis=2)


class OncoClassifier:
    """
    OncoResolve diagnostic classifier.
    Exposes APIs to load pre-trained models or train new models for PAM50 breast cancer subtyping.
    Includes inference-time HER2 decision threshold gating to suppress false positives.
    Strictly restricted to the Tri-Architecture: Logistic Regression, Random Forest, and PyTorch MLP.
    """
    def __init__(self, model_type="logistic_regression", model_path=None, label_encoder_path=None, her2_confidence_threshold=0.40):
        self.model_type = model_type.lower()
        if self.model_type not in ["logistic_regression", "lr", "random_forest", "rf", "mlp", "pytorch_mlp"]:
            raise ValueError(
                f"Unsupported model_type: '{model_type}'. "
                "OncoResolve tri-architecture strictly permits 'logistic_regression', 'random_forest', or 'mlp'."
            )
        self.model_ = None
        self.label_encoder_ = None
        self.her2_confidence_threshold = her2_confidence_threshold
        
        # Determine default paths if not specified
        if model_path is None or label_encoder_path is None:
            package_root = Path(__file__).resolve().parent.parent
            artifacts_dir = package_root / "data" / "artifacts"
            results_dir = package_root / "results"
            
            if model_path is None:
                candidate_paths = [
                    artifacts_dir / f"final_{self.model_type}_pipeline.pkl",
                    artifacts_dir / f"final_{self.model_type}.pkl",
                    results_dir / f"{self.model_type}_model.pkl"
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
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier

        y_vals = np.asarray(y)
        first_val = y_vals[0] if len(y_vals) > 0 else None
        if isinstance(first_val, (str, np.str_)):
            self.label_encoder_ = LabelEncoder()
            y_encoded = self.label_encoder_.fit_transform(y_vals)
        else:
            y_encoded = y_vals

        params = model_params or {}
        if self.model_type in ["logistic_regression", "lr"]:
            self.model_ = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42, **params)
        elif self.model_type in ["random_forest", "rf"]:
            self.model_ = RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1, **params)
        elif self.model_type in ["mlp", "pytorch_mlp"]:
            self.model_ = PyTorchMLPClassifier(random_state=42, **params)
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}. Select 'logistic_regression', 'random_forest', or 'mlp'.")

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

    def compute_feature_attributions(self, X):
        """
        Computes model-appropriate feature attributions / importances.
        - PyTorch MLP: Gradient x Input autograd feature attributions
        - Logistic Regression: Subtype-specific model coefficients
        - Random Forest: MDI Gini feature importances
        """
        if self.model_ is None:
            raise RuntimeError("Model is not loaded or trained.")

        # Extract underlying classifier if inside a Pipeline
        model_obj = self.model_
        if hasattr(model_obj, "named_steps"):
            model_obj = model_obj.named_steps.get("model", model_obj.named_steps.get("classifier", model_obj))

        if isinstance(model_obj, PyTorchMLPClassifier):
            return model_obj.compute_gradient_attributions(X)
        elif hasattr(model_obj, "coef_"):
            # Logistic Regression coefficients
            return model_obj.coef_
        elif hasattr(model_obj, "feature_importances_"):
            # Random Forest importances
            return model_obj.feature_importances_
        else:
            raise NotImplementedError(f"Feature attributions not implemented for {type(model_obj)}.")


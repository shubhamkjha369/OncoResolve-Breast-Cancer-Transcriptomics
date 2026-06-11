import os
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

class OncoPrognosis:
    """
    OncoResolve prognostic risk model.
    Wraps an L2-regularized (Ridge) Cox Proportional Hazards model
    to predict the Consensus Risk Score (CRS) for survival stratification.
    """
    def __init__(self, model_path=None, penalizer=0.5):
        self.penalizer = penalizer
        self.model_ = None
        
        # Determine default model path if not specified
        if model_path is None:
            package_root = Path(__file__).resolve().parent.parent
            model_path = package_root / "data" / "artifacts" / "survival_crs_ridge_model.pkl"
            
        if os.path.exists(model_path):
            self.load_pretrained(model_path)

    def load_pretrained(self, model_path):
        """
        Loads a pre-trained CoxPHFitter model.
        """
        self.model_ = joblib.load(model_path)
        return self

    def fit(self, X, survival_df, event_col="OS_STATUS_BIN", time_col="OS_MONTHS"):
        """
        Trains a new regularized Ridge Cox model.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Expression dataframe (samples as rows, genes as columns).
        survival_df : pd.DataFrame
            Dataframe containing overall survival time and status.
            Must be index-aligned with X.
        event_col : str
            Column name for event status (1 = event, 0 = censored).
        time_col : str
            Column name for survival time.
        """
        from lifelines import CoxPHFitter
        
        # Align expression data and survival metadata
        df_combined = X.join(survival_df[[time_col, event_col]], how="inner")
        
        self.model_ = CoxPHFitter(penalizer=self.penalizer, l1_ratio=0.0)
        self.model_.fit(df_combined, duration_col=time_col, event_col=event_col)
        return self

    def predict_risk(self, X):
        """
        Predicts the Consensus Risk Score (CRS) for the input samples.
        Higher scores indicate higher hazard / poorer prognosis.
        
        Parameters:
        -----------
        X : pd.DataFrame or np.ndarray
            Expression dataframe. Must have identical feature columns
            and names as the training features of the model.
        """
        if self.model_ is None:
            raise RuntimeError("Prognostic model is not loaded or trained. Call fit() or load_pretrained() first.")
            
        # lifelines expect pandas DataFrame with feature names
        if not isinstance(X, pd.DataFrame):
            # Try to restore column names if available on model
            feature_names = self.model_.summary.index.tolist()
            X_df = pd.DataFrame(X, columns=feature_names)
        else:
            X_df = X
            
        # Get partial hazard scores
        crs_scores = self.model_.predict_partial_hazard(X_df)
        return crs_scores

import pandas as pd
import numpy as np
from collections import Counter
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

class ConsensusSelector:
    """
    Ensemble consensus feature selector using:
    1. ANOVA F-Test (Linear class separation)
    2. LASSO L1 Penalty (Feature shrinkage)
    3. Random Forest Gini Importance (Non-linear interactions)
    """
    def __init__(self, top_k=50, random_state=42):
        self.top_k = top_k
        self.random_state = random_state
        self.consensus_features_ = []
        self.feature_frequencies_ = {}

    def fit(self, X, y, feature_names=None):
        """
        Fits the consensus ensemble to select the most stable features.
        
        Parameters:
        -----------
        X : np.ndarray or pd.DataFrame
            Scaled expression matrix.
        y : np.ndarray or pd.Series
            Subtype labels (or encoded labels).
        feature_names : list or np.ndarray, optional
            Gene symbols. If X is a DataFrame, columns are used.
        """
        if isinstance(X, pd.DataFrame):
            if feature_names is None:
                feature_names = X.columns
            X_arr = X.values
        else:
            X_arr = X
            if feature_names is None:
                feature_names = np.array([f"Gene_{i}" for i in range(X_arr.shape[1])])

        feature_names = np.array(feature_names)
        n_features = X_arr.shape[1]
        safe_k = min(self.top_k, n_features)

        # 1. ANOVA F-Test
        anova_sel = SelectKBest(score_func=f_classif, k=safe_k)
        anova_sel.fit(X_arr, y)
        anova_scores = pd.DataFrame({"gene": feature_names, "score": anova_sel.scores_}).sort_values("score", ascending=False)
        top_anova = set(anova_scores.head(safe_k)["gene"])

        # 2. Random Forest Importance
        rf = RandomForestClassifier(n_estimators=100, random_state=self.random_state, n_jobs=-1, max_depth=20)
        rf.fit(X_arr, y)
        rf_imp = pd.DataFrame({"gene": feature_names, "importance": rf.feature_importances_}).sort_values("importance", ascending=False)
        top_rf = set(rf_imp.head(safe_k)["gene"])

        # 3. LASSO L1 Penalty
        lasso = LogisticRegression(penalty="l1", solver="saga", max_iter=2000, random_state=self.random_state, n_jobs=-1)
        lasso.fit(X_arr, y)
        lasso_imp = pd.DataFrame({"gene": feature_names, "importance": np.abs(lasso.coef_).mean(axis=0)}).sort_values("importance", ascending=False)
        top_lasso = set(lasso_imp.head(safe_k)["gene"])

        # Consensus majority voting
        all_selected = list(top_anova) + list(top_rf) + list(top_lasso)
        self.feature_frequencies_ = Counter(all_selected)
        
        # Select features nominated by at least 2 out of 3 methods
        self.consensus_features_ = [gene for gene, freq in self.feature_frequencies_.items() if freq >= 2]
        # Sort by frequency and then name for stability
        self.consensus_features_ = sorted(self.consensus_features_, key=lambda g: (-self.feature_frequencies_[g], g))
        
        return self

    def transform(self, X, feature_names=None):
        """
        Subsets the input data to selected consensus features.
        """
        if isinstance(X, pd.DataFrame):
            return X[self.consensus_features_]
        
        if feature_names is None:
            raise ValueError("feature_names must be provided if input X is a NumPy array.")
        
        feature_list = list(feature_names)
        indices = [feature_list.index(gene) for gene in self.consensus_features_ if gene in feature_list]
        return X[:, indices]

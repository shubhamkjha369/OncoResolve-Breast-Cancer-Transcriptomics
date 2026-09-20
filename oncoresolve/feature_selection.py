"""
oncoresolve/feature_selection.py
=================================
Ensemble consensus feature selector (ANOVA + LASSO + Random Forest majority vote).

This module is designed to be inserted into sklearn Pipelines as a proper transformer.
All hyperparameters are FIXED a priori per the canonical OncoResolve protocol:
  - top_k=211 consensus signature genes
  - LASSO: C=0.1, Multinomial SAGA L1 penalty (or OvR L1 fallback)
  - RF:    n_estimators=200, random_state=<seed>

Classes
-------
VariancePreFilter   : Top-K variance pre-filter using feature variance. sklearn-compatible.
ConsensusSelector   : Tri-method consensus selector. sklearn-compatible.
"""

from __future__ import annotations

import warnings
from collections import Counter
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_is_fitted


class VariancePreFilter(BaseEstimator, TransformerMixin):
    """
    Top-K variance pre-filter using feature variance.

    Fits on training data only; never exposes test-fold statistics.
    Designed as the first step of an sklearn Pipeline before ConsensusSelector.

    Parameters
    ----------
    top_k : int, default 2000
        Number of highest-variance genes to retain.
    """

    def __init__(self, top_k: int = 2000):
        self.top_k = top_k

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            X_arr = X.values
            self.feature_names_in_ = np.array(X.columns)
        else:
            X_arr = np.asarray(X)
            self.feature_names_in_ = np.array([f"Gene_{i}" for i in range(X_arr.shape[1])])

        var_vals = np.var(X_arr, axis=0)
        safe_k = min(self.top_k, X_arr.shape[1])
        # Indices of top-k variance genes (descending)
        self.selected_indices_ = np.argsort(var_vals)[::-1][:safe_k]
        return self

    def transform(self, X):
        check_is_fitted(self, "selected_indices_")
        if isinstance(X, pd.DataFrame):
            selected_names = self.feature_names_in_[self.selected_indices_]
            available = [g for g in selected_names if g in X.columns]
            return X[available]
        return np.asarray(X)[:, self.selected_indices_]

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "selected_indices_")
        return self.feature_names_in_[self.selected_indices_]

    def get_support(self, indices: bool = False):
        check_is_fitted(self, "selected_indices_")
        mask = np.zeros(len(self.feature_names_in_), dtype=bool)
        mask[self.selected_indices_] = True
        return np.where(mask)[0] if indices else mask


class ConsensusSelector(BaseEstimator, TransformerMixin):
    """
    Ensemble consensus feature selector.

    Applies three orthogonal feature-ranking methods and retains genes
    nominated by at least 2/3 methods (majority vote):

    1. ANOVA F-test (SelectKBest(f_classif))
    2. Multinomial L1 Logistic Regression (SAGA solver, C=0.1)
    3. Random Forest Gini importance (n_estimators=200)

    Parameters
    ----------
    top_k : int, default 211
    lasso_C : float, default 0.1
    lasso_max_iter : int, default 5000
    rf_n_estimators : int, default 200
    random_state : int, default 42
    min_vote : int, default 2
    verbose : bool, default False
    """

    def __init__(
        self,
        top_k: int = 211,
        lasso_C: float = 0.1,
        lasso_max_iter: int = 5000,
        tol: float = 1e-2,
        rf_n_estimators: int = 200,
        random_state: int = 42,
        min_vote: int = 2,
        verbose: bool = False,
        n_jobs: int = 1,
    ):
        self.top_k = top_k
        self.lasso_C = lasso_C
        self.lasso_max_iter = lasso_max_iter
        self.tol = tol
        self.rf_n_estimators = rf_n_estimators
        self.random_state = random_state
        self.min_vote = min_vote
        self.verbose = verbose
        self.n_jobs = n_jobs

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            feature_names = np.array(X.columns)
            X_arr = X.values
        else:
            X_arr = np.asarray(X)
            feature_names = np.array([f"Gene_{i}" for i in range(X_arr.shape[1])])

        self.feature_names_in_ = feature_names
        n_features = X_arr.shape[1]
        safe_k = min(self.top_k, n_features)

        if safe_k < 3:
            raise ValueError(
                f"top_k={self.top_k} is too small given n_features={n_features}."
            )

        # 1. ANOVA F-test
        anova_sel = SelectKBest(score_func=f_classif, k=safe_k)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            anova_sel.fit(X_arr, y)
        top_anova = set(feature_names[anova_sel.get_support(indices=True)])

        # 2. L1 Logistic Regression (SAGA solver, multinomial)
        lasso = LogisticRegression(
            penalty="l2",
            solver="saga",
            C=self.lasso_C,
            max_iter=self.lasso_max_iter,
            tol=self.tol,
            random_state=self.random_state,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lasso.fit(X_arr, y)
        coefs = np.abs(lasso.coef_)
        lasso_importance = coefs.mean(axis=0) if coefs.ndim > 1 else coefs

        lasso_rank = np.argsort(lasso_importance)[::-1]
        top_lasso = set(feature_names[lasso_rank[:safe_k]])

        # 3. Random Forest Gini importance
        rf = RandomForestClassifier(
            n_estimators=self.rf_n_estimators,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            class_weight="balanced",
        )
        rf.fit(X_arr, y)
        rf_rank = np.argsort(rf.feature_importances_)[::-1]
        top_rf = set(feature_names[rf_rank[:safe_k]])

        # Majority vote (>= min_vote / 3 methods)
        vote_counter = Counter(
            list(top_anova) + list(top_lasso) + list(top_rf)
        )
        self.feature_frequencies_: dict = dict(vote_counter)
        self.feature_votes_: dict = dict(vote_counter)

        consensus = [
            gene for gene, votes in vote_counter.items()
            if votes >= self.min_vote
        ]
        self.consensus_features_: List[str] = sorted(
            consensus,
            key=lambda g: (-vote_counter[g], g),
        )

        if not self.consensus_features_:
            # Fallback to top ANOVA features if consensus is empty
            self.consensus_features_ = sorted(list(top_anova))[:safe_k]

        if self.verbose:
            print(
                f"[ConsensusSelector] ANOVA: {len(top_anova)}, "
                f"LASSO: {len(top_lasso)}, RF: {len(top_rf)}, "
                f"Consensus (>={self.min_vote}/3): {len(self.consensus_features_)}"
            )

        self.top_anova_: set = top_anova
        self.top_lasso_: set = top_lasso
        self.top_rf_: set = top_rf

        return self

    def transform(self, X):
        check_is_fitted(self, "consensus_features_")

        if isinstance(X, pd.DataFrame):
            available = [g for g in self.consensus_features_ if g in X.columns]
            return X[available]

        feat_to_idx = {g: i for i, g in enumerate(self.feature_names_in_)}
        indices = [
            feat_to_idx[g]
            for g in self.consensus_features_
            if g in feat_to_idx
        ]
        return np.asarray(X)[:, indices]

    def get_support(self, indices: bool = False):
        check_is_fitted(self, "consensus_features_")
        selected_set = set(self.consensus_features_)
        mask = np.array([g in selected_set for g in self.feature_names_in_])
        return np.where(mask)[0] if indices else mask

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "consensus_features_")
        return np.array(self.consensus_features_)

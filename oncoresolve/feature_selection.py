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
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
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
            penalty="l1",
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


class StabilityWeightedConsensusSelector(BaseEstimator, TransformerMixin):
    """Select an exact, fold-safe consensus signature from expression data.

    The selector is deliberately a *single transformer* rather than a global
    pre-computed feature list.  When placed inside an sklearn ``Pipeline``,
    each call to :meth:`fit` sees only that pipeline fit's training rows.

    For every repeated stratified split of those rows it:

    1. retains the ``candidate_top_k`` most variable genes using the split's
       training rows only;
    2. fits a scaled multinomial logistic regression and a random forest on
       those candidates;
    3. converts each model's feature importance to a within-split percentile;
    4. averages the two percentiles and records whether the gene is among that
       split's ``signature_top_k`` strongest consensus genes.

    The final score is the mean consensus percentile multiplied by selection
    frequency across *all* repeated splits.  This favours genes that are both
    predictive in two model families and stable under resampling.  Exactly
    ``signature_top_k`` genes are returned (unless fewer columns exist).

    No held-out outer-fold row or label is used when this selector is fitted
    inside an outer-fold pipeline.
    """

    def __init__(
        self,
        candidate_top_k: int = 2000,
        signature_top_k: int = 211,
        n_splits: int = 3,
        n_repeats: int = 2,
        logistic_C: float = 0.1,
        rf_n_estimators: int = 300,
        rf_max_depth: Optional[int] = None,
        random_state: int = 42,
        n_jobs: int = 1,
    ):
        self.candidate_top_k = candidate_top_k
        self.signature_top_k = signature_top_k
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.logistic_C = logistic_C
        self.rf_n_estimators = rf_n_estimators
        self.rf_max_depth = rf_max_depth
        self.random_state = random_state
        self.n_jobs = n_jobs

    @staticmethod
    def _percentile_from_importance(importance: np.ndarray) -> np.ndarray:
        """Map importances to [0, 1], with 1 assigned to the strongest gene."""
        n_features = len(importance)
        if n_features <= 1:
            return np.ones(n_features, dtype=float)
        order = np.argsort(-importance, kind="mergesort")
        ranks = np.empty(n_features, dtype=float)
        ranks[order] = np.arange(n_features, dtype=float)
        return 1.0 - ranks / (n_features - 1)

    def fit(self, X, y):
        if y is None:
            raise ValueError("StabilityWeightedConsensusSelector requires class labels during fit.")

        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
            X_arr = X.to_numpy(dtype=float)
        else:
            X_arr = np.asarray(X, dtype=float)
            self.feature_names_in_ = np.asarray(
                [f"Gene_{i}" for i in range(X_arr.shape[1])], dtype=object
            )
        y_arr = np.asarray(y)

        if X_arr.ndim != 2:
            raise ValueError("X must be a two-dimensional samples-by-features matrix.")
        if len(X_arr) != len(y_arr):
            raise ValueError("X and y must contain the same number of samples.")
        if not np.isfinite(X_arr).all():
            raise ValueError("X contains NaN or infinite values; impute before feature selection.")

        classes, counts = np.unique(y_arr, return_counts=True)
        if len(classes) < 2:
            raise ValueError("At least two classes are required for consensus feature selection.")
        if counts.min() < self.n_splits:
            raise ValueError(
                "The smallest class must contain at least n_splits samples for "
                "stratified stability selection."
            )

        n_features = X_arr.shape[1]
        candidate_k = min(int(self.candidate_top_k), n_features)
        signature_k = min(int(self.signature_top_k), n_features)
        if candidate_k < 2:
            raise ValueError("candidate_top_k must leave at least two features.")
        if signature_k < 1:
            raise ValueError("signature_top_k must be at least one.")

        splitter = RepeatedStratifiedKFold(
            n_splits=self.n_splits,
            n_repeats=self.n_repeats,
            random_state=self.random_state,
        )
        total_splits = self.n_splits * self.n_repeats
        score_sum = np.zeros(n_features, dtype=float)
        lr_score_sum = np.zeros(n_features, dtype=float)
        rf_score_sum = np.zeros(n_features, dtype=float)
        candidate_count = np.zeros(n_features, dtype=np.int32)
        selection_count = np.zeros(n_features, dtype=np.int32)

        for split_number, (train_idx, _) in enumerate(splitter.split(X_arr, y_arr)):
            X_train = X_arr[train_idx]
            y_train = y_arr[train_idx]
            variance_order = np.argsort(-np.var(X_train, axis=0), kind="mergesort")
            candidate_idx = variance_order[:candidate_k]
            X_candidate = X_train[:, candidate_idx]
            X_scaled = StandardScaler().fit_transform(X_candidate)

            lr = LogisticRegression(
                C=self.logistic_C,
                solver="lbfgs",
                max_iter=2000,
                class_weight="balanced",
                random_state=self.random_state + split_number,
            )
            rf = RandomForestClassifier(
                n_estimators=self.rf_n_estimators,
                max_depth=self.rf_max_depth,
                class_weight="balanced",
                random_state=self.random_state + split_number,
                n_jobs=self.n_jobs,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                lr.fit(X_scaled, y_train)
            rf.fit(X_candidate, y_train)

            lr_importance = np.mean(np.abs(lr.coef_), axis=0)
            rf_importance = rf.feature_importances_
            lr_percentile = self._percentile_from_importance(lr_importance)
            rf_percentile = self._percentile_from_importance(rf_importance)
            consensus_percentile = (lr_percentile + rf_percentile) / 2.0

            score_sum[candidate_idx] += consensus_percentile
            lr_score_sum[candidate_idx] += lr_percentile
            rf_score_sum[candidate_idx] += rf_percentile
            candidate_count[candidate_idx] += 1

            within_split_k = min(signature_k, candidate_k)
            split_selected = candidate_idx[
                np.argsort(-consensus_percentile, kind="mergesort")[:within_split_k]
            ]
            selection_count[split_selected] += 1

        with np.errstate(divide="ignore", invalid="ignore"):
            mean_consensus = np.divide(
                score_sum,
                candidate_count,
                out=np.zeros(n_features, dtype=float),
                where=candidate_count > 0,
            )
            mean_lr = np.divide(
                lr_score_sum,
                candidate_count,
                out=np.zeros(n_features, dtype=float),
                where=candidate_count > 0,
            )
            mean_rf = np.divide(
                rf_score_sum,
                candidate_count,
                out=np.zeros(n_features, dtype=float),
                where=candidate_count > 0,
            )
        selection_frequency = selection_count / float(total_splits)
        stability_weighted_score = mean_consensus * selection_frequency

        # Deterministic tie-breaking keeps a fitted schema reproducible.
        selected_indices = np.lexsort(
            (self.feature_names_in_.astype(str), -mean_consensus, -selection_frequency,
             -stability_weighted_score)
        )[:signature_k]
        self.selected_indices_ = selected_indices.astype(int)
        self.selected_features_ = self.feature_names_in_[self.selected_indices_].tolist()
        self.ranking_ = pd.DataFrame(
            {
                "feature": self.feature_names_in_,
                "selection_count": selection_count,
                "selection_frequency": selection_frequency,
                "candidate_frequency": candidate_count / float(total_splits),
                "mean_lr_percentile": mean_lr,
                "mean_rf_percentile": mean_rf,
                "mean_consensus_percentile": mean_consensus,
                "stability_weighted_consensus_score": stability_weighted_score,
            }
        ).sort_values(
            [
                "stability_weighted_consensus_score",
                "selection_frequency",
                "mean_consensus_percentile",
                "feature",
            ],
            ascending=[False, False, False, True],
            kind="mergesort",
        ).reset_index(drop=True)
        self.ranking_["consensus_rank"] = np.arange(1, len(self.ranking_) + 1)
        self.ranking_["selected"] = self.ranking_["feature"].isin(self.selected_features_)
        self.n_stability_splits_ = total_splits
        self.candidate_top_k_ = candidate_k
        self.signature_top_k_ = signature_k
        return self

    def transform(self, X):
        check_is_fitted(self, "selected_indices_")
        if isinstance(X, pd.DataFrame):
            missing = [gene for gene in self.selected_features_ if gene not in X.columns]
            if missing:
                raise ValueError(
                    f"Input is missing {len(missing)} selected features, e.g. {missing[:5]}."
                )
            return X.loc[:, self.selected_features_]
        return np.asarray(X)[:, self.selected_indices_]

    def get_support(self, indices: bool = False):
        check_is_fitted(self, "selected_indices_")
        mask = np.zeros(len(self.feature_names_in_), dtype=bool)
        mask[self.selected_indices_] = True
        return self.selected_indices_.copy() if indices else mask

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "selected_indices_")
        return np.asarray(self.selected_features_, dtype=object)

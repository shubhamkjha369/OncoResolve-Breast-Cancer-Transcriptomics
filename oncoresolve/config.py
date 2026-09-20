"""
oncoresolve/config.py
=====================
Single source of truth for OncoResolve hyperparameters and protocol settings.
"""

from dataclasses import dataclass
from pathlib import Path

# Path Configuration
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ARTIFACT_DIR = DATA_DIR / "artifacts"
RESULTS_DIR = PROJECT_ROOT / "results"
EXT_DIR = DATA_DIR / "external_cohort"

@dataclass(frozen=True)
class OncoResolveConfig:
    random_state: int = 42

    # Feature selection parameters
    variance_top_k: int = 2000
    selector_top_k: int = 211
    consensus_threshold: int = 2
    consensus_min_vote: int = 2

    # LASSO parameters
    lasso_C: float = 0.1
    lasso_max_iter: int = 5000
    lasso_solver: str = "saga"

    # Random Forest parameters
    rf_n_estimators: int = 200

    # Decision Threshold Gating
    her2_confidence_threshold: float = 0.40

    # PCA for CUS
    cus_pca_components: int = 2

    # PSN parameters
    psn_percentile: float = 85.0

    # Stability evaluation
    stability_B: int = 100
    stability_P: int = 500

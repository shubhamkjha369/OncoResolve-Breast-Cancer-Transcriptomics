"""
oncoresolve/utils.py
=====================
Shared utilities for the OncoResolve pipeline:

  - harmonize_namespaces : Entrez → HUGO gene symbol mapping
  - scale_cohort         : Z-score normalization (supports frozen reference scaler)
  - save_reference_scaler / load_reference_scaler : persist TCGA reference normalization
  - align_features       : Align expression DataFrame to a required gene set

Normalization protocol
----------------------
Two modes are supported and serve different purposes:

  ``reference_scaler=None`` (fit-on-incoming)
      Fits a new StandardScaler on the supplied DataFrame.
      Used for the TRAINING cohort preprocessing step.
      Must not be used for external validation (would be cohort-relative).

  ``reference_scaler=<fitted StandardScaler>`` (frozen reference)
      Applies a pre-fitted scaler — typically fitted on the 784-sample
      TCGA discovery cohort — to any incoming DataFrame.
      Used for external cohort validation (SCAN-B, SMC, METABRIC) and
      for any future deployment inference.
      Guarantees: no .fit() call touches external cohort data.

  Both are cohort-level operations; neither is an N-of-1 normalization.
  "Single-patient prospective" normalization is NOT implemented here.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Gene namespace harmonization

def harmonize_namespaces(
    df: pd.DataFrame,
    mapping_path: Union[str, Path],
) -> pd.DataFrame:
    """
    Map Entrez gene IDs (or other identifiers) in column names to HUGO gene symbols.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with genes as columns.
    mapping_path : str or Path
        Path to the ``tcga_entrez_to_hugo.pkl`` mapping file.

    Returns
    -------
    pd.DataFrame
        DataFrame with HUGO symbol columns. Unmapped columns are dropped.
        Duplicate HUGO symbols (from multiple Entrez IDs) are averaged.
    """
    mapping_path = Path(mapping_path)
    if not mapping_path.exists():
        # Check alternative default artifact locations
        alt_paths = [
            Path("data/artifacts/tcga_entrez_to_hugo.pkl"),
            Path("data/processed/tcga_entrez_to_hugo.pkl"),
            Path("../data/artifacts/tcga_entrez_to_hugo.pkl"),
            Path("../data/processed/tcga_entrez_to_hugo.pkl"),
        ]
        for alt in alt_paths:
            if alt.exists():
                mapping_path = alt
                break
    if not mapping_path.exists():
        raise FileNotFoundError(
            f"Mapping file '{mapping_path}' not found. Please ensure tcga_entrez_to_hugo.pkl artifact exists."
        )
    entrez_to_hugo = joblib.load(mapping_path)

    mapping_dict = {str(k).split('.')[0].strip(): str(v).strip() for k, v in entrez_to_hugo.items()}
    valid_hugo_set = set(mapping_dict.values())

    df_clean = df.copy()
    
    new_cols = []
    for c in df_clean.columns:
        c_str = str(c).strip()
        c_norm = c_str.split('.')[0] if '.' in c_str else c_str
        
        if c_str in valid_hugo_set:
            new_cols.append(c_str)
        elif c_norm in mapping_dict:
            new_cols.append(mapping_dict[c_norm])
        else:
            new_cols.append(np.nan)
            
    df_clean.columns = new_cols

    # Drop unmapped columns
    df_clean = df_clean.loc[:, df_clean.columns.notna()]
    
    if len(df_clean.columns) == 0:
        raise ValueError("harmonize_namespaces: 0 columns remain after namespace mapping. Check input identifiers.")

    # Average duplicates (multiple Entrez IDs → same HUGO symbol)
    if not df_clean.columns.is_unique:
        df_clean = df_clean.T.groupby(df_clean.columns).mean().T

    return df_clean


# Z-score normalization

def scale_cohort(
    df: pd.DataFrame,
    reference_scaler: Optional[StandardScaler] = None,
) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Z-score normalization (StandardScaler).

    Parameters
    ----------
    df : pd.DataFrame
        Expression DataFrame (samples × genes).
    reference_scaler : fitted StandardScaler or None
        If provided: applies the frozen reference transform.
            → Use for external cohort validation and deployment.
            → No .fit() call touches ``df``.
        If None: fits a new StandardScaler on ``df``.
            → Use only for training cohort preprocessing.
            → Save the returned scaler as the reference for future use.

    Returns
    -------
    df_scaled : pd.DataFrame
        Normalized DataFrame with identical index and columns.
    scaler : StandardScaler
        The scaler used (frozen reference or newly fitted).
    """
    if reference_scaler is not None:
        if hasattr(reference_scaler, "feature_names_in_"):
            expected = list(reference_scaler.feature_names_in_)
            actual = list(df.columns)
            if actual != expected:
                if set(actual) == set(expected):
                    df = df.reindex(columns=expected)
                else:
                    raise ValueError(
                        f"scale_cohort: Input columns do not match reference scaler features.\n"
                        f"  Expected ({len(expected)}): {expected[:5]}...\n"
                        f"  Actual ({len(actual)}): {actual[:5]}..."
                    )
        scaled = reference_scaler.transform(df)
        return (
            pd.DataFrame(scaled, index=df.index, columns=df.columns),
            reference_scaler,
        )
    else:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(df)
        return (
            pd.DataFrame(scaled, index=df.index, columns=df.columns),
            scaler,
        )


def save_reference_scaler(scaler: StandardScaler, path: Union[str, Path]) -> None:
    """
    Serialize a fitted StandardScaler to disk for frozen reference normalization.

    Parameters
    ----------
    scaler : fitted StandardScaler
        The TCGA training scaler to persist.
    path : str or Path
        Destination file path (e.g., ``data/artifacts/reference_scaler_tcga784.pkl``).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, path)


def load_reference_scaler(path: Union[str, Path]) -> StandardScaler:
    """
    Load a serialized reference scaler.

    Parameters
    ----------
    path : str or Path
        Path to the pickled StandardScaler.

    Returns
    -------
    StandardScaler
        Fitted reference scaler ready for ``.transform()`` calls.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Reference scaler not found at: {path}\n"
            "Fit a scaler on the TCGA discovery cohort and save it with "
            "save_reference_scaler() before applying to external cohorts."
        )
    return joblib.load(path)


# Feature alignment

def align_features(
    df: pd.DataFrame,
    required_features: List[str],
    reference_scaler: Optional[StandardScaler] = None,
    fill_value: Optional[float] = None,
    warn_missing: bool = True,
    mapping_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Align a DataFrame's columns to a required feature set.

    Supports bidirectional namespace resolution between Entrez IDs and HUGO symbols
    using `tcga_entrez_to_hugo.pkl`.

    If reference_scaler is provided, missing genes are filled with reference_scaler.mean_[j]
    in raw space so that post-normalization z = 0.0 (reference cohort mean).

    Parameters
    ----------
    df : pd.DataFrame
        Input expression DataFrame.
    required_features : list of str
        Ordered list of required gene names.
    reference_scaler : Optional[StandardScaler]
        Fitted reference scaler for extracting gene-specific means for imputation.
    fill_value : Optional[float]
        Fill value for missing genes. If None and reference_scaler is provided,
        uses reference_scaler.mean_[j]. Otherwise defaults to 0.0.
    warn_missing : bool, default True
        Emit a UserWarning listing missing genes.
    mapping_path : Optional[str or Path]
        Path to tcga_entrez_to_hugo.pkl mapping file.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns exactly matching ``required_features`` in order.
    """
    df_aligned = pd.DataFrame(index=df.index)

    # Load Entrez to HUGO mapping dictionary
    entrez_to_hugo = {}
    hugo_to_entrez = {}
    try:
        m_path = Path(mapping_path) if mapping_path else Path("data/artifacts/tcga_entrez_to_hugo.pkl")
        if not m_path.exists():
            for alt in [
                Path("data/processed/tcga_entrez_to_hugo.pkl"),
                Path("../data/artifacts/tcga_entrez_to_hugo.pkl"),
                Path("../data/processed/tcga_entrez_to_hugo.pkl"),
            ]:
                if alt.exists():
                    m_path = alt
                    break
        if m_path.exists():
            raw_map = joblib.load(m_path)
            entrez_to_hugo = {str(k).split('.')[0].strip(): str(v).strip() for k, v in raw_map.items()}
            hugo_to_entrez = {str(v).strip(): str(k).split('.')[0].strip() for k, v in raw_map.items()}
    except Exception:
        pass

    col_map = {str(c).split('.')[0].strip(): c for c in df.columns}
    missing = []

    feat_idx_map = {}
    if reference_scaler is not None and hasattr(reference_scaler, "feature_names_in_"):
        feat_idx_map = {name: idx for idx, name in enumerate(reference_scaler.feature_names_in_)}

    for idx, feat in enumerate(required_features):
        feat_str = str(feat).split('.')[0].strip()
        hugo = entrez_to_hugo.get(feat_str, None)
        entrez = hugo_to_entrez.get(feat_str, None)

        target_col = None
        if feat_str in col_map:
            target_col = col_map[feat_str]
        elif hugo and hugo in col_map:
            target_col = col_map[hugo]
        elif entrez and entrez in col_map:
            target_col = col_map[entrez]
        elif feat_str in df.columns:
            target_col = feat_str

        if target_col is not None:
            val = df[target_col]
            if isinstance(val, pd.DataFrame):
                val = val.iloc[:, 0]
            df_aligned[feat] = val.values
        else:
            missing.append(feat)
            if fill_value is not None:
                df_aligned[feat] = fill_value
            elif reference_scaler is not None and feat in feat_idx_map:
                df_aligned[feat] = reference_scaler.mean_[feat_idx_map[feat]]
            else:
                df_aligned[feat] = 0.0

    if missing and warn_missing:
        import warnings
        warnings.warn(
            f"align_features: {len(missing)} feature(s) missing from input DataFrame after namespace resolution. "
            f"Missing: {missing[:5]}{'...' if len(missing) > 5 else ''}",
            UserWarning,
        )

    return df_aligned.reindex(columns=required_features)


# I/O Helpers

def safe_joblib_dump(value: any, filename: Union[str, Path], **kwargs) -> None:
    """Safely dump a joblib artifact by writing to a temporary file first."""
    p = Path(filename).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    try:
        joblib.dump(value, str(tmp), **kwargs)
        os.replace(tmp, p)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except Exception:
                pass


def safe_to_parquet(df: pd.DataFrame, filename: Union[str, Path], **kwargs) -> None:
    """Safely write a DataFrame to Parquet format, creating directories if needed."""
    p = Path(filename).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        try:
            p.unlink()
        except Exception:
            pass
    return df.to_parquet(str(p), **kwargs)


def safe_to_csv(df: pd.DataFrame, filename: Union[str, Path], **kwargs) -> None:
    """Safely write a DataFrame to CSV format, creating directories if needed."""
    p = Path(filename).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        try:
            p.unlink()
        except Exception:
            pass
    return df.to_csv(str(p), **kwargs)


def align_external_cohort(
    df_expr: pd.DataFrame,
    tcga_scaler: StandardScaler,
    candidate_features: List[str],
    entrez_to_hugo: Optional[dict] = None,
) -> Tuple[np.ndarray, int]:
    """
    Aligns external cohort gene expression to the required candidate feature set
    and applies cross-platform Z-score normalization matching the TCGA reference scale.

    Parameters
    ----------
    df_expr : pd.DataFrame
        Raw or log-transformed expression matrix for the external cohort (samples x genes).
    tcga_scaler : StandardScaler
        Frozen TCGA reference StandardScaler fitted on discovery cohort training features.
    candidate_features : list of str
        Ordered list of candidate gene feature identifiers.
    entrez_to_hugo : dict, optional
        Dictionary mapping Entrez IDs to HUGO gene symbols.

    Returns
    -------
    X_aligned : np.ndarray
        Aligned and scaled expression array of shape (n_samples, len(candidate_features)).
    mapped_count : int
        Number of successfully matched features.
    """
    cand_features_str = [str(g).strip() for g in candidate_features]
    if entrez_to_hugo is None:
        entrez_to_hugo = {}

    ext_col_dict = {str(c).strip().upper(): c for c in df_expr.columns}

    n_samples = len(df_expr)
    n_features = len(cand_features_str)

    X_raw = np.zeros((n_samples, n_features), dtype=np.float64)

    # Initialize missing genes with reference scaler mean
    for j in range(n_features):
        X_raw[:, j] = tcga_scaler.mean_[j]

    mapped_indices = []
    for j, gene_str in enumerate(cand_features_str):
        matched_col = None
        if gene_str.upper() in ext_col_dict:
            matched_col = ext_col_dict[gene_str.upper()]
        else:
            hugo = entrez_to_hugo.get(gene_str)
            if hugo and hugo.upper() in ext_col_dict:
                matched_col = ext_col_dict[hugo.upper()]

        if matched_col is not None:
            vals = pd.to_numeric(df_expr[matched_col], errors="coerce").to_numpy()
            m_v = np.nanmean(vals)
            if np.isnan(m_v):
                m_v = tcga_scaler.mean_[j]
            vals = np.nan_to_num(vals, nan=m_v)
            X_raw[:, j] = vals
            mapped_indices.append(j)

    # Cross-platform z-score alignment to TCGA training scale
    X_aligned = X_raw.copy()
    for j in mapped_indices:
        vals = X_raw[:, j]
        m_ext = np.mean(vals)
        s_ext = np.std(vals)
        if s_ext < 1e-6:
            s_ext = 1.0
        z_ext = (vals - m_ext) / s_ext
        X_aligned[:, j] = tcga_scaler.mean_[j] + z_ext * tcga_scaler.scale_[j]

    return X_aligned, len(mapped_indices)



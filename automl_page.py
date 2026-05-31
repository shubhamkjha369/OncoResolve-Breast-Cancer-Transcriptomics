"""
automl_page.py - Premium, research-grade AutoML tab with two functional modes:
1. Direct Clinical Subtype Predictor (runs pre-trained models on internal/external data).
2. Single-click end-to-end AutoML training with live progress and reusable session models.
"""
import html
import time
import traceback
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

import pipeline_engine as pe


PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#fafaf7",
    font=dict(family="Inter", color="#334155"),
    title_font=dict(size=15, color="#1e293b"),
    margin=dict(t=50, b=40),
)
GRID_COLOR = "#eeefe9"
SUBTYPE_COLORS = {
    "basal": "#ef4444",
    "HER": "#f59e0b",
    "luminal_A": "#10b981",
    "luminal_B": "#3b82f6",
    "normal": "#ec4899",
}

PRETRAINED_RF_KEY = "pretrained_rf"
PRETRAINED_MLP_KEY = "pretrained_mlp"
TRAINED_PREFIX = "trained::"
TRAINED_PREDICTORS_KEY = "latest_trained_predictors"
TRAINED_META_KEY = "latest_trained_pipeline_meta"
SELECT_TRAINED_ON_NEXT_RENDER_KEY = "pred_select_latest_trained_on_next_render"

INTERNAL_SOURCE = "Internal Clinical Cohort (GSE45827 - 137 Patients)"
CUSTOM_SOURCE = "Upload Custom Prediction Dataset (CSV/Parquet)"
LATEST_TRAINING_SOURCE = "Use Latest Training Dataset from Training Tab"

IDENTIFIER_COLUMN_NAMES = {
    "",
    "id",
    "ids",
    "index",
    "sample",
    "samples",
    "sample_id",
    "sample_ids",
    "sampleid",
    "sample_name",
    "sample_names",
    "sample_barcode",
    "patient",
    "patients",
    "patient_id",
    "patient_ids",
    "patientid",
    "patient_name",
    "patient_names",
    "pateint",
    "pateints",
    "pateint_id",
    "pateint_ids",
    "pateintid",
    "subject",
    "subjects",
    "subject_id",
    "subject_ids",
    "case",
    "cases",
    "case_id",
    "case_ids",
    "specimen",
    "specimen_id",
    "barcode",
    "accession",
    "geo_accession",
    "gsm",
}

CONSOLE_CSS = """
<style>
.console-output {
    background: #1a1b26; color: #a9b1d6; font-family: 'Consolas','Fira Code',monospace;
    font-size: 13px; padding: 16px 20px; border-radius: 10px; max-height: 340px;
    overflow-y: auto; line-height: 1.7; border: 1px solid #2a2b3d;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}
.console-output .log-time { color: #565f89; }
.console-output .log-ok { color: #9ece6a; }
.console-output .log-info { color: #7aa2f7; }
.console-output .log-warn { color: #e0af68; }
.console-output .log-step { color: #bb9af7; font-weight: 600; }
</style>
"""


class LiveConsole:
    """Terminal-style console that updates in real time via st.empty()."""

    def __init__(self, placeholder):
        self._ph = placeholder
        self._lines = []
        self._start = time.time()

    def log(self, msg):
        elapsed = time.time() - self._start
        safe_msg = html.escape(str(msg))
        ts = f'<span class="log-time">[{elapsed:6.1f}s]</span>'
        if safe_msg.startswith("[") or safe_msg.startswith("Phase"):
            styled = f'<span class="log-step">{safe_msg}</span>'
        elif "done" in safe_msg.lower() or "success" in safe_msg.lower() or "[ok]" in safe_msg.lower():
            styled = f'<span class="log-ok">{safe_msg}</span>'
        elif "High-dim" in safe_msg or "Warning" in safe_msg:
            styled = f'<span class="log-warn">{safe_msg}</span>'
        else:
            styled = f'<span class="log-info">{safe_msg}</span>'
        self._lines.append(f"{ts} {styled}")
        console_html = CONSOLE_CSS + '<div class="console-output">' + "<br>".join(self._lines) + "</div>"
        self._ph.markdown(console_html, unsafe_allow_html=True)

    def finish(self, success=True):
        elapsed = time.time() - self._start
        if success:
            line = (
                f'<span class="log-time">[{elapsed:6.1f}s]</span> '
                f'<span class="log-ok">DONE ({elapsed:.1f}s total)</span>'
            )
        else:
            line = f'<span class="log-time">[{elapsed:6.1f}s]</span> <span style="color:#f7768e;">FAILED</span>'
        self._lines.append(line)
        console_html = CONSOLE_CSS + '<div class="console-output">' + "<br>".join(self._lines) + "</div>"
        self._ph.markdown(console_html, unsafe_allow_html=True)


def custom_card(val, label, accent=False):
    cls = "accent-card" if accent else "metric-card"
    return f'<div class="{cls}"><div class="metric-value">{val}</div><div class="metric-label">{label}</div></div>'


def render(card_fn=None):
    # Inject custom clinical premium styles.
    st.markdown(
        """
<style>
/* Pill-based segment tabs */
div[data-baseweb="tab-list"] {
    background-color: #ebebe5 !important;
    padding: 6px 12px !important;
    border-radius: 12px !important;
    gap: 16px !important;
    border-bottom: none !important;
}
div[data-baseweb="tab"] {
    background-color: transparent !important;
    color: #475569 !important;
    border-radius: 10px !important;
    padding: 10px 24px !important;
    margin: 0 4px !important;
    font-weight: 600 !important;
    border: none !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
}
div[data-baseweb="tab"]:hover {
    color: #4f46e5 !important;
    background-color: rgba(255, 255, 255, 0.3) !important;
}
div[data-baseweb="tab"][aria-selected="true"] {
    background-color: #ffffff !important;
    color: #4f46e5 !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05) !important;
}
div[data-baseweb="tab-border"] {
    display: none !important;
}

/* Premium Card Layout containers */
div[data-testid="stVerticalBlockBordered"] {
    background-color: #ffffff !important;
    border: 1px solid #dcdcd3 !important;
    border-radius: 16px !important;
    padding: 28px !important;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.015) !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
}
div[data-testid="stVerticalBlockBordered"]:hover {
    border-color: #4f46e5 !important;
    box-shadow: 0 10px 30px rgba(79, 70, 229, 0.07) !important;
    transform: translateY(-2px);
}

/* Form inputs: selectbox, number input, sliders */
div[data-testid="stSelectbox"] > div,
div[data-testid="stNumberInput"] > div,
div[data-testid="stSlider"] > div {
    border-radius: 8px !important;
}

/* Custom styled action buttons */
div[data-testid="stVerticalBlockBordered"] button[data-testid="baseButton-secondary"] {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 24px !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    letter-spacing: 0.5px !important;
    box-shadow: 0 4px 14px rgba(79, 70, 229, 0.25) !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    width: 100% !important;
}
div[data-testid="stVerticalBlockBordered"] button[data-testid="baseButton-secondary"]:hover {
    background: linear-gradient(135deg, #4338ca, #6d28d9) !important;
    box-shadow: 0 6px 20px rgba(79, 70, 229, 0.35) !important;
    transform: translateY(-1px) !important;
}
div[data-testid="stVerticalBlockBordered"] button[data-testid="baseButton-secondary"]:active {
    transform: translateY(1px) !important;
}

/* File Uploader styling */
div[data-testid="stFileUploader"] {
    background-color: #fcfcfb !important;
    border: 1px dashed #4f46e5 !important;
    border-radius: 12px !important;
    padding: 16px !important;
    transition: all 0.2s ease !important;
}
div[data-testid="stFileUploader"]:hover {
    background-color: #f8f8f6 !important;
    border-color: #7c3aed !important;
}

/* Diagnostic molecular panels */
.diag-card {
    background: #ffffff;
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 18px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.015);
    border: 1px solid #eeefe9;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
}
.diag-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(0,0,0,0.035);
    border-color: #dcdcd3;
}
.diag-card-basal { border-left: 6px solid #ef4444; }
.diag-card-her { border-left: 6px solid #f59e0b; }
.diag-card-luma { border-left: 6px solid #10b981; }
.diag-card-lumb { border-left: 6px solid #3b82f6; }
.diag-card-normal { border-left: 6px solid #ec4899; }

.diag-title {
    font-family: 'Outfit', sans-serif;
    font-size: 17px;
    font-weight: 800;
    margin-bottom: 8px;
    letter-spacing: 0.3px;
}
.diag-title-basal { color: #dc2626; }
.diag-title-her { color: #d97706; }
.diag-title-luma { color: #059669; }
.diag-title-lumb { color: #2563eb; }
.diag-title-normal { color: #db2777; }

.diag-desc {
    color: #475569;
    font-size: 14px;
    line-height: 1.65;
}
</style>
""",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="main-title">AutoML Pipeline and Clinical Diagnostic Predictor</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Deploy pre-trained models for diagnostic subtype prediction, '
        "or train a custom end-to-end machine learning pipeline.</div>",
        unsafe_allow_html=True,
    )

    t_predict, t_train = st.tabs(["Clinical Subtype Predictor (Inference)", "End-to-End AutoML Training"])

    with t_predict:
        _render_predictor_tab()

    with t_train:
        _render_training_tab()


# =============================================================================
# Session-trained model registry
# =============================================================================
def _trained_predictors():
    return st.session_state.get(TRAINED_PREDICTORS_KEY, {}) or {}


def _is_trained_choice(choice_key):
    return isinstance(choice_key, str) and choice_key.startswith(TRAINED_PREFIX)


def _trained_registry_key(choice_key):
    return choice_key.replace(TRAINED_PREFIX, "", 1)


def _slugify(text):
    safe = []
    for ch in str(text).lower():
        safe.append(ch if ch.isalnum() else "_")
    slug = "".join(safe).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "model"


def _last_estimator(model):
    if hasattr(model, "steps") and model.steps:
        return model.steps[-1][1]
    if hasattr(model, "named_steps") and model.named_steps:
        return list(model.named_steps.values())[-1]
    return model


def _short_model_name(model):
    estimator = _last_estimator(model)
    name = estimator.__class__.__name__
    lookup = {
        "RandomForestClassifier": "RF",
        "ExtraTreesClassifier": "Extra Trees",
        "GradientBoostingClassifier": "Gradient Boosting",
        "HistGradientBoostingClassifier": "Hist Gradient Boosting",
        "XGBClassifier": "XGBoost",
        "LGBMClassifier": "LightGBM",
        "CatBoostClassifier": "CatBoost",
        "MLPClassifier": "MLP",
        "LogisticRegression": "Logistic Regression",
        "SVC": "SVM",
    }
    return lookup.get(name, name.replace("Classifier", ""))


def _looks_like_predictor(obj):
    return obj is not None and callable(getattr(obj, "predict", None))


def _add_model_spec(specs, seen_ids, label, model, source, input_mode):
    if not _looks_like_predictor(model) or id(model) in seen_ids:
        return
    short_name = _short_model_name(model)
    clean_label = str(label or short_name).strip()
    if clean_label.lower() in {"best", "best_pipeline", "pipeline", "model", "best_estimator"}:
        clean_label = short_name
    elif short_name.lower() not in clean_label.lower():
        clean_label = f"{clean_label} ({short_name})"
    specs.append({"label": clean_label, "model": model, "source": source, "input_mode": input_mode})
    seen_ids.add(id(model))


def _collect_trained_models(grid_results, bench_results, session_model_results=None):
    specs = []
    seen_ids = set()

    def scan_grid_mapping(mapping):
        if not isinstance(mapping, dict):
            return

        preferred_keys = ["best_pipeline", "best_estimator", "pipeline", "model"]
        for key in preferred_keys:
            if key in mapping:
                label = mapping.get("best_model_name") or mapping.get("model_name") or key
                _add_model_spec(specs, seen_ids, label, mapping[key], "GridSearchCV", "raw_pipeline")

        for container_key in ["trained_models", "models", "pipelines", "classifiers", "estimators"]:
            container = mapping.get(container_key)
            if isinstance(container, dict):
                for label, model in container.items():
                    _add_model_spec(specs, seen_ids, label, model, "GridSearchCV", "raw_pipeline")

    def scan_benchmark_mapping(mapping):
        if not isinstance(mapping, dict):
            return

        for container_key in ["trained_models", "models", "pipelines", "classifiers", "estimators"]:
            container = mapping.get(container_key)
            if isinstance(container, dict):
                for label, model in container.items():
                    _add_model_spec(specs, seen_ids, label, model, "Benchmark", "consensus_scaled")

    scan_grid_mapping(grid_results)
    scan_benchmark_mapping(bench_results)
    scan_benchmark_mapping(session_model_results)
    return specs


def _store_latest_trained_predictors(
    X,
    y,
    prep_results,
    fs_results,
    bench_results,
    grid_results,
    target_col,
    params,
    session_model_results=None,
):
    model_specs = _collect_trained_models(grid_results, bench_results, session_model_results=session_model_results)
    label_encoder = prep_results.get("label_encoder") if isinstance(prep_results, dict) else None
    top_consensus_genes = fs_results.get("top_consensus_genes") if isinstance(fs_results, dict) else None
    numeric_X = X.apply(pd.to_numeric, errors="coerce")
    feature_columns = list(X.columns)
    feature_medians = numeric_X.median(numeric_only=True)

    registry = {}
    for idx, spec in enumerate(model_specs, start=1):
        key = f"{idx}_{_slugify(spec['label'])}"
        registry[key] = {
            "label": spec["label"],
            "model": spec["model"],
            "source": spec["source"],
            "input_mode": spec["input_mode"],
            "target_col": target_col,
            "feature_columns": feature_columns,
            "feature_medians": feature_medians,
            "label_encoder": label_encoder,
            "variance_selector": prep_results.get("variance_selector") if isinstance(prep_results, dict) else None,
            "scaler": prep_results.get("scaler") if isinstance(prep_results, dict) else None,
            "selected_features": prep_results.get("selected_features") if isinstance(prep_results, dict) else None,
            "top_consensus_genes": top_consensus_genes,
            "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "params": params,
        }

    st.session_state[TRAINED_PREDICTORS_KEY] = registry
    st.session_state[TRAINED_META_KEY] = {
        "target_col": target_col,
        "n_rows": len(X),
        "n_features": len(feature_columns),
        "n_classes": y.nunique() if hasattr(y, "nunique") else len(np.unique(y)),
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_labels": [spec["label"] for spec in registry.values()],
    }
    st.session_state[SELECT_TRAINED_ON_NEXT_RENDER_KEY] = bool(registry)
    return registry


# =============================================================================
# Shared prediction helpers
# =============================================================================
def _read_uploaded_dataframe(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith(".parquet"):
        return pd.read_parquet(uploaded_file)
    return pd.read_csv(uploaded_file)


def _normalize_column_name(name):
    text = str(name).strip().lower()
    for char in [" ", "-", ".", "/", "\\", ":", ";", "(", ")", "[", "]", "{", "}"]:
        text = text.replace(char, "_")
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_")


def _is_probable_identifier_column(column_name, series=None, position=None):
    normalized = _normalize_column_name(column_name)
    if normalized in IDENTIFIER_COLUMN_NAMES:
        return True

    if position == 0 and normalized.startswith("unnamed"):
        return True

    if normalized.endswith("_id") or normalized.endswith("_ids") or normalized.startswith("id_"):
        return True

    if series is None:
        return False

    non_null = series.dropna()
    if len(non_null) == 0:
        return False

    unique_ratio = non_null.nunique(dropna=True) / len(non_null)
    first_column_like_id = position == 0 and unique_ratio >= 0.90
    string_like = pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)

    return bool(first_column_like_id and string_like)


def _drop_identifier_columns(df, target_col=None):
    protected = {target_col} if target_col is not None else set()
    drop_cols = []
    for position, column in enumerate(df.columns):
        if column in protected:
            continue
        if _is_probable_identifier_column(column, df[column], position):
            drop_cols.append(column)
    if not drop_cols:
        return df.copy(), []
    return df.drop(columns=drop_cols).copy(), drop_cols


def _extract_ids_and_numeric_features(raw_df, target_col=None):
    df = raw_df.copy()
    if target_col and target_col in df.columns:
        df = df.drop(columns=[target_col])

    first_col = df.columns[0] if len(df.columns) else None
    first_is_id = len(df.columns) > 0 and _is_probable_identifier_column(first_col, df.iloc[:, 0], 0)

    if first_is_id:
        patient_ids = df.iloc[:, 0].astype(str).tolist()
        X_raw = df.set_index(df.columns[0]).copy()
    else:
        patient_ids = [str(idx) for idx in df.index] if not isinstance(df.index, pd.RangeIndex) else [
            f"Sample_{i + 1}" for i in range(len(df))
        ]
        X_raw = df.copy()

    X_features, _ = _drop_identifier_columns(X_raw)
    X_numeric = X_features.apply(pd.to_numeric, errors="coerce")
    X_numeric = X_numeric.loc[:, X_numeric.notna().any(axis=0)]
    return X_numeric, patient_ids


def _prepare_pretrained_uploaded_features(uploaded_file):
    raw_df = _read_uploaded_dataframe(uploaded_file)

    if raw_df.shape[1] < 150 and raw_df.shape[0] > 10000:
        st.warning("Detected (genes x patients) format. Automatically transposing gene expression matrix.")
        raw_df = raw_df.set_index(raw_df.columns[0]).T
        patient_ids = raw_df.index.astype(str).tolist()
        X_raw = raw_df.apply(pd.to_numeric, errors="coerce")
        X_raw = X_raw.loc[:, X_raw.notna().any(axis=0)]
        return X_raw, patient_ids

    return _extract_ids_and_numeric_features(raw_df)


def _align_to_feature_schema(X_raw, feature_columns, feature_medians):
    mapped_cols = [col for col in X_raw.columns if col in feature_columns]
    missing_cols = [col for col in feature_columns if col not in X_raw.columns]
    X_aligned = X_raw.reindex(columns=feature_columns)
    X_filled = X_aligned.fillna(feature_medians).fillna(0)
    return X_filled, mapped_cols, missing_cols


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, (pd.Index, np.ndarray)):
        return value.tolist()
    return list(value)


def _consensus_feature_indices(selected_features, top_consensus_genes):
    selected_features = _as_list(selected_features)
    top_consensus_genes = _as_list(top_consensus_genes)
    feature_index_map = {feature: idx for idx, feature in enumerate(selected_features)}
    return [feature_index_map[feature] for feature in top_consensus_genes if feature in feature_index_map]


def _subset_consensus_scaled(X_scaled, selected_features, top_consensus_genes):
    consensus_indices = _consensus_feature_indices(selected_features, top_consensus_genes)
    if not consensus_indices:
        raise ValueError("No consensus feature indices could be reconstructed for this trained model.")
    if hasattr(X_scaled, "iloc"):
        return X_scaled.iloc[:, consensus_indices]
    return X_scaled[:, consensus_indices]


def _model_input_for_trained_spec(spec, X_aligned):
    if spec.get("input_mode") != "consensus_scaled":
        return X_aligned

    var_selector = spec.get("variance_selector")
    scaler = spec.get("scaler")
    selected_features = _as_list(spec.get("selected_features"))
    top_consensus_genes = _as_list(spec.get("top_consensus_genes"))

    if var_selector is None or scaler is None or not selected_features or not top_consensus_genes:
        raise ValueError(
            "This benchmark model was trained on scaled consensus features, but the fitted preprocessing "
            "objects were not returned by pipeline_engine.preprocess/run_feature_selection."
        )

    X_var = var_selector.transform(X_aligned)
    X_scaled = scaler.transform(X_var)
    return _subset_consensus_scaled(X_scaled, selected_features, top_consensus_genes)


def _decode_with_label_encoder(values, label_encoder):
    arr = np.asarray(values)
    if label_encoder is not None:
        try:
            if np.issubdtype(arr.dtype, np.integer):
                return label_encoder.inverse_transform(arr.astype(int))
            if np.issubdtype(arr.dtype, np.floating) and np.all(np.isfinite(arr)) and np.all(arr == arr.astype(int)):
                return label_encoder.inverse_transform(arr.astype(int))
        except Exception:
            pass
    return arr.astype(str)


def _model_classes(model, label_encoder, n_prob_cols):
    classes = getattr(model, "classes_", None)
    if classes is None:
        classes = getattr(_last_estimator(model), "classes_", None)
    if classes is None and label_encoder is not None:
        classes = getattr(label_encoder, "classes_", None)
    if classes is None:
        classes = [f"Class {i}" for i in range(n_prob_cols)]
    decoded = list(_decode_with_label_encoder(classes, label_encoder))
    if len(decoded) > n_prob_cols:
        decoded = decoded[:n_prob_cols]
    while len(decoded) < n_prob_cols:
        decoded.append(f"Class {len(decoded)}")
    return decoded


def _predict_with_fitted_model(model, X_aligned, label_encoder=None):
    X_array = X_aligned.to_numpy() if hasattr(X_aligned, "to_numpy") else np.asarray(X_aligned)
    try:
        raw_preds = model.predict(X_aligned)
    except Exception:
        raw_preds = model.predict(X_array)

    probs = None
    if callable(getattr(model, "predict_proba", None)):
        try:
            probs = model.predict_proba(X_aligned)
        except Exception:
            probs = model.predict_proba(X_array)

    pred_labels = _decode_with_label_encoder(raw_preds, label_encoder)
    class_labels = _model_classes(model, label_encoder, probs.shape[1]) if probs is not None else []
    return pred_labels, probs, class_labels


def _train_session_mlp_predictor(prep_results, fs_results, log=None):
    try:
        from sklearn.metrics import accuracy_score
        from sklearn.neural_network import MLPClassifier

        selected_features = prep_results["selected_features"]
        top_consensus_genes = fs_results["top_consensus_genes"]
        X_train = _subset_consensus_scaled(prep_results["X_train_scaled"], selected_features, top_consensus_genes)
        X_test = _subset_consensus_scaled(prep_results["X_test_scaled"], selected_features, top_consensus_genes)
        y_train = prep_results["y_train"]
        y_test = prep_results["y_test"]

        if log:
            log("Training session MLPClassifier on consensus features...")

        mlp = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=350,
            random_state=42,
            early_stopping=False,
        )
        mlp.fit(X_train, y_train)

        test_accuracy = accuracy_score(y_test, mlp.predict(X_test)) if len(y_test) else np.nan
        if log:
            log(f"Session MLPClassifier ready. Holdout accuracy: {test_accuracy:.2%}")

        return {
            "models": {"MLP": mlp},
            "metrics": {"MLP": {"test_accuracy": test_accuracy}},
        }
    except Exception as exc:
        if log:
            log(f"Warning: session MLP training failed: {exc}")
        return {"models": {}, "metrics": {}, "error": str(exc)}


def _format_percent_or_na(value):
    if pd.isna(value):
        return "N/A"
    return f"{value:.2%}"


def _prediction_color_map(labels):
    subtype_matches = {label: SUBTYPE_COLORS[label] for label in labels if label in SUBTYPE_COLORS}
    if len(subtype_matches) == len(labels):
        return subtype_matches

    palette = px.colors.qualitative.Safe + px.colors.qualitative.Set2
    color_map = {}
    for idx, label in enumerate(labels):
        color_map[label] = SUBTYPE_COLORS.get(label, palette[idx % len(palette)])
    return color_map


def _render_prediction_outputs(
    results_df,
    label_col,
    confidence_col,
    model_label,
    mapped_feature_count,
    total_feature_count,
    feature_vocab_label,
    csv_filename,
    clinical_interpretation=False,
):
    st.markdown("<div class='custom-hr'></div>", unsafe_allow_html=True)
    st.markdown(
        (
            '<div class="success-box"><b>Prediction Complete!</b> Mapped and processed '
            f"<b>{mapped_feature_count:,} out of {total_feature_count:,} {feature_vocab_label}</b> successfully.</div>"
        ),
        unsafe_allow_html=True,
    )

    mean_conf = results_df[confidence_col].dropna().mean()
    mean_conf_label = f"{mean_conf:.2%}" if not pd.isna(mean_conf) else "N/A"

    c_stats = st.columns(3)
    with c_stats[0]:
        st.markdown(custom_card(str(len(results_df)), "Samples Evaluated"), unsafe_allow_html=True)
    with c_stats[1]:
        st.markdown(custom_card(mean_conf_label, "Mean Prediction Confidence"), unsafe_allow_html=True)
    with c_stats[2]:
        st.markdown(custom_card(model_label, "Classifier Utilized", True), unsafe_allow_html=True)

    col_chart, col_stats = st.columns([3, 2])
    dist = results_df[label_col].value_counts().reset_index()
    dist.columns = [label_col, "Count"]
    dist[label_col] = dist[label_col].astype(str)
    color_map = _prediction_color_map(dist[label_col].tolist())

    with col_chart:
        fig = px.bar(
            dist,
            x=label_col,
            y="Count",
            color=label_col,
            color_discrete_map=color_map,
            title="Predicted Class Distribution",
            template="plotly_white",
            text="Count",
        )
        fig.update_traces(textposition="outside", marker_line_width=0)
        fig.update_layout(**PLOTLY_LAYOUT, showlegend=False)
        fig.update_xaxes(gridcolor=GRID_COLOR)
        fig.update_yaxes(gridcolor=GRID_COLOR)
        st.plotly_chart(fig, use_container_width=True)

    with col_stats:
        st.markdown("#### Prediction Summary Statistics")
        st.markdown("Below is the count of predicted classes in your dataset:")
        for _, row in dist.iterrows():
            st.write(f"- **{row[label_col]}:** {row['Count']} sample(s)")

    st.markdown("#### Sample-Level Predictions")
    st.markdown("Filter predictions by predicted class or search by sample ID below:")

    sf_col1, sf_col2 = st.columns([1, 2])
    with sf_col1:
        filter_options = ["All Classes"] + sorted(results_df[label_col].astype(str).unique().tolist())
        class_filter = st.selectbox("Filter by Predicted Class", filter_options)
    with sf_col2:
        search_query = st.text_input("Search Sample ID", placeholder="Enter sample ID...")

    filtered_df = results_df.copy()
    if class_filter != "All Classes":
        filtered_df = filtered_df[filtered_df[label_col].astype(str) == class_filter]
    if search_query:
        filtered_df = filtered_df[
            filtered_df["Patient ID"].astype(str).str.contains(search_query, case=False, regex=False)
        ]

    display_df = filtered_df.copy()
    display_df[confidence_col] = display_df[confidence_col].apply(_format_percent_or_na)
    for col in display_df.columns:
        if col.startswith("Prob "):
            display_df[col] = display_df[col].apply(_format_percent_or_na)

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    csv_data = results_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Full Prediction Report (.csv)",
        csv_data,
        csv_filename,
        "text/csv",
        key=f"download-{csv_filename}",
    )

    if clinical_interpretation:
        _render_clinical_interpretation()


def _render_clinical_interpretation():
    st.markdown("<div class='custom-hr'></div>", unsafe_allow_html=True)
    st.markdown("### Clinical Interpretation of Detected Subtypes")
    st.markdown(
        """
        The molecular subtypes identified in the analyzed cohort represent distinct transcriptomic profiles
        with established clinical, pathological, and therapeutic implications:
        """
    )

    st.markdown(
        """
        <div class="diag-card diag-card-basal">
            <div class="diag-title diag-title-basal">Basal-like (basal)</div>
            <div class="diag-desc">Typically corresponds to Triple-Negative Breast Cancer (TNBC). Characterized by high proliferation indices and the absence of hormone receptors and HER2 amplification. These tumors do not respond to endocrine therapy or Herceptin, and are clinically managed using aggressive systemic chemotherapy and emerging immunotherapies.</div>
        </div>
        <div class="diag-card diag-card-her">
            <div class="diag-title diag-title-her">HER2-Enriched (HER)</div>
            <div class="diag-desc">Driven primarily by the amplification and over-expression of the <i>ERBB2</i> (<i>HER2</i>) gene on chromosome 17q. These tumors exhibit aggressive behavior but are highly responsive to anti-HER2 targeted monoclonal antibodies such as Trastuzumab (Herceptin) and Pertuzumab.</div>
        </div>
        <div class="diag-card diag-card-luma">
            <div class="diag-title diag-title-luma">Luminal A (luminal_A)</div>
            <div class="diag-desc">The most common and low-grade molecular subtype. Characterized by high expression of estrogen (<i>ESR1</i>) and progesterone receptors, low proliferation markers (e.g., Ki-67), and slow growth. These patients have a favorable prognosis and are highly responsive to hormonal/endocrine therapies (e.g., Tamoxifen, Aromatase inhibitors).</div>
        </div>
        <div class="diag-card diag-card-lumb">
            <div class="diag-title diag-title-lumb">Luminal B (luminal_B)</div>
            <div class="diag-desc">Expresses hormone receptors but exhibits higher growth rates, higher cellular proliferation markers, and occasionally co-amplification of HER2. These tumors have a more guarded prognosis than Luminal A and often require a combination of chemotherapy and endocrine therapy.</div>
        </div>
        <div class="diag-card diag-card-normal">
            <div class="diag-title diag-title-normal">Normal-like (normal)</div>
            <div class="diag-desc">A rare molecular subtype showing expression profiles similar to non-tumor, healthy breast epithelial cells. They are typically managed similarly to luminal tumors but require careful pathological review to rule out stromal contamination.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# TAB 1: CLINICAL SUBTYPE PREDICTOR / SESSION MODEL INFERENCE
# =============================================================================
def _render_predictor_tab():
    st.markdown("### Clinical Subtype Predictor")
    st.markdown(
        '<div class="info-box">Upload a feature matrix to predict samples using pre-trained clinical '
        "artifacts or the latest model trained in this Streamlit session.</div>",
        unsafe_allow_html=True,
    )

    trained_specs = _trained_predictors()
    trained_meta = st.session_state.get(TRAINED_META_KEY, {}) or {}
    if trained_specs:
        model_names = ", ".join(spec["label"] for spec in trained_specs.values())
        st.markdown(
            (
                '<div class="success-box"><b>Latest training session detected.</b> '
                f"Target: <b>{html.escape(str(trained_meta.get('target_col', 'unknown')))}</b>; "
                f"features: <b>{trained_meta.get('n_features', 'N/A')}</b>; "
                f"available model(s): <b>{html.escape(model_names)}</b>.</div>"
            ),
            unsafe_allow_html=True,
        )

    if st.session_state.pop(SELECT_TRAINED_ON_NEXT_RENDER_KEY, False) and trained_specs:
        latest_key = list(trained_specs.keys())[0]
        st.session_state["pred_classifier_key_v2"] = f"{TRAINED_PREFIX}{latest_key}"

    classifier_labels = {
        PRETRAINED_RF_KEY: "Tuned Random Forest (CV Stability: 97.01% +/- 4.81%)",
        PRETRAINED_MLP_KEY: "PyTorch Multi-Layer Perceptron (Test Accuracy: 100.00%)",
    }
    for key, spec in trained_specs.items():
        classifier_labels[f"{TRAINED_PREFIX}{key}"] = f"Newly Trained Pipeline ({spec['label']})"

    classifier_keys = list(classifier_labels.keys())
    if st.session_state.get("pred_classifier_key_v2") not in classifier_keys:
        st.session_state["pred_classifier_key_v2"] = PRETRAINED_RF_KEY

    with st.container(border=True):
        col1, col2 = st.columns(2)
        with col1:
            classifier_choice = st.radio(
                "Select Diagnostic Classifier",
                classifier_keys,
                format_func=lambda key: classifier_labels[key],
                key="pred_classifier_key_v2",
            )

        if _is_trained_choice(classifier_choice):
            source_options = [LATEST_TRAINING_SOURCE, CUSTOM_SOURCE]
            source_help = "Session-trained models use the feature schema from the latest completed training run."
        else:
            source_options = [INTERNAL_SOURCE, CUSTOM_SOURCE]
            source_help = "Pre-trained clinical models use the built-in breast cancer feature schema."

        if st.session_state.get("pred_source_v2") not in source_options:
            st.session_state["pred_source_v2"] = source_options[0]

        with col2:
            data_source = st.radio(
                "Select Evaluation Dataset Source",
                source_options,
                key="pred_source_v2",
                help=source_help,
            )

        uploaded_file = None
        if data_source == CUSTOM_SOURCE:
            uploaded_file = st.file_uploader(
                "Upload prediction dataset",
                type=["csv", "parquet"],
                key="pred_file_uploader",
            )

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        if st.button("Run Prediction", key="btn_run_prediction", use_container_width=True):
            with st.spinner("Analyzing features and generating predictions..."):
                _execute_prediction(data_source, uploaded_file, classifier_choice, classifier_labels[classifier_choice])


def _execute_prediction(data_source, uploaded_file, classifier_choice, classifier_label):
    if _is_trained_choice(classifier_choice):
        _execute_session_trained_prediction(data_source, uploaded_file, classifier_choice, classifier_label)
    else:
        _execute_pretrained_prediction(data_source, uploaded_file, classifier_choice, classifier_label)


def _execute_pretrained_prediction(data_source, uploaded_file, classifier_choice, classifier_label):
    base_dir = Path(__file__).resolve().parent
    artifact_dir = base_dir / "data" / "artifacts"
    processed_dir = base_dir / "data" / "processed"

    try:
        is_mlp = classifier_choice == PRETRAINED_MLP_KEY

        if is_mlp:
            mlp_path = artifact_dir / "mlp_best.pt"
            if not mlp_path.exists():
                st.error("Classifier file mlp_best.pt not found in data/artifacts/. Run deep learning first.")
                return

            import torch
            import torch.nn as nn

            class BreastCancerMLP(nn.Module):
                def __init__(self, in_dim, n_cls):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(in_dim, 512),
                        nn.BatchNorm1d(512),
                        nn.ReLU(),
                        nn.Dropout(0.4),
                        nn.Linear(512, 256),
                        nn.BatchNorm1d(256),
                        nn.ReLU(),
                        nn.Dropout(0.3),
                        nn.Linear(256, 128),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(128, n_cls),
                    )

                def forward(self, x):
                    return self.net(x)

            model = BreastCancerMLP(in_dim=1480, n_cls=5)
            model.load_state_dict(torch.load(mlp_path, map_location="cpu"))
            model.eval()
        else:
            tuned_model_file = "tuned_rf.pkl"
            tuned_model_path = artifact_dir / tuned_model_file
            if not tuned_model_path.exists():
                st.error(f"Classifier file {tuned_model_file} not found in data/artifacts/. Run notebook/AutoML first.")
                return

            model = joblib.load(tuned_model_path)

        scaler = joblib.load(artifact_dir / "scaler.pkl")
        var_sel = joblib.load(artifact_dir / "variance_selector.pkl")
        consensus_genes = joblib.load(artifact_dir / "top_consensus_genes.pkl")
        le = joblib.load(artifact_dir / "label_encoder.pkl")

        internal_cohort_path = processed_dir / "breast_cancer.parquet"
        if not internal_cohort_path.exists():
            st.error("Internal clinical dataset breast_cancer.parquet not found in data/processed/.")
            return
        train_df = pd.read_parquet(internal_cohort_path)
        train_X = train_df.drop(columns=["type"])
        train_medians = train_X.apply(pd.to_numeric, errors="coerce").median(numeric_only=True)

        if data_source == INTERNAL_SOURCE:
            st.info("Loading internal patient clinical cohort (GSE45827)...")
            X_raw = train_X.copy()
            patient_ids = X_raw.index.astype(str).tolist()
        else:
            if uploaded_file is None:
                st.warning("Please upload a CSV or Parquet file to proceed.")
                return

            st.info(f"Ingesting custom dataset: {uploaded_file.name}...")
            X_raw, patient_ids = _prepare_pretrained_uploaded_features(uploaded_file)

        X_new_filled, mapped_cols, _ = _align_to_feature_schema(X_raw, list(train_X.columns), train_medians)

        if len(mapped_cols) == 0:
            st.error(
                "The uploaded dataset does not share any Affymetrix probe IDs/features with the clinical "
                "training cohort. Please verify that the file contains probes such as '1007_s_at' or ERBB2."
            )
            return

        X_new_var = var_sel.transform(X_new_filled)
        X_new_scaled = scaler.transform(X_new_var)

        selected_features = train_X.columns[var_sel.get_support()]
        feature_index_map = {gene: i for i, gene in enumerate(selected_features)}
        consensus_indices = [feature_index_map[gene] for gene in consensus_genes if gene in feature_index_map]
        X_new_cons = X_new_scaled[:, consensus_indices]

        if is_mlp:
            X_tensor = torch.FloatTensor(X_new_cons)
            with torch.no_grad():
                logits = model(X_tensor)
                probs = torch.softmax(logits, dim=1).numpy()
                preds_encoded = np.argmax(probs, axis=1)
        else:
            preds_encoded = model.predict(X_new_cons)
            probs = model.predict_proba(X_new_cons)

        pred_labels = le.inverse_transform(preds_encoded)
        class_labels = list(le.classes_)
        max_probs = np.max(probs, axis=1)

        results_df = pd.DataFrame(
            {
                "Patient ID": patient_ids,
                "Predicted Subtype": pred_labels,
                "Diagnostic Confidence": max_probs,
            }
        )
        for idx, cls in enumerate(class_labels):
            results_df[f"Prob {cls}"] = probs[:, idx]

        _render_prediction_outputs(
            results_df=results_df,
            label_col="Predicted Subtype",
            confidence_col="Diagnostic Confidence",
            model_label="Multi-Layer Perceptron" if is_mlp else "Random Forest",
            mapped_feature_count=len(mapped_cols),
            total_feature_count=len(train_X.columns),
            feature_vocab_label="clinical features",
            csv_filename="subtype_predictions_report.csv",
            clinical_interpretation=True,
        )

    except Exception as e:
        st.error(f"An error occurred during prediction parsing: {e}")
        with st.expander("Diagnostic Traceback"):
            st.code(traceback.format_exc())


def _execute_session_trained_prediction(data_source, uploaded_file, classifier_choice, classifier_label):
    try:
        registry_key = _trained_registry_key(classifier_choice)
        spec = _trained_predictors().get(registry_key)
        if spec is None:
            st.error("The latest trained pipeline could not be found in session state. Please rerun training.")
            return

        target_col = spec.get("target_col")
        feature_columns = spec["feature_columns"]
        feature_medians = spec["feature_medians"]
        model = spec["model"]
        label_encoder = spec.get("label_encoder")

        if data_source == LATEST_TRAINING_SOURCE:
            aml_data = st.session_state.get("aml_data") or {}
            X_raw = aml_data.get("X")
            if X_raw is None:
                st.error("The latest training dataset is no longer available in session state. Please rerun training.")
                return
            X_raw = X_raw.copy()
            patient_ids = X_raw.index.astype(str).tolist()
            st.info("Using the latest dataset retained from the training tab.")
        else:
            if uploaded_file is None:
                st.warning("Please upload a CSV or Parquet file to proceed.")
                return
            st.info(f"Ingesting custom prediction dataset: {uploaded_file.name}...")
            raw_df = _read_uploaded_dataframe(uploaded_file)
            X_raw, patient_ids = _extract_ids_and_numeric_features(raw_df, target_col=target_col)

        X_aligned, mapped_cols, _ = _align_to_feature_schema(X_raw, feature_columns, feature_medians)
        if len(mapped_cols) == 0:
            st.error(
                "The selected prediction dataset does not share any features with the latest trained pipeline. "
                "Upload a dataset with the same feature columns used during training."
            )
            return

        model_input = _model_input_for_trained_spec(spec, X_aligned)
        pred_labels, probs, class_labels = _predict_with_fitted_model(model, model_input, label_encoder=label_encoder)

        confidence_col = "Prediction Confidence"
        label_col = f"Predicted {target_col}" if target_col else "Predicted Label"
        if probs is not None:
            max_probs = np.max(probs, axis=1)
        else:
            max_probs = np.full(len(pred_labels), np.nan)

        results_df = pd.DataFrame(
            {
                "Patient ID": patient_ids,
                label_col: pred_labels,
                confidence_col: max_probs,
            }
        )

        if probs is not None:
            for idx, cls in enumerate(class_labels):
                results_df[f"Prob {cls}"] = probs[:, idx]

        _render_prediction_outputs(
            results_df=results_df,
            label_col=label_col,
            confidence_col=confidence_col,
            model_label=spec.get("label", classifier_label),
            mapped_feature_count=len(mapped_cols),
            total_feature_count=len(feature_columns),
            feature_vocab_label="trained-session features",
            csv_filename="latest_trained_pipeline_predictions.csv",
            clinical_interpretation=False,
        )

    except Exception as e:
        st.error(f"An error occurred while using the latest trained pipeline: {e}")
        with st.expander("Diagnostic Traceback"):
            st.code(traceback.format_exc())


# =============================================================================
# TAB 2: END-TO-END AUTOML TRAINING PIPELINE (SINGLE-CLICK RUNNER)
# =============================================================================
def _render_training_tab():
    st.markdown("### End-to-End AutoML Training")
    st.markdown(
        '<div class="info-box">Upload a training dataset (CSV/Parquet) and click one button to execute '
        "the entire computational biology machine learning pipeline. You can follow live progress in the "
        "terminal console below.</div>",
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        uploaded_file = st.file_uploader("Upload custom training dataset", type=["csv", "parquet"], key="train_file_uploader")

        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith(".parquet"):
                    raw_df = pd.read_parquet(uploaded_file)
                else:
                    raw_df = pd.read_csv(uploaded_file)

                st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
                st.markdown("#### Dataset Preview")
                st.dataframe(raw_df.head(10), use_container_width=True, hide_index=True)

                target_col = st.selectbox(
                    "Select Target Label Column",
                    raw_df.columns.tolist(),
                    index=len(raw_df.columns) - 1,
                    key="train_target_col",
                )

                _, dropped_id_cols = _drop_identifier_columns(raw_df, target_col=target_col)
                if dropped_id_cols:
                    shown_cols = ", ".join(str(col) for col in dropped_id_cols[:8])
                    suffix = "" if len(dropped_id_cols) <= 8 else f", and {len(dropped_id_cols) - 8} more"
                    st.info(f"Identifier/non-feature columns will be excluded from training: {shown_cols}{suffix}.")

                st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
                st.markdown("#### Hyperparameter and Preprocessing Tuning Options")

                c_inputs = st.columns(3)
                with c_inputs[0]:
                    var_thresh = st.number_input(
                        "Variance Filter Threshold",
                        value=0.1,
                        min_value=0.0,
                        step=0.01,
                        key="train_var_t",
                    )
                with c_inputs[1]:
                    test_size = st.slider("Evaluation Test Size", 0.1, 0.4, 0.2, 0.05, key="train_test_s")
                with c_inputs[2]:
                    top_k = st.number_input("Consensus Top-K features", value=250, min_value=10, key="train_top_k")

                st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
                if st.button("Start Complete End-to-End AutoML", key="btn_run_automl", use_container_width=True):
                    _execute_automl_pipeline(raw_df, target_col, var_thresh, test_size, top_k)

            except Exception as e:
                st.error(f"Error parsing training file: {e}")
        else:
            st.info("Please upload a CSV or Parquet training dataset to initialize the custom pipeline.")


def _execute_automl_pipeline(df, target_col, var_threshold, test_size, top_k):
    progress_bar = st.progress(0.0)
    status_text = st.empty()
    console_ph = st.empty()
    console = LiveConsole(console_ph)

    try:
        progress_bar.progress(0.1)
        status_text.write("Step 1 of 9: Loading and validating raw data...")
        training_df, dropped_id_cols = _drop_identifier_columns(df, target_col=target_col)
        if dropped_id_cols:
            console.log("Excluded identifier/non-feature columns: " + ", ".join(str(col) for col in dropped_id_cols))
        X, y, report = pe.load_and_validate(training_df, target_col, log=console.log)
        time.sleep(0.5)

        progress_bar.progress(0.25)
        status_text.write("Step 2 of 9: Performing latent dimensionality reduction (PCA)...")
        eda_results = pe.run_eda(X, y, log=console.log)
        time.sleep(0.5)

        progress_bar.progress(0.40)
        status_text.write("Step 3 of 9: Preprocessing, variance filtering, and scaling...")
        prep_results = pe.preprocess(X, y, var_threshold=var_threshold, test_size=test_size, log=console.log)
        time.sleep(0.5)

        progress_bar.progress(0.55)
        status_text.write("Step 4 of 9: Running 4-model consensus feature selection (ANOVA, MI, RF, LASSO)...")
        fs_results = pe.run_feature_selection(
            prep_results["X_train_scaled"],
            prep_results["y_train"],
            prep_results["selected_features"],
            top_k=top_k,
            log=console.log,
        )
        time.sleep(0.5)

        progress_bar.progress(0.64)
        status_text.write("Step 5 of 9: Training neural-network classifier for predictor reuse...")
        session_mlp_results = _train_session_mlp_predictor(prep_results, fs_results, log=console.log)
        time.sleep(0.5)

        progress_bar.progress(0.70)
        status_text.write("Step 6 of 9: Benchmarking classical machine learning classifiers...")
        bench_results = pe.run_baseline_benchmark(
            prep_results["X_train_scaled"],
            prep_results["X_test_scaled"],
            prep_results["y_train"],
            prep_results["y_test"],
            prep_results["selected_features"],
            fs_results["top_consensus_genes"],
            log=console.log,
        )
        time.sleep(0.5)

        progress_bar.progress(0.80)
        status_text.write("Step 7 of 9: Calculating K-Fold Cross-Validation model stability...")
        encoded_y = prep_results["label_encoder"].transform(y)
        cv_results, best_cv_name = pe.run_cross_validation(X, encoded_y, k_features=top_k, log=console.log)
        time.sleep(0.5)

        progress_bar.progress(0.90)
        status_text.write("Step 8 of 9: Tuning optimal model hyperparameters...")
        grid_results = pe.run_gridsearch(X, encoded_y, log=console.log)
        time.sleep(0.5)

        progress_bar.progress(0.98)
        status_text.write("Step 9 of 9: Running TreeSHAP explanations...")
        shap_results = pe.run_shap_analysis(grid_results["best_pipeline"], X, top_n=25, log=console.log)
        time.sleep(0.5)

        progress_bar.progress(1.0)
        status_text.write("AutoML Training Successfully Completed!")
        console.finish(success=True)

        st.session_state.aml_data = {
            "X": X,
            "y": y,
            "report": report,
            "target_col": target_col,
            "dropped_identifier_columns": dropped_id_cols,
        }
        st.session_state.aml_eda = eda_results
        st.session_state.aml_prep = prep_results
        st.session_state.aml_fs = fs_results
        st.session_state.aml_bench = bench_results
        st.session_state.aml_session_mlp = session_mlp_results
        st.session_state.aml_cv = (cv_results, best_cv_name)
        st.session_state.aml_grid = grid_results
        st.session_state.aml_shap = shap_results

        latest_registry = _store_latest_trained_predictors(
            X=X,
            y=y,
            prep_results=prep_results,
            fs_results=fs_results,
            bench_results=bench_results,
            grid_results=grid_results,
            target_col=target_col,
            params={
                "var_threshold": var_threshold,
                "test_size": test_size,
                "top_k": top_k,
                "dropped_identifier_columns": dropped_id_cols,
                "session_mlp_metrics": session_mlp_results.get("metrics", {}),
            },
            session_model_results=session_mlp_results,
        )

        if latest_registry:
            st.success(
                "All steps completed successfully. The latest trained pipeline is now available in the "
                "predict tab and will stay available while this Streamlit session is active."
            )
        else:
            st.warning(
                "Training completed, but no fitted sklearn-style predictor was returned by pipeline_engine. "
                "Return best_pipeline, best_estimator, or a models/pipelines dictionary to enable inference."
            )

        time.sleep(1.0)
        st.rerun()

    except Exception as e:
        progress_bar.progress(1.0)
        status_text.write("Pipeline failed due to a mathematical/parsing error.")
        console.log(f"ERROR: {e}")
        console.finish(success=False)
        st.error(f"An error occurred during pipeline execution: {e}")
        with st.expander("Full Traceback"):
            st.code(traceback.format_exc())

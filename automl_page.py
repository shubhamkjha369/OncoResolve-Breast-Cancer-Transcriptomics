"""
automl_page.py - Premium Clinical Subtype Predictor & Benchmarking Interface
=============================================================================
Deploys locked v3.3 pre-trained models (Logistic Regression + Support Vector Machine)
on user-uploaded transcriptomics datasets.
"""
import time
import html
import traceback
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# =============================================================================
# CONSTANTS & CONFIG
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "data" / "artifacts"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

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
    "her2": "#f59e0b",
    "luminal_A": "#10b981",
    "luminal_B": "#3b82f6",
    "normal": "#ec4899",
}

PRETRAINED_LR_KEY = "lr_model"
PRETRAINED_SVM_KEY = "svm_model"

# Ground-truth mapping options for benchmarking
SUBTYPE_MAPPINGS = {
    "Basal": "basal", "basal": "basal", "TNBC": "basal",
    "Her2": "her2", "HER2": "her2", "HER": "her2", "her2": "her2",
    "LumA": "luminal_A", "Luminal A": "luminal_A", "luminal_A": "luminal_A", "luminal-a": "luminal_A",
    "LumB": "luminal_B", "Luminal B": "luminal_B", "luminal_B": "luminal_B", "luminal-b": "luminal_B",
    "Normal": "normal", "normal": "normal", "Normal-like": "normal",
}

# =============================================================================
# COMPONENT STYLING & HELPERS
# =============================================================================
def custom_card(val, label, accent=False):
    cls = "accent-card" if accent else "metric-card"
    return f'<div class="{cls}"><div class="metric-value">{val}</div><div class="metric-label">{label}</div></div>'

def _prediction_color_map(labels):
    subtype_matches = {label: SUBTYPE_COLORS[label] for label in labels if label in SUBTYPE_COLORS}
    if len(subtype_matches) == len(labels):
        return subtype_matches

    palette = px.colors.qualitative.Safe + px.colors.qualitative.Set2
    color_map = {}
    for idx, label in enumerate(labels):
        color_map[label] = SUBTYPE_COLORS.get(label.lower(), palette[idx % len(palette)])
    return color_map

# =============================================================================
# DATA PREPARATION & ALIGNMENT
# =============================================================================
def _load_and_align_features(df, top_deg_genes, entrez_to_hugo):
    """
    Transposes, filters, and aligns the uploaded dataframe to match the 178
    consensus biomarker signature, applying log2 scaling if raw counts are detected.
    """
    raw_df = df.copy()

    # Determine index and columns
    # If the first column is string-like/identifiers, set it as index
    first_col = raw_df.columns[0]
    first_col_series = raw_df[first_col]
    if pd.api.types.is_object_dtype(first_col_series) or pd.api.types.is_string_dtype(first_col_series):
        raw_df.set_index(first_col, inplace=True)

    # Detect genes x samples format (e.g. if >10000 rows and <500 columns) and transpose
    if raw_df.shape[0] > 500 and raw_df.shape[1] < 500:
        st.warning("Detected (genes × patients) orientation. Automatically transposing gene expression matrix.")
        raw_df = raw_df.T

    patient_ids = [str(x) for x in raw_df.index]

    # Clean whitespace from column names
    raw_df.columns = raw_df.columns.astype(str).str.strip()

    # Pre-allocate array for aligned feature values
    correct_genes_order = sorted([str(g) for g in top_deg_genes])
    X_aligned = np.zeros((raw_df.shape[0], len(correct_genes_order)))
    mapped_features = []
    missing_features = []

    # Map TCGA Entrez IDs to HUGO symbols for alignment
    for idx, entrez_id in enumerate(correct_genes_order):
        hugo_symbol = entrez_to_hugo.get(str(entrez_id), None)
        if hugo_symbol is not None and hugo_symbol in raw_df.columns:
            val = raw_df[hugo_symbol].astype(float).fillna(0.0).values
            # Dynamic log2 scaling for raw counts / RSEM values
            if val.max() > 50:
                val = np.log2(np.clip(val, 0, None) + 1.0)
            X_aligned[:, idx] = val
            mapped_features.append(hugo_symbol)
        else:
            # Check if Entrez ID itself is in raw_df columns
            if str(entrez_id) in raw_df.columns:
                val = raw_df[str(entrez_id)].astype(float).fillna(0.0).values
                if val.max() > 50:
                    val = np.log2(np.clip(val, 0, None) + 1.0)
                X_aligned[:, idx] = val
                mapped_features.append(str(entrez_id))
            else:
                missing_features.append(hugo_symbol if hugo_symbol else str(entrez_id))

    return X_aligned, patient_ids, mapped_features, missing_features

# =============================================================================
# RENDERING INTERFACE
# =============================================================================
def render(card_fn=None):
    # Inject premium styles
    st.markdown(
        """
<style>
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

/* File Uploader styling */
div[data-testid="stFileUploader"] {
    background-color: #fcfcfb !important;
    border: 1px dashed #4f46e5 !important;
    border-radius: 12px !important;
    padding: 16px !important;
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
.diag-card-norm { border-left: 6px solid #ec4899; }

.diag-title {
    font-family: 'Outfit', sans-serif;
    font-size: 17px;
    font-weight: 800;
    margin-bottom: 8px;
}
.diag-title-basal { color: #dc2626; }
.diag-title-her { color: #d97706; }
.diag-title-luma { color: #059669; }
.diag-title-lumb { color: #2563eb; }
.diag-title-norm { color: #db2777; }

.diag-desc {
    color: #475569;
    font-size: 14px;
    line-height: 1.65;
}
</style>
""",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="main-title">Clinical Subtype Predictor & <span class="main-title-accent">Benchmark Tool</span></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Evaluate locked v3.3 pre-trained breast cancer transcriptomic classifiers on custom cohort datasets.</div>',
        unsafe_allow_html=True,
    )

    # 1. LOAD ARTIFACTS
    try:
        lr_pipeline = joblib.load(ARTIFACT_DIR / "logistic_regression_model.pkl")
        svm_pipeline = joblib.load(ARTIFACT_DIR / "SVM_probability.pkl")
        top_deg_genes = joblib.load(ARTIFACT_DIR / "top_deg_genes.pkl")
        entrez_to_hugo = joblib.load(ARTIFACT_DIR / "tcga_entrez_to_hugo.pkl")
        label_encoder = joblib.load(ARTIFACT_DIR / "label_encoder_cohort.pkl")
    except Exception as exc:
        st.error(f"Critical Error: Failed to load pre-trained OncoResolve v3.3 model artifacts: {exc}")
        return

    # Render Prediction Setup Form
    with st.container(border=True):
        st.markdown("### 🧬 Diagnostic Predictor Configurations")
        
        col_model, col_scaling = st.columns(2)
        with col_model:
            model_choice = st.selectbox(
                "Select Pre-trained Model",
                [PRETRAINED_LR_KEY, PRETRAINED_SVM_KEY],
                format_func=lambda x: "Logistic Regression (Linear Classifier — Holdout ACC: 91.37%)" if x == PRETRAINED_LR_KEY else "Support Vector Machine (RBF Classifier — Holdout ACC: 90.36%)"
            )
        with col_scaling:
            scaling_choice = st.selectbox(
                "Select Standardization Protocol",
                ["Independent Cohort Scaling (Recommended)", "Discovery Cohort Scaling (N-of-1)"],
                help="Independent scaling fits a new StandardScaler to the uploaded dataset. This is highly recommended for cohort-level external datasets to bridge platform shifts. Discovery scaling uses the static TCGA-BRCA training statistics (ideal for single-sample/N-of-1 predictions)."
            )

        # Upload file
        uploaded_file = st.file_uploader(
            "Upload Gene Expression Dataset (CSV or Parquet)",
            type=["csv", "parquet"],
            help="The dataset must contain HUGO gene symbols (e.g., ESR1, ERBB2, MKI67, KRT5) either as columns (patients as rows) or rows (transposed genes)."
        )

        # Ground truth column selection (optional for benchmarking)
        ground_truth_col = None
        if uploaded_file is not None:
            try:
                # Read header
                if uploaded_file.name.lower().endswith(".parquet"):
                    preview_df = pd.read_parquet(uploaded_file, columns=None)
                else:
                    preview_df = pd.read_csv(uploaded_file, nrows=5)

                col_options = ["None (Unsupervised / Pure Prediction)"] + preview_df.columns.tolist()
                st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
                selected_col = st.selectbox(
                    "Ground-Truth Subtype Column (Optional — select to evaluate accuracy and F1)",
                    col_options
                )
                if selected_col != "None (Unsupervised / Pure Prediction)":
                    ground_truth_col = selected_col
            except Exception:
                pass

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        run_btn = st.button("Execute Subtyping Prediction", key="btn_run_subtyping", use_container_width=True)

    # 2. RUN INFERENCE
    if run_btn:
        if uploaded_file is None:
            st.warning("Please upload a transcriptomics dataset to run predictions.")
            return

        with st.spinner("Aligning transcriptomes, normalizing expressions, and generating clinical profiles..."):
            try:
                # Load uploaded data
                if uploaded_file.name.lower().endswith(".parquet"):
                    raw_df = pd.read_parquet(uploaded_file)
                else:
                    raw_df = pd.read_csv(uploaded_file)

                # Separate ground truth if provided
                y_true_raw = None
                if ground_truth_col and ground_truth_col in raw_df.columns:
                    y_true_raw = raw_df[ground_truth_col].copy()

                # Align features to 178 signature
                X_aligned, patient_ids, mapped_features, missing_features = _load_and_align_features(
                    raw_df, top_deg_genes, entrez_to_hugo
                )

                if len(mapped_features) == 0:
                    st.error("Error: The uploaded dataset does not share any mapped HUGO symbols with the 178 consensus biomarker signature. Check your column labels.")
                    return

                # Choose model pipeline
                pipeline = lr_pipeline if model_choice == PRETRAINED_LR_KEY else svm_pipeline
                scaler_fitted = pipeline.named_steps["scaler"]
                clf = pipeline.named_steps["clf"]

                # Apply standardization protocol
                if scaling_choice.startswith("Independent"):
                    if X_aligned.shape[0] < 2:
                        st.warning("Independent Z-scaling requires at least 2 samples. Falling back to pre-trained TCGA Discovery scaling.")
                        X_scaled = scaler_fitted.transform(X_aligned)
                    else:
                        from sklearn.preprocessing import StandardScaler
                        scaler_new = StandardScaler()
                        X_scaled = scaler_new.fit_transform(X_aligned)
                else:
                    # Discovery scale
                    X_scaled = scaler_fitted.transform(X_aligned)

                X_scaled = np.nan_to_num(X_scaled, nan=0.0)

                # Generate predictions and probabilities
                preds_encoded = clf.predict(X_scaled)
                probs = clf.predict_proba(X_scaled)

                pred_labels = label_encoder.inverse_transform(preds_encoded)
                class_labels = list(label_encoder.classes_)
                max_probs = np.max(probs, axis=1)

                # Prepare results dataframe
                results_df = pd.DataFrame({
                    "Patient ID": patient_ids,
                    "Predicted Subtype": pred_labels,
                    "Prediction Confidence": max_probs
                })
                for idx, cls in enumerate(class_labels):
                    results_df[f"Prob {cls}"] = probs[:, idx]

                # Render metrics
                st.markdown("<div class='custom-hr'></div>", unsafe_allow_html=True)
                st.markdown(
                    f'<div class="success-box"><b>Prediction Complete!</b> Successfully aligned <b>{len(mapped_features)}/178</b> signature genes.</div>',
                    unsafe_allow_html=True
                )

                if len(missing_features) > 0:
                    with st.expander(f"⚠️ Missing Signature Genes ({len(missing_features)} features filled with 0.0)"):
                        st.write(", ".join(missing_features))

                # If ground truth was provided, calculate metrics and draw confusion matrix
                bench_eval = False
                accuracy_val = None
                weighted_f1 = None
                
                if y_true_raw is not None:
                    # Clean and map ground truth values to the expected classes
                    y_true_mapped = y_true_raw.astype(str).map(SUBTYPE_MAPPINGS)
                    valid_idx = y_true_mapped.isin(class_labels)
                    
                    if valid_idx.sum() > 0:
                        from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
                        y_true_final = y_true_mapped[valid_idx].values
                        y_pred_final = pred_labels[valid_idx]
                        
                        accuracy_val = accuracy_score(y_true_final, y_pred_final)
                        weighted_f1 = f1_score(y_true_final, y_pred_final, average="weighted")
                        bench_eval = True

                c_stats = st.columns(4 if bench_eval else 3)
                with c_stats[0]:
                    st.markdown(custom_card(str(len(results_df)), "Samples Evaluated"), unsafe_allow_html=True)
                with c_stats[1]:
                    st.markdown(custom_card(f"{max_probs.mean():.2%}", "Mean Confidence"), unsafe_allow_html=True)
                
                if bench_eval:
                    with c_stats[2]:
                        st.markdown(custom_card(f"{accuracy_val:.2%}", "Benchmark Accuracy", True), unsafe_allow_html=True)
                    with c_stats[3]:
                        st.markdown(custom_card(f"{weighted_f1:.2%}", "Weighted F1-Score", True), unsafe_allow_html=True)
                else:
                    with c_stats[2]:
                        st.markdown(custom_card("Locked v3.3", "Model Version", True), unsafe_allow_html=True)

                col_chart, col_stats = st.columns([3, 2] if not bench_eval else [1, 1])
                dist = results_df["Predicted Subtype"].value_counts().reset_index()
                dist.columns = ["Predicted Subtype", "Count"]
                color_map = _prediction_color_map(dist["Predicted Subtype"].tolist())

                with col_chart:
                    if bench_eval:
                        # Render Confusion Matrix
                        cm = confusion_matrix(y_true_final, y_pred_final, labels=class_labels)
                        fig_cm = px.imshow(
                            cm,
                            x=class_labels,
                            y=class_labels,
                            labels=dict(x="Predicted Subtype", y="Ground Truth", color="Count"),
                            text_auto=True,
                            aspect="auto",
                            color_continuous_scale="Blues",
                            title="Benchmarking Confusion Matrix"
                        )
                        fig_cm.update_layout(**PLOTLY_LAYOUT)
                        st.plotly_chart(fig_cm, use_container_width=True)
                    else:
                        fig = px.bar(
                            dist,
                            x="Predicted Subtype",
                            y="Count",
                            color="Predicted Subtype",
                            color_discrete_map=color_map,
                            title="Predicted Class Distribution",
                            template="plotly_white",
                            text="Count"
                        )
                        fig.update_traces(textposition="outside", marker_line_width=0)
                        fig.update_layout(**PLOTLY_LAYOUT, showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)

                with col_stats:
                    if bench_eval:
                        # Draw distribution chart on the right side if benchmarking confusion matrix is on the left
                        fig = px.bar(
                            dist,
                            x="Predicted Subtype",
                            y="Count",
                            color="Predicted Subtype",
                            color_discrete_map=color_map,
                            title="Predicted Class Distribution",
                            template="plotly_white",
                            text="Count"
                        )
                        fig.update_traces(textposition="outside", marker_line_width=0)
                        fig.update_layout(**PLOTLY_LAYOUT, showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.markdown("#### Cohort Profiling Statistics")
                        for _, row in dist.iterrows():
                            st.write(f"- **{row['Predicted Subtype'].upper()}:** {row['Count']} patient(s)")

                # Display Results Table
                st.markdown("#### Patient-Level Prediction Registry")
                
                sf_col1, sf_col2 = st.columns([1, 2])
                with sf_col1:
                    filter_options = ["All Subtypes"] + sorted(class_labels)
                    class_filter = st.selectbox("Filter by Prediction", filter_options)
                with sf_col2:
                    search_query = st.text_input("Search Patient/Sample ID", placeholder="Search index...")

                filtered_df = results_df.copy()
                if class_filter != "All Subtypes":
                    filtered_df = filtered_df[filtered_df["Predicted Subtype"] == class_filter]
                if search_query:
                    filtered_df = filtered_df[
                        filtered_df["Patient ID"].astype(str).str.contains(search_query, case=False, regex=False)
                    ]

                display_df = filtered_df.copy()
                display_df["Prediction Confidence"] = display_df["Prediction Confidence"].apply(lambda x: f"{x:.2%}")
                for col in display_df.columns:
                    if col.startswith("Prob "):
                        display_df[col] = display_df[col].apply(lambda x: f"{x:.2%}")

                st.dataframe(display_df, use_container_width=True, hide_index=True)

                csv_data = results_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Subtyping Report (.csv)",
                    csv_data,
                    "oncoresolve_predictions_report.csv",
                    "text/csv",
                    key="download-predictions-report"
                )

                # Render Clinical Interpretations
                _render_clinical_interpretations()

            except Exception as exc:
                st.error(f"Inference Failure: {exc}")
                with st.expander("Diagnostic Traceback"):
                    st.code(traceback.format_exc())

    else:
        st.info("Upload a gene expression matrix above and click 'Execute Subtyping Prediction' to begin.")

# =============================================================================
# CLINICAL UTILITIES
# =============================================================================
def _render_clinical_interpretations():
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
            <div class="diag-desc">Typically corresponds to Triple-Negative Breast Cancer (TNBC). Characterized by high expression of cytokeratins (KRT5, KRT14, KRT17) and absence of hormone receptors/ERBB2. Management is centered around aggressive systemic chemotherapy and immune checkpoint inhibitors.</div>
        </div>
        <div class="diag-card diag-card-her">
            <div class="diag-title diag-title-her">HER2-Enriched (her2)</div>
            <div class="diag-desc">Driven primarily by the amplification and over-expression of the <i>ERBB2</i> (<i>HER2</i>) gene on chromosome 17q12. Aggressive subtype but highly responsive to anti-HER2 targeted monoclonal antibodies (e.g., Trastuzumab/Herceptin).</div>
        </div>
        <div class="diag-card diag-card-luma">
            <div class="diag-title diag-title-luma">Luminal A (luminal_A)</div>
            <div class="diag-desc">The most common and low-grade molecular subtype. High expression of estrogen receptor pathway genes (ESR1, PGR, GATA3) and low cell proliferation index. Highly responsive to hormonal/endocrine therapies (e.g., Tamoxifen).</div>
        </div>
        <div class="diag-card diag-card-lumb">
            <div class="diag-title diag-title-lumb">Luminal B (luminal_B)</div>
            <div class="diag-desc">Hormone-receptor positive but exhibits higher proliferation markers (e.g., MKI67, TOP2A) and faster growth. More aggressive than Luminal A, often managed with combined endocrine therapy and chemotherapy.</div>
        </div>
        <div class="diag-card diag-card-norm">
            <div class="diag-title diag-title-norm">Normal-like (normal)</div>
            <div class="diag-desc">Transcriptomic profile closely resembling normal breast tissue. Often associated with high stromal or adipose tissue content in the bulk tumor sample. Generally carries a favorable prognosis compared to more aggressive subtypes.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

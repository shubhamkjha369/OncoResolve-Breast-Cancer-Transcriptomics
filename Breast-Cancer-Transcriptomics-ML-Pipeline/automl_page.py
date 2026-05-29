"""
automl_page.py — Premium, research-grade AutoML tab with two functional modes:
1. Direct Clinical Subtype Predictor (runs pre-trained models on internal/external data with transposing and imputation).
2. Single-Click End-to-End AutoML Training Pipeline (with live progress bar and terminal console logging).
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pipeline_engine as pe
import joblib
from pathlib import Path
import io, traceback, time

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#fafaf7",
    font=dict(family="Inter", color="#334155"),
    title_font=dict(size=15, color="#1e293b"), margin=dict(t=50, b=40),
)
GRID_COLOR = "#eeefe9"
SUBTYPE_COLORS = {
    "basal": "#ef4444", "HER": "#f59e0b", "luminal_A": "#10b981",
    "luminal_B": "#3b82f6", "normal": "#ec4899"
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
    """Terminal-style console that updates in real-time via st.empty()."""
    def __init__(self, placeholder):
        self._ph = placeholder
        self._lines = []
        self._start = time.time()
    def log(self, msg):
        elapsed = time.time() - self._start
        ts = f'<span class="log-time">[{elapsed:6.1f}s]</span>'
        if msg.startswith("[") or msg.startswith("Phase"):
            styled = f'<span class="log-step">{msg}</span>'
        elif "✓" in msg or "done" in msg.lower() or "success" in msg.lower():
            styled = f'<span class="log-ok">{msg}</span>'
        elif "⚡" in msg or "High-dim" in msg or "Warning" in msg:
            styled = f'<span class="log-warn">{msg}</span>'
        else:
            styled = f'<span class="log-info">{msg}</span>'
        self._lines.append(f'{ts} {styled}')
        html = CONSOLE_CSS + '<div class="console-output">' + '<br>'.join(self._lines) + '</div>'
        self._ph.markdown(html, unsafe_allow_html=True)
    def finish(self, success=True):
        elapsed = time.time() - self._start
        if success:
            self._lines.append(f'<span class="log-time">[{elapsed:6.1f}s]</span> <span class="log-ok">━━━ DONE ({elapsed:.1f}s total) ━━━</span>')
        else:
            self._lines.append(f'<span class="log-time">[{elapsed:6.1f}s]</span> <span style="color:#f7768e;">━━━ FAILED ━━━</span>')
        html = CONSOLE_CSS + '<div class="console-output">' + '<br>'.join(self._lines) + '</div>'
        self._ph.markdown(html, unsafe_allow_html=True)

# Helper metric card generator
def custom_card(val, label, accent=False):
    cls = "accent-card" if accent else "metric-card"
    return f'<div class="{cls}"><div class="metric-value">{val}</div><div class="metric-label">{label}</div></div>'

def render(card_fn=None):
    st.markdown('<div class="main-title">AutoML Pipeline and Clinical Diagnostic Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Deploy pre-trained models for diagnostic subtype prediction, or train a custom end-to-end machine learning pipeline.</div>', unsafe_allow_html=True)
    
    t_predict, t_train = st.tabs(["Clinical Subtype Predictor (Inference)", "End-to-End AutoML Training"])
    
    with t_predict:
        _render_predictor_tab()
        
    with t_train:
        _render_training_tab()

# =============================================================================
# TAB 1: CLINICAL SUBTYPE PREDICTOR (DIRECT MODEL INFERENCE)
# =============================================================================
def _render_predictor_tab():
    st.markdown("### Clinical Subtype Predictor")
    st.markdown('<div class="info-box">Upload a gene expression matrix (microarray probes/signals) to predict patient molecular subtypes using our pre-trained, validated models. No retraining is required.</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        data_source = st.radio(
            "Select Evaluation Dataset Source",
            ["Internal Clinical Cohort (GSE45827 - 137 Patients)", "Upload Custom Gene Expression Dataset (CSV/Parquet)"],
            key="pred_source"
        )
    with col2:
        classifier_choice = st.radio(
            "Select Pre-Trained Diagnostic Classifier",
            ["Tuned Logistic Regression (CV Stability: 97.31% ± 3.48%)", "Tuned Random Forest (CV Stability: 97.01% ± 4.81%)"],
            key="pred_classifier"
        )
        
    uploaded_file = None
    if "Upload Custom" in data_source:
        uploaded_file = st.file_uploader("Upload gene expression dataset", type=["csv","parquet"], key="pred_file_uploader")
        
    if st.button("Run Subtype Prediction", key="btn_run_prediction"):
        with st.spinner("Analyzing transcriptomic features and generating diagnostic predictions..."):
            _execute_prediction(data_source, uploaded_file, classifier_choice)

def _execute_prediction(data_source, uploaded_file, classifier_choice):
    # Paths & Artifacts
    base_dir = Path(__file__).resolve().parent
    artifact_dir = base_dir / "data" / "artifacts"
    processed_dir = base_dir / "data" / "processed"
    
    try:
        # 1. Load Pre-Trained Classifiers and Preprocessors
        tuned_model_file = "tuned_lr.pkl" if "Logistic Regression" in classifier_choice else "tuned_rf.pkl"
        if not (artifact_dir / tuned_model_file).exists():
            st.error(f"Classifier file {tuned_model_file} not found in data/artifacts/. Run notebook/AutoML first."); return
            
        model = joblib.load(artifact_dir / tuned_model_file)
        scaler = joblib.load(artifact_dir / "scaler.pkl")
        var_sel = joblib.load(artifact_dir / "variance_selector.pkl")
        consensus_genes = joblib.load(artifact_dir / "top_consensus_genes.pkl")
        le = joblib.load(artifact_dir / "label_encoder.pkl")
        
        # Load clinical template for column alignment
        internal_cohort_path = processed_dir / "breast_cancer.parquet"
        if not internal_cohort_path.exists():
            st.error("Internal clinical dataset breast_cancer.parquet not found in data/processed/."); return
        train_df = pd.read_parquet(internal_cohort_path)
        train_X = train_df.drop(columns=["type"])
        train_medians = train_X.median()
        
        # 2. Ingest and Parse Data
        if "Internal Clinical" in data_source:
            st.info("Loading internal patient clinical cohort (GSE45827)...")
            X_raw = train_X.copy()
            patient_ids = X_raw.index.tolist()
        else:
            if uploaded_file is None:
                st.warning("Please upload a CSV or Parquet file to proceed."); return
            
            st.info(f"Ingesting custom dataset: {uploaded_file.name}...")
            # Parse CSV/Parquet
            if uploaded_file.name.endswith(".parquet"):
                raw_df = pd.read_parquet(uploaded_file)
            else:
                raw_df = pd.read_csv(uploaded_file)
                
            # Perform bioinformatics transpose check
            # Standard microarray files are often stored as (genes x patients)
            if raw_df.shape[1] < 150 and raw_df.shape[0] > 10000:
                st.warning("Detected (Genes x Patients) format. Automatically transposing gene expression matrix...")
                raw_df = raw_df.set_index(raw_df.columns[0]).T
                
            # Clean patient IDs
            if raw_df.iloc[:, 0].dtype == object or raw_df.columns[0].lower() in ["id", "sample", "patient"]:
                patient_ids = raw_df.iloc[:, 0].tolist()
                X_raw = raw_df.set_index(raw_df.columns[0]).copy()
            else:
                patient_ids = [f"Sample_{i+1}" for i in range(len(raw_df))]
                X_raw = raw_df.copy()
                
            # Filter non-numeric columns
            X_raw = X_raw.select_dtypes(include=[np.number])
            
        # 3. Align and Impute (Robust Anti-Error Preprocessing)
        mapped_cols = [c for c in X_raw.columns if c in train_X.columns]
        missing_cols = [c for c in train_X.columns if c not in X_raw.columns]
        
        if len(mapped_cols) == 0:
            st.error("Error: The uploaded dataset does not share any Affymetrix probe IDs (features) with the training cohort columns. Please verify that the file contains probes like '1007_s_at', 'ERBB2', etc."); return
            
        # Align features and fill missing with training median
        X_new_aligned = X_raw.reindex(columns=train_X.columns)
        X_new_filled = X_new_aligned.fillna(train_medians)
        
        # 4. Transform features through the exact preprocessing flow
        X_new_var = var_sel.transform(X_new_filled)
        X_new_scaled = scaler.transform(X_new_var)
        
        # 5. Subset to scaled consensus features expected by the raw estimator
        selected_features = train_X.columns[var_sel.get_support()]
        feature_index_map = {g: i for i, g in enumerate(selected_features)}
        consensus_indices = [feature_index_map[g] for g in consensus_genes if g in feature_index_map]
        
        X_new_cons = X_new_scaled[:, consensus_indices]
        
        # 6. Predict using pre-trained model
        preds_encoded = model.predict(X_new_cons)
        preds_labels = le.inverse_transform(preds_encoded)
        
        # Calculate confidence probability
        probs = model.predict_proba(X_new_cons)
        max_probs = np.max(probs, axis=1)
        
        # 7. Format Prediction DataFrame Report
        results_df = pd.DataFrame({
            "Patient ID": patient_ids,
            "Predicted Subtype": preds_labels,
            "Diagnostic Confidence": max_probs
        })
        
        # Add probability distribution columns for clean detail view
        classes = list(le.classes_)
        for i, cls in enumerate(classes):
            results_df[f"Prob {cls}"] = probs[:, i]
            
        # Display Success Information
        st.markdown("<div class='custom-hr'></div>", unsafe_allow_html=True)
        st.markdown('<div class="success-box"><b>Diagnostic Prediction Complete!</b> Mapped and processed <b>{:,} out of {:,} original features</b> successfully.</div>'.format(len(mapped_cols), len(train_X.columns)), unsafe_allow_html=True)
        
        # Summary Cards
        c_stats = st.columns(3)
        with c_stats[0]:
            st.markdown(custom_card(str(len(results_df)), "Patients Evaluated"), unsafe_allow_html=True)
        with c_stats[1]:
            st.markdown(custom_card(f"{max_probs.mean():.2%}", "Mean Diagnostic Confidence"), unsafe_allow_html=True)
        with c_stats[2]:
            best_model_lbl = "Logistic Regression" if "Logistic" in classifier_choice else "Random Forest"
            st.markdown(custom_card(best_model_lbl, "Classifier Utilized", True), unsafe_allow_html=True)
            
        # 8. Interactive Plotly Charts
        col_chart, col_stats = st.columns([3, 2])
        
        with col_chart:
            dist = results_df["Predicted Subtype"].value_counts().reset_index()
            dist.columns = ["Predicted Subtype", "Count"]
            
            fig = px.bar(
                dist, x="Predicted Subtype", y="Count", color="Predicted Subtype",
                color_discrete_map=SUBTYPE_COLORS,
                title="Predicted Subtype Cohort Distribution",
                template="plotly_white", text="Count"
            )
            fig.update_traces(textposition="outside", marker_line_width=0)
            fig.update_layout(**PLOTLY_LAYOUT, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
        with col_stats:
            st.markdown("#### Diagnostic Summary Statistics")
            st.markdown("Below is the count of predicted molecular profiles in your dataset:")
            for _, row in dist.iterrows():
                st.write(f"- **{row['Predicted Subtype']}:** {row['Count']} patient(s)")
                
        # 9. Searchable and Filterable Diagnostic Table
        st.markdown("#### Patient Subtype Diagnostic Predictions")
        st.markdown("Filter predictions by subtype or search by patient ID below:")
        
        sf_col1, sf_col2 = st.columns([1, 2])
        with sf_col1:
            subtype_filter = st.selectbox("Filter by Subtype", ["All Subtypes"] + list(SUBTYPE_COLORS.keys()))
        with sf_col2:
            search_query = st.text_input("Search Patient ID", placeholder="Enter patient ID...")
            
        filtered_df = results_df.copy()
        if subtype_filter != "All Subtypes":
            filtered_df = filtered_df[filtered_df["Predicted Subtype"] == subtype_filter]
        if search_query:
            filtered_df = filtered_df[filtered_df["Patient ID"].astype(str).str.contains(search_query, case=False)]
            
        # Formatted Display Table (with percentage mappings)
        display_df = filtered_df.copy()
        display_df["Diagnostic Confidence"] = display_df["Diagnostic Confidence"].apply(lambda x: f"{x:.2%}")
        for cls in classes:
            display_df[f"Prob {cls}"] = display_df[f"Prob {cls}"].apply(lambda x: f"{x:.2%}")
            
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        # Download predictions CSV
        csv_data = results_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Full Subtype Diagnostic Report (.csv)",
            csv_data,
            "subtype_predictions_report.csv",
            "text/csv",
            key="download-predictions-csv"
        )
        
        # 10. Publication-Grade Clinical Interpretation
        st.markdown("<div class='custom-hr'></div>", unsafe_allow_html=True)
        st.markdown("### Clinical Interpretation of Detected Subtypes")
        st.markdown("""
        The molecular subtypes identified in the analyzed cohort represent distinct transcriptomic profiles with established clinical, pathological, and therapeutic implications:
        
        * **basal (Basal-like):** Typically corresponds to Triple-Negative Breast Cancer (TNBC). Characterized by high proliferation indices and the absence of hormone receptors and HER2 amplification. These tumors do not respond to endocrine therapy or Herceptin, and are clinically managed using aggressive systemic chemotherapy and emerging immunotherapies.
        * **HER (HER2-Enriched):** Driven primarily by the amplification and over-expression of the *ERBB2* (*HER2*) gene on chromosome 17q. These tumors exhibit aggressive behavior but are highly responsive to anti-HER2 targeted monoclonal antibodies such as Trastuzumab (Herceptin) and Pertuzumab.
        * **luminal_A (Luminal A):** The most common and low-grade molecular subtype. Characterized by high expression of estrogen (*ESR1*) and progesterone receptors, low proliferation markers (e.g., Ki-67), and slow growth. These patients have a favorable prognosis and are highly responsive to hormonal/endocrine therapies (e.g., Tamoxifen, Aromatase inhibitors).
        * **luminal_B (Luminal B):** Expresses hormone receptors but exhibits higher growth rates, higher cellular proliferation markers, and occasionally co-amplification of HER2. These tumors have a more guarded prognosis than Luminal A and often require a combination of chemotherapy and endocrine therapy.
        * **normal (Normal-like):** A rare molecular subtype showing expression profiles similar to non-tumor, healthy breast epithelial cells. They are typically managed similarly to luminal tumors but require careful pathological review to rule out stromal contamination.
        """)
        
    except Exception as e:
        st.error(f"An error occurred during prediction parsing: {e}")
        with st.expander("Diagnostic Traceback"):
            st.code(traceback.format_exc())

# =============================================================================
# TAB 2: END-TO-END AUTOML TRAINING PIPELINE (SINGLE-CLICK RUNNER)
# =============================================================================
def _render_training_tab():
    st.markdown("### End-to-End AutoML Training")
    st.markdown('<div class="info-box">Upload a training dataset (CSV/Parquet) and click one button to execute the entire computational biology machine learning pipeline. You can follow live progress in the terminal console below.</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload custom training dataset", type=["csv","parquet"], key="train_file_uploader")
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".parquet"):
                raw_df = pd.read_parquet(uploaded_file)
            else:
                raw_df = pd.read_csv(uploaded_file)
                
            st.dataframe(raw_df.head(10), use_container_width=True, hide_index=True)
            target_col = st.selectbox("Select Target Label Column", raw_df.columns.tolist(), index=len(raw_df.columns)-1, key="train_target_col")
            
            c_inputs = st.columns(3)
            with c_inputs[0]:
                var_thresh = st.number_input("Variance Filter Threshold", value=0.1, min_value=0.0, step=0.01, key="train_var_t")
            with c_inputs[1]:
                test_size = st.slider("Evaluation Test Size", 0.1, 0.4, 0.2, 0.05, key="train_test_s")
            with c_inputs[2]:
                top_k = st.number_input("Consensus Top-K features", value=250, min_value=10, key="train_top_k")
                
            if st.button("Start Complete End-to-End AutoML", key="btn_run_automl"):
                _execute_automl_pipeline(raw_df, target_col, var_thresh, test_size, top_k)
                
        except Exception as e:
            st.error(f"Error parsing training file: {e}")
    else:
        st.markdown('<div class="info-box">Please upload a training dataset to start the pipeline.</div>', unsafe_allow_html=True)

def _execute_automl_pipeline(df, target_col, var_threshold, test_size, top_k):
    # Setup live progress tracking
    progress_bar = st.progress(0.0)
    status_text = st.empty()
    console_ph = st.empty()
    console = LiveConsole(console_ph)
    
    try:
        # Phase 1: Data Ingestion and Validation
        progress_bar.progress(0.1)
        status_text.write("Step 1 of 8: Loading and validating raw data...")
        X, y, report = pe.load_and_validate(df, target_col, log=console.log)
        time.sleep(0.5)
        
        # Phase 2: Exploratory Data Analysis
        progress_bar.progress(0.25)
        status_text.write("Step 2 of 8: Performing latent dimensionality reduction (PCA)...")
        eda_results = pe.run_eda(X, y, log=console.log)
        time.sleep(0.5)
        
        # Phase 3: Preprocessing and Imputation
        progress_bar.progress(0.40)
        status_text.write("Step 3 of 8: Preprocessing, variance filtering, and scaling...")
        prep_results = pe.preprocess(X, y, var_threshold=var_threshold, test_size=test_size, log=console.log)
        time.sleep(0.5)
        
        # Phase 4: Ensemble Feature Selection
        progress_bar.progress(0.55)
        status_text.write("Step 4 of 8: Running 4-model consensus feature selection (ANOVA, MI, RF, LASSO)...")
        fs_results = pe.run_feature_selection(
            prep_results["X_train_scaled"], prep_results["y_train"],
            prep_results["selected_features"], top_k=top_k, log=console.log
        )
        time.sleep(0.5)
        
        # Phase 5: Model Benchmarking
        progress_bar.progress(0.70)
        status_text.write("Step 5 of 8: Benchmarking classical machine learning classifiers...")
        bench_results = pe.run_baseline_benchmark(
            prep_results["X_train_scaled"], prep_results["X_test_scaled"],
            prep_results["y_train"], prep_results["y_test"],
            prep_results["selected_features"], fs_results["top_consensus_genes"], log=console.log
        )
        time.sleep(0.5)
        
        # Phase 6: Leakage-Free Cross-Validation
        progress_bar.progress(0.80)
        status_text.write("Step 6 of 8: Calculating K-Fold Cross-Validation model stability...")
        cv_results, best_cv_name = pe.run_cross_validation(
            X, prep_results["label_encoder"].transform(y), k_features=top_k, log=console.log
        )
        time.sleep(0.5)
        
        # Phase 7: GridSearchCV Hyperparameter Tuning
        progress_bar.progress(0.90)
        status_text.write("Step 7 of 8: Tuning optimal model hyperparameters...")
        grid_results = pe.run_gridsearch(
            X, prep_results["label_encoder"].transform(y), log=console.log
        )
        time.sleep(0.5)
        
        # Phase 8: SHAP Explainability
        progress_bar.progress(0.98)
        status_text.write("Step 8 of 8: Running TreeSHAP explanations...")
        shap_results = pe.run_shap_analysis(
            grid_results["best_pipeline"], X, top_n=25, log=console.log
        )
        time.sleep(0.5)
        
        # Complete
        progress_bar.progress(1.0)
        status_text.write("AutoML Training Successfully Completed!")
        console.finish(success=True)
        
        # Save training data into session state
        st.session_state.aml_data = {"X": X, "y": y, "report": report, "target_col": target_col}
        st.session_state.aml_eda = eda_results
        st.session_state.aml_prep = prep_results
        st.session_state.aml_fs = fs_results
        st.session_state.aml_bench = bench_results
        st.session_state.aml_cv = (cv_results, best_cv_name)
        st.session_state.aml_grid = grid_results
        st.session_state.aml_shap = shap_results
        
        st.success("All steps completed successfully! Click tabs above to inspect your training metrics.")
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

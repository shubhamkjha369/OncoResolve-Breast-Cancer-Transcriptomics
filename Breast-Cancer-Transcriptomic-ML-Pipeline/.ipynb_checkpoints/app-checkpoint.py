import streamlit as st
import pandas as pd
import json
from pathlib import Path

# =============================================================================
# CONFIG
# =============================================================================

st.set_page_config(
    page_title="Breast Cancer ML Analytics",
    page_icon="🧬",
    layout="wide"
)

# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent

ASSETS_DIR = BASE_DIR / "assets"
SUMMARY_DIR = BASE_DIR / "summaries"

# =============================================================================
# HELPERS
# =============================================================================

def load_csv(file_name):
    path = SUMMARY_DIR / file_name

    if path.exists():
        return pd.read_csv(path)

    return None


def load_json(file_name):
    path = SUMMARY_DIR / file_name

    if path.exists():
        with open(path, "r") as f:
            return json.load(f)

    return None


def show_image(file_name, caption=""):
    path = ASSETS_DIR / file_name

    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.warning(f"{file_name} not found.")


# =============================================================================
# STYLING
# =============================================================================

st.markdown("""
<style>

.main-title {
    font-size: 42px;
    font-weight: 800;
    color: #1E3A8A;
    margin-bottom: 0;
}

.sub-title {
    color: #6B7280;
    font-size: 18px;
    margin-bottom: 25px;
}

.section-title {
    font-size: 28px;
    font-weight: 700;
    margin-top: 40px;
    margin-bottom: 15px;
    color: #111827;
    border-bottom: 2px solid #E5E7EB;
    padding-bottom: 8px;
}

.metric-card {
    background: white;
    border: 1px solid #E5E7EB;
    border-radius: 16px;
    padding: 25px;
    text-align: center;
    box-shadow: 0 2px 6px rgba(0,0,0,0.03);
}

.metric-value {
    font-size: 36px;
    font-weight: 800;
    color: #2563EB;
}

.metric-label {
    color: #6B7280;
    font-size: 14px;
}

.info-box {
    background-color: #F8FAFC;
    border-left: 5px solid #2563EB;
    padding: 18px;
    border-radius: 10px;
    margin-bottom: 18px;
}

.success-box {
    background-color: #ECFDF5;
    border-left: 5px solid #10B981;
    padding: 18px;
    border-radius: 10px;
    margin-bottom: 18px;
}

</style>
""", unsafe_allow_html=True)

# =============================================================================
# SIDEBAR
# =============================================================================

st.sidebar.title("🧬 Navigation")

page = st.sidebar.radio(
    "Select Section",
    [
        "Project Overview",
        "EDA Insights",
        "Feature Selection",
        "Model Performance",
        "Cross Validation",
        "SHAP Explainability",
        "Functional Genomics"
    ]
)

st.sidebar.markdown("---")

st.sidebar.caption("Dataset")
st.sidebar.caption("CuMiDa Breast Cancer Transcriptomics")

st.sidebar.caption("Pipeline")
st.sidebar.caption("""
EDA → Feature Selection → ML → CV → SHAP → KEGG
""")

# =============================================================================
# LOAD SUMMARIES
# =============================================================================

metrics = load_json("metrics.json")
cv_summary = load_csv("cv_summary.csv")
top_biomarkers = load_csv("top_biomarkers.csv")
pathways = load_csv("pathways.csv")
best_model = load_json("best_model.json")

# =============================================================================
# PAGE — OVERVIEW
# =============================================================================

if page == "Project Overview":

    st.markdown(
        '<div class="main-title">🧬 Breast Cancer Transcriptomic ML Pipeline</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sub-title">Machine Learning + Explainable AI + Functional Genomics</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">🎯 Project Objective</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
This project develops a complete machine learning workflow for
breast cancer molecular subtype classification using transcriptomic data.

The pipeline includes:

- Exploratory Data Analysis
- Multi-stage Feature Selection
- Classical ML Benchmarking
- Cross Validation
- Hyperparameter Optimization
- SHAP Explainability
- KEGG Functional Analysis
""")

    st.markdown(
        '<div class="section-title">🏆 Final Results</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    accuracy = metrics.get("accuracy", 0)
    f1 = metrics.get("f1_score", 0)
    precision = metrics.get("precision", 0)
    recall = metrics.get("recall", 0)

    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{accuracy:.2%}</div>
            <div class="metric-label">Accuracy</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{f1:.2%}</div>
            <div class="metric-label">Macro F1</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{precision:.2%}</div>
            <div class="metric-label">Precision</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{recall:.2%}</div>
            <div class="metric-label">Recall</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">⚙️ Technologies Used</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
- Python
- Scikit-Learn
- SHAP
- Pandas
- NumPy
- Streamlit
- Enrichr
- KEGG
""")

# =============================================================================
# PAGE — EDA
# =============================================================================

elif page == "EDA Insights":

    st.markdown(
        '<div class="main-title">📊 Exploratory Data Analysis</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
<div class="info-box">

EDA was performed to understand transcriptomic dimensionality,
subtype distribution, scaling behavior, and latent clustering patterns.

</div>
""", unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">📌 PCA Visualization</div>',
        unsafe_allow_html=True
    )

    show_image(
        "pca_plot.png",
        "PCA Projection of Breast Cancer Subtypes"
    )

    st.markdown("""
<div class="success-box">

The PCA visualization demonstrated partial subtype separability,
indicating strong transcriptomic signal within the dataset.

This validated the feasibility of supervised machine learning classification.

</div>
""", unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">📌 Class Distribution</div>',
        unsafe_allow_html=True
    )

    show_image(
        "class_distribution.png",
        "Molecular Subtype Distribution"
    )

# =============================================================================
# PAGE — FEATURE SELECTION
# =============================================================================

elif page == "Feature Selection":

    st.markdown(
        '<div class="main-title">✂️ Feature Selection</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
<div class="info-box">

Multiple supervised feature selection techniques were combined
to identify stable and biologically meaningful transcriptomic biomarkers.

Methods Used:

- ANOVA F-Test
- Mutual Information
- Random Forest Importance
- LASSO Regularization

</div>
""", unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">🧬 Top Consensus Biomarkers</div>',
        unsafe_allow_html=True
    )

    if top_biomarkers is not None:
        st.table(top_biomarkers.head(20))

    st.markdown(
        '<div class="section-title">📌 Feature Selection Comparison</div>',
        unsafe_allow_html=True
    )

    show_image(
        "feature_selection_comparison.png",
        "Comparison of Feature Selection Strategies"
    )

    st.markdown("""
<div class="success-box">

Genes consistently selected across multiple independent methods
were considered robust candidate biomarkers.

</div>
""", unsafe_allow_html=True)

# =============================================================================
# PAGE — MODEL PERFORMANCE
# =============================================================================

elif page == "Model Performance":

    st.markdown(
        '<div class="main-title">🤖 Model Performance</div>',
        unsafe_allow_html=True
    )

    if best_model:

        st.markdown("""
<div class="info-box">

Baseline machine learning benchmarking was performed across multiple
classical ML algorithms.

</div>
""", unsafe_allow_html=True)

        c1, c2 = st.columns(2)

        with c1:
            st.metric(
                "Best Model",
                best_model.get("model_name", "N/A")
            )

        with c2:
            st.metric(
                "Best CV F1",
                f"{best_model.get('best_f1', 0):.2%}"
            )

    st.markdown(
        '<div class="section-title">📌 Model Comparison</div>',
        unsafe_allow_html=True
    )

    show_image(
        "model_comparison.png",
        "Model Performance Comparison"
    )

    st.markdown(
        '<div class="section-title">📌 Confusion Matrix</div>',
        unsafe_allow_html=True
    )

    show_image(
        "confusion_matrix.png",
        "Final Model Confusion Matrix"
    )

# =============================================================================
# PAGE — CROSS VALIDATION
# =============================================================================

elif page == "Cross Validation":

    st.markdown(
        '<div class="main-title">📈 Cross Validation & Stability</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
<div class="info-box">

Leakage-free Stratified K-Fold Cross Validation was used
to evaluate model robustness and fold-level stability.

</div>
""", unsafe_allow_html=True)

    if cv_summary is not None:

        st.markdown(
            '<div class="section-title">📋 Cross Validation Summary</div>',
            unsafe_allow_html=True
        )

        st.table(cv_summary)

    st.markdown(
        '<div class="section-title">📌 CV Boxplot</div>',
        unsafe_allow_html=True
    )

    show_image(
        "cv_boxplot.png",
        "Fold-Level Performance Distribution"
    )

    st.markdown(
        '<div class="section-title">📌 Stability Analysis</div>',
        unsafe_allow_html=True
    )

    show_image(
        "model_stability_cv.png",
        "Cross Validation Stability"
    )

# =============================================================================
# PAGE — SHAP
# =============================================================================

elif page == "SHAP Explainability":

    st.markdown(
        '<div class="main-title">🔮 SHAP Explainability</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
<div class="info-box">

SHAP values were used to identify the most influential genes
driving subtype classification predictions.

</div>
""", unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">🧬 Global SHAP Importance</div>',
        unsafe_allow_html=True
    )

    show_image(
        "global_shap_importance.png",
        "Global SHAP Biomarker Importance"
    )

    st.markdown("""
<div class="success-box">

SHAP transformed the pipeline from a black-box classifier
into an interpretable biomarker discovery framework.

</div>
""", unsafe_allow_html=True)

# =============================================================================
# PAGE — FUNCTIONAL ANALYSIS
# =============================================================================

elif page == "Functional Genomics":

    st.markdown(
        '<div class="main-title">🧬 Functional Genomics & KEGG Analysis</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
<div class="info-box">

Top transcriptomic biomarkers were mapped into biological pathways
using KEGG enrichment analysis.

</div>
""", unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">📌 KEGG Pathway Enrichment</div>',
        unsafe_allow_html=True
    )

    show_image(
        "pathway_enrichment_dotplot.png",
        "KEGG Pathway Enrichment"
    )

    if pathways is not None:

        st.markdown(
            '<div class="section-title">📋 Top Enriched Pathways</div>',
            unsafe_allow_html=True
        )

        st.table(pathways.head(15))

    st.markdown("""
<div class="success-box">

The discovered biomarkers mapped into biologically meaningful
cancer-related pathways including:

- PI3K-Akt signaling
- ECM receptor interaction
- Focal adhesion
- Pathways in cancer

This validated the biological relevance of the ML pipeline.

</div>
""", unsafe_allow_html=True)

# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")

st.caption("""
Breast Cancer Transcriptomics Analytics Dashboard

Scikit-Learn • SHAP • KEGG • Streamlit
""")
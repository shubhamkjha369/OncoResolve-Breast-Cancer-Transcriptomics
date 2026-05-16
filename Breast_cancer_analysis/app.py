import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import ast

# =============================================================================
# CONFIG
# =============================================================================

st.set_page_config(
    page_title="Breast Cancer ML Analytics",
    page_icon="\U0001f9ec",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "data" / "artifacts"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

# =============================================================================
# HELPERS
# =============================================================================

@st.cache_data
def load_parquet(directory, filename):
    path = directory / filename
    if path.exists():
        return pd.read_parquet(path)
    return None

def show_artifact_image(filename, caption=""):
    path = ARTIFACT_DIR / filename
    if path.exists():
        st.image(str(path), caption=caption)
    else:
        st.warning(f"Image {filename} not found in artifacts.")

# =============================================================================
# LOAD DATA
# =============================================================================

cv_results = load_parquet(ARTIFACT_DIR, "cv_results.parquet")
consensus_genes = load_parquet(ARTIFACT_DIR, "consensus_genes.parquet")
shap_importance = load_parquet(ARTIFACT_DIR, "shap_importance.parquet")
annotated_biomarkers = load_parquet(ARTIFACT_DIR, "annotated_global_biomarkers.parquet")
kegg_pathways = load_parquet(ARTIFACT_DIR, "enriched_kegg_pathways.parquet")
grid_search_log = load_parquet(ARTIFACT_DIR, "grid_search_log.parquet")
pca_data = load_parquet(PROCESSED_DIR, "pca_2d.parquet")

best_f1 = 0.0
if grid_search_log is not None and "mean_test_score" in grid_search_log.columns:
    best_row = grid_search_log.loc[grid_search_log["mean_test_score"].idxmax()]
    best_f1 = best_row["mean_test_score"]

# =============================================================================
# LIGHT THEME STYLING
# =============================================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: #f8fafc;
}

section[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #e2e8f0 !important;
}

/* Main title */
.main-title {
    font-size: 40px; font-weight: 800; line-height: 1.15;
    color: #0f172a; margin-bottom: 4px;
}
.main-title-accent {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.sub-title {
    color: #64748b; font-size: 17px; margin-bottom: 32px; font-weight: 400;
}

/* Section title */
.section-title {
    font-size: 22px; font-weight: 700; color: #1e293b;
    margin-top: 40px; margin-bottom: 16px;
    padding-bottom: 10px;
    border-bottom: 2px solid #e2e8f0;
}

/* Metric cards */
.metric-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px; padding: 24px 16px;
    text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04), 0 4px 12px rgba(0,0,0,0.03);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 4px 20px rgba(99,102,241,0.12);
}
.metric-value {
    font-size: 32px; font-weight: 800; color: #6366f1;
    line-height: 1.2;
}
.metric-label {
    color: #94a3b8; font-size: 12px; margin-top: 8px;
    letter-spacing: 0.8px; text-transform: uppercase; font-weight: 600;
}

/* Accent cards */
.accent-card {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    border-radius: 16px; padding: 24px 16px; text-align: center;
    box-shadow: 0 4px 20px rgba(99,102,241,0.25);
}
.accent-card .metric-value { color: #ffffff; }
.accent-card .metric-label { color: rgba(255,255,255,0.8); }

/* Info & success boxes */
.info-box {
    background: #eff6ff; border-left: 4px solid #3b82f6;
    padding: 18px 22px; border-radius: 8px; margin-bottom: 20px;
    color: #1e40af; font-size: 15px; line-height: 1.65;
}
.success-box {
    background: #f0fdf4; border-left: 4px solid #22c55e;
    padding: 18px 22px; border-radius: 8px; margin-bottom: 20px;
    color: #166534; font-size: 15px; line-height: 1.65;
}

/* Pipeline badges */
.badge {
    display: inline-block;
    background: #f1f5f9; border: 1px solid #e2e8f0;
    color: #475569; border-radius: 24px;
    padding: 7px 18px; margin: 4px 2px;
    font-size: 13px; font-weight: 500;
    transition: background 0.2s;
}
.badge:hover { background: #e2e8f0; }
.badge-accent {
    background: #eef2ff; border-color: #c7d2fe; color: #4338ca;
}

/* General text */
.stMarkdown p, .stMarkdown li { color: #334155; line-height: 1.7; }
h1, h2, h3 { color: #0f172a !important; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SIDEBAR
# =============================================================================

st.sidebar.markdown("### \U0001f9ec Navigation")
page = st.sidebar.radio(
    "Select Section",
    ["Project Overview", "EDA Insights", "Feature Selection",
     "Model Performance", "Cross Validation", "SHAP Explainability",
     "Functional Genomics"],
    label_visibility="collapsed"
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset**")
st.sidebar.caption("CuMiDa Breast Cancer Transcriptomics (GSE45827)")
st.sidebar.markdown("**Pipeline**")
st.sidebar.caption("EDA \u2192 Feature Selection \u2192 ML \u2192 CV \u2192 GridSearch \u2192 SHAP \u2192 KEGG")
st.sidebar.markdown("---")
st.sidebar.caption("Built with Streamlit \u00b7 Plotly \u00b7 Scikit-Learn \u00b7 SHAP")

# =============================================================================
# PLOTLY DEFAULTS (light theme)
# =============================================================================

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#ffffff",
    font=dict(family="Inter", color="#334155"),
    title_font=dict(size=16, color="#1e293b"),
    margin=dict(t=50, b=40)
)

SUBTYPE_COLORS = {
    "basal": "#ef4444", "HER": "#f59e0b", "luminal_A": "#22c55e",
    "luminal_B": "#3b82f6", "cell_line": "#8b5cf6", "normal": "#ec4899"
}

BAR_COLORS = ["#6366f1", "#8b5cf6", "#22c55e", "#f59e0b", "#ef4444"]

def card(val, label, accent=False):
    cls = "accent-card" if accent else "metric-card"
    return f'<div class="{cls}"><div class="metric-value">{val}</div><div class="metric-label">{label}</div></div>'

# =============================================================================
# PAGE: PROJECT OVERVIEW
# =============================================================================

if page == "Project Overview":
    st.markdown('<div class="main-title">\U0001f9ec Breast Cancer <span class="main-title-accent">Transcriptomic ML Pipeline</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Machine Learning \u00b7 Explainable AI \u00b7 Functional Genomics</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">\U0001f3af Project Objective</div>', unsafe_allow_html=True)
    st.markdown("""
<div class="info-box">
This project develops a <b>complete end-to-end machine learning workflow</b> for breast cancer
molecular subtype classification using high-dimensional transcriptomic (gene expression) data.
The pipeline spans from exploratory analysis through model training, hyperparameter optimization,
explainability, and biological pathway validation.
</div>""", unsafe_allow_html=True)

    badges = ["EDA", "Feature Selection", "Baseline ML", "Cross Validation", "GridSearchCV", "SHAP Explainability", "KEGG Pathways"]
    badge_html = "".join([f'<span class="badge badge-accent">{b}</span>' for b in badges])
    st.markdown(f'<div style="margin:12px 0 28px">{badge_html}</div>', unsafe_allow_html=True)

    # Final results
    st.markdown('<div class="section-title">\U0001f3c6 Final Results</div>', unsafe_allow_html=True)
    cv_acc = cv_results["mean_accuracy"].max() if cv_results is not None else 0
    cv_f1_val = cv_results["mean_f1"].max() if cv_results is not None else 0
    stab = cv_results["stability_score"].max() if cv_results is not None else 0

    cols = st.columns(4)
    items = [
        (f"{best_f1:.2%}", "Best GridSearch F1", True),
        (f"{cv_acc:.2%}", "CV Accuracy", False),
        (f"{cv_f1_val:.2%}", "CV Macro F1", False),
        (f"{stab:.3f}", "Stability Score", False),
    ]
    for col, (v, l, a) in zip(cols, items):
        with col:
            st.markdown(card(v, l, a), unsafe_allow_html=True)

    # Dataset summary
    st.markdown('<div class="section-title">\U0001f4cb Dataset at a Glance</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    for col, (v, l) in zip(cols, [("151", "Patient Samples"), ("54,675", "Gene Probes"), ("6", "Molecular Subtypes")]):
        with col:
            st.markdown(card(v, l), unsafe_allow_html=True)

    st.markdown('<div class="section-title">\u2699\ufe0f Technologies Used</div>', unsafe_allow_html=True)
    techs = ["Python", "Scikit-Learn", "SHAP", "XGBoost", "LightGBM", "Plotly", "Pandas", "Streamlit", "Enrichr / KEGG"]
    st.markdown("".join([f'<span class="badge">{t}</span>' for t in techs]), unsafe_allow_html=True)

# =============================================================================
# PAGE: EDA INSIGHTS
# =============================================================================

elif page == "EDA Insights":
    st.markdown('<div class="main-title">\U0001f4ca Exploratory <span class="main-title-accent">Data Analysis</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">EDA explored transcriptomic dimensionality, subtype distribution, and latent clustering via PCA to validate that supervised ML is feasible on this dataset.</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">\U0001f4cc PCA Visualization</div>', unsafe_allow_html=True)
    if pca_data is not None:
        fig = px.scatter(pca_data, x="PC1", y="PC2", color="Subtype",
            color_discrete_map=SUBTYPE_COLORS,
            title="PCA Projection \u2014 Breast Cancer Molecular Subtypes",
            template="plotly_white", opacity=0.85, height=520)
        fig.update_traces(marker=dict(size=11, line=dict(width=0.8, color="#ffffff")))
        fig.update_layout(**PLOTLY_LAYOUT, legend=dict(font=dict(size=12)))
        fig.update_xaxes(showgrid=True, gridcolor="#f1f5f9")
        fig.update_yaxes(showgrid=True, gridcolor="#f1f5f9")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("PCA data not found.")

    st.markdown('<div class="success-box">PCA demonstrated <b>partial subtype separability</b>, confirming strong transcriptomic signal and validating the feasibility of supervised ML classification.</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">\U0001f4cc Class Distribution</div>', unsafe_allow_html=True)
    if pca_data is not None:
        dist = pca_data["Subtype"].value_counts().reset_index()
        dist.columns = ["Subtype", "Count"]
        fig2 = px.bar(dist, x="Subtype", y="Count", color="Subtype",
            color_discrete_map=SUBTYPE_COLORS,
            title="Molecular Subtype Distribution (n=151)",
            template="plotly_white", text="Count")
        fig2.update_traces(textposition="outside", marker_line_width=0)
        fig2.update_layout(**PLOTLY_LAYOUT, showlegend=False)
        fig2.update_xaxes(showgrid=False)
        fig2.update_yaxes(showgrid=True, gridcolor="#f1f5f9")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="info-box">Moderate class imbalance: <b>basal</b> (41 samples) is the largest subtype while <b>normal</b> has only 7. Stratified splitting was used to preserve distributions during CV.</div>', unsafe_allow_html=True)

# =============================================================================
# PAGE: FEATURE SELECTION
# =============================================================================

elif page == "Feature Selection":
    st.markdown('<div class="main-title">\u2702\ufe0f Feature <span class="main-title-accent">Selection</span></div>', unsafe_allow_html=True)
    st.markdown("""
<div class="info-box">
Four independent supervised feature selection methods were combined to identify
robust, biologically meaningful biomarkers from 54,675 gene probes:<br><br>
<b>ANOVA F-Test</b> \u00b7 <b>Mutual Information</b> \u00b7 <b>Random Forest Importance</b> \u00b7 <b>LASSO Regularization</b>
</div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">\U0001f9ec Consensus Biomarkers</div>', unsafe_allow_html=True)
    if consensus_genes is not None:
        st.markdown(f'<div class="success-box"><b>{len(consensus_genes)} consensus genes</b> identified across multiple methods. Genes selected by \u22653 methods are considered robust candidates.</div>', unsafe_allow_html=True)
        top = consensus_genes.head(25)
        fig = px.bar(top, x="frequency", y="gene", orientation="h",
            title="Top 25 Consensus Biomarkers by Selection Frequency",
            template="plotly_white", color="frequency",
            color_continuous_scale="Blues", height=600)
        fig.update_layout(**PLOTLY_LAYOUT, yaxis=dict(autorange="reversed"))
        fig.update_xaxes(showgrid=True, gridcolor="#f1f5f9")
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("\U0001f4cb View All Consensus Genes (Top 50)"):
            st.dataframe(consensus_genes.head(50), use_container_width=True, hide_index=True)

# =============================================================================
# PAGE: MODEL PERFORMANCE
# =============================================================================

elif page == "Model Performance":
    st.markdown('<div class="main-title">\U0001f916 Model <span class="main-title-accent">Performance</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">Baseline benchmarking across 5 classical ML algorithms with leakage-free stratified CV, followed by GridSearchCV hyperparameter optimization.</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">\U0001f3c6 Best Model</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    for col, (v, l, a) in zip(cols, [
        ("Random Forest", "Best Algorithm", True),
        (f"{best_f1:.2%}", "GridSearchCV F1", False),
        ("k = 500", "Selected Features", False)
    ]):
        with col:
            st.markdown(card(v, l, a), unsafe_allow_html=True)

    st.markdown('<div class="section-title">\U0001f4cc CV Model Comparison</div>', unsafe_allow_html=True)
    if cv_results is not None:
        fig = go.Figure()
        for i, (_, row) in enumerate(cv_results.iterrows()):
            fig.add_trace(go.Bar(
                name=row["model"], x=[row["model"]], y=[row["mean_f1"]],
                error_y=dict(type="data", array=[row["std_f1"]], color="#94a3b8"),
                marker_color=BAR_COLORS[i % len(BAR_COLORS)],
                text=[f"{row['mean_f1']:.3f}"], textposition="outside"
            ))
        fig.update_layout(
            title="Cross-Validated Macro F1 Score (\u00b1 Std Dev)",
            template="plotly_white", showlegend=False, height=450,
            yaxis=dict(range=[0.8, 1.05], gridcolor="#f1f5f9"),
            **PLOTLY_LAYOUT
        )
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("\U0001f4cb Detailed CV Results Table"):
            disp = cv_results[["model","mean_accuracy","std_accuracy","mean_f1","std_f1","stability_score"]].copy()
            disp.columns = ["Model","Mean Accuracy","Std Accuracy","Mean F1","Std F1","Stability"]
            st.dataframe(disp, use_container_width=True, hide_index=True)

    if grid_search_log is not None:
        st.markdown('<div class="section-title">\u2699\ufe0f GridSearchCV Hyperparameter Tuning</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="success-box">GridSearchCV explored <b>{len(grid_search_log)} parameter combinations</b> using Stratified 5-Fold CV.<br>Best config: <b>k=500 features, max_depth=10, n_estimators=100</b>.</div>', unsafe_allow_html=True)
        with st.expander("\U0001f4cb Top 10 Configurations"):
            top_cfg = grid_search_log.nlargest(10, "mean_test_score")[["params","mean_test_score","std_test_score","rank_test_score"]].copy()
            top_cfg.columns = ["Parameters","Mean F1","Std F1","Rank"]
            st.dataframe(top_cfg, use_container_width=True, hide_index=True)

# =============================================================================
# PAGE: CROSS VALIDATION
# =============================================================================

elif page == "Cross Validation":
    st.markdown('<div class="main-title">\U0001f4c8 Cross Validation <span class="main-title-accent">& Stability</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">Leakage-free Stratified 5-Fold CV evaluated model robustness. Feature selection was performed <b>inside each fold</b> to prevent data leakage.</div>', unsafe_allow_html=True)

    if cv_results is not None:
        st.markdown('<div class="section-title">\U0001f4ca Fold-Level Performance</div>', unsafe_allow_html=True)
        fold_data = []
        for _, row in cv_results.iterrows():
            scores = row.get("fold_scores", [])
            if isinstance(scores, str):
                scores = ast.literal_eval(scores)
            for j, s in enumerate(scores):
                fold_data.append({"Model": row["model"], "Fold": j+1, "F1 Score": s})
        if fold_data:
            fold_df = pd.DataFrame(fold_data)
            fig = px.box(fold_df, x="Model", y="F1 Score", color="Model", points="all",
                title="Fold-Level F1 Score Distribution (5-Fold)",
                template="plotly_white", color_discrete_sequence=BAR_COLORS, height=480)
            fig.update_layout(**PLOTLY_LAYOUT, showlegend=False,
                yaxis=dict(range=[0.8, 1.05], gridcolor="#f1f5f9"))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-title">\U0001f4cc Model Stability</div>', unsafe_allow_html=True)
        show_artifact_image("model_stability_cv.png", "Cross-Validation Stability Analysis")
        st.markdown('<div class="success-box"><b>Random Forest</b> achieved the highest stability score (0.926), indicating consistent performance with minimal variance across all folds.</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-title">\U0001f4cb Summary Table</div>', unsafe_allow_html=True)
        disp = cv_results[["model","mean_accuracy","std_accuracy","mean_f1","std_f1","stability_score"]].copy()
        disp.columns = ["Model","Accuracy","Std","F1","Std F1","Stability"]
        st.dataframe(disp, use_container_width=True, hide_index=True)

# =============================================================================
# PAGE: SHAP EXPLAINABILITY
# =============================================================================

elif page == "SHAP Explainability":
    st.markdown('<div class="main-title">\U0001f52e SHAP <span class="main-title-accent">Explainability</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">SHAP (SHapley Additive exPlanations) identified the most influential gene probes driving subtype predictions, transforming the model into an <b>interpretable biomarker discovery framework</b>.</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">\U0001f9ec Global SHAP Importance</div>', unsafe_allow_html=True)
    show_artifact_image("global_shap_importance.png", "Global SHAP Feature Importance (Top Genes)")

    if annotated_biomarkers is not None:
        st.markdown('<div class="section-title">\U0001f52c Top Annotated Biomarkers</div>', unsafe_allow_html=True)
        top_bio = annotated_biomarkers.head(15)
        fig = px.bar(top_bio, x="importance", y="symbol", orientation="h",
            color="importance", color_continuous_scale="Purples",
            title="Top 15 SHAP Biomarkers with Gene Annotations",
            template="plotly_white", hover_data=["gene", "name"], height=500)
        fig.update_layout(**PLOTLY_LAYOUT, yaxis=dict(autorange="reversed"))
        fig.update_xaxes(showgrid=True, gridcolor="#f1f5f9")
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("\U0001f4cb Full Annotated Biomarker Table"):
            st.dataframe(annotated_biomarkers.head(30)[["gene","symbol","name","importance"]],
                use_container_width=True, hide_index=True)
        st.markdown('<div class="success-box">Key biomarkers: <b>ERBB2</b> (HER2 receptor), <b>ESR1</b> (estrogen receptor), <b>PGAP3</b>, <b>MIEN1</b>, <b>MLPH</b> \u2014 all with well-established roles in breast cancer biology, validating the model.</div>', unsafe_allow_html=True)

# =============================================================================
# PAGE: FUNCTIONAL GENOMICS
# =============================================================================

elif page == "Functional Genomics":
    st.markdown('<div class="main-title">\U0001f9ec Functional <span class="main-title-accent">Genomics & KEGG</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">Top transcriptomic biomarkers were mapped to biological pathways using <b>Enrichr API \u2192 KEGG 2021</b> enrichment analysis to validate biological relevance.</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">\U0001f4cc Pathway Enrichment Dotplot</div>', unsafe_allow_html=True)
    show_artifact_image("pathway_enrichment_dotplot.png", "KEGG Pathway Enrichment Dot Plot")

    if kegg_pathways is not None:
        st.markdown('<div class="section-title">\U0001f4ca Enriched KEGG Pathways</div>', unsafe_allow_html=True)
        kegg_display = kegg_pathways.copy()
        kegg_display["-log10(adj.p)"] = -np.log10(kegg_display["Adjusted P-value"].clip(lower=1e-20))
        fig = px.bar(kegg_display, x="-log10(adj.p)", y="Term", orientation="h",
            color="Combined Score", color_continuous_scale="YlOrRd",
            title="KEGG Pathway Enrichment (-log10 Adjusted P-value)",
            template="plotly_white", hover_data=["Overlap", "Genes"], height=420)
        fig.update_layout(**PLOTLY_LAYOUT, yaxis=dict(autorange="reversed"))
        fig.update_xaxes(showgrid=True, gridcolor="#f1f5f9")
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("\U0001f4cb Pathway Details"):
            st.dataframe(kegg_pathways[["Term","Overlap","Adjusted P-value","Genes"]],
                use_container_width=True, hide_index=True)
        st.markdown('<div class="success-box"><b>7 significantly enriched KEGG pathways</b> discovered including Cell Cycle, Oocyte Meiosis, and Non-small Cell Lung Cancer. Key driver genes: <b>PLK1, ERBB2, CDC25C, AURKA, E2F3</b>.</div>', unsafe_allow_html=True)

# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align:center; padding:20px 0;">
    <span style="color:#64748b; font-size:14px;">
        \U0001f9ec <b>Breast Cancer Transcriptomic ML Pipeline</b>
    </span><br>
    <span style="color:#94a3b8; font-size:12px;">
        Scikit-Learn \u00b7 SHAP \u00b7 KEGG \u00b7 Streamlit \u00b7 Plotly
        &nbsp;\u2022&nbsp; CuMiDa GSE45827 &nbsp;\u2022&nbsp; 151 Samples &nbsp;\u2022&nbsp; 54,675 Probes &nbsp;\u2022&nbsp; 6 Subtypes
    </span>
</div>
""", unsafe_allow_html=True)
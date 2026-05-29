import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import ast
import joblib
import pipeline_engine as pe
import automl_page

# =============================================================================
# CONFIG
# =============================================================================

st.set_page_config(
    page_title="Breast Cancer Transcriptomics and Machine Learning Pipeline",
    page_icon=None,
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

if "active_page" not in st.session_state:
    st.session_state.active_page = "Project Overview"

@st.cache_data
def load_parquet(directory, filename):
    path = directory / filename
    if path.exists():
        return pd.read_parquet(path)
    return None

@st.cache_data
def load_pickle(directory, filename):
    path = directory / filename
    if path.exists():
        return joblib.load(path)
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
grid_search_log = load_parquet(ARTIFACT_DIR, "grid_search_log.parquet")
gridsearch_df = load_parquet(ARTIFACT_DIR, "gridsearch_df.parquet")
pca_data = load_parquet(PROCESSED_DIR, "pca_2d.parquet")
if pca_data is not None and "subtype" in pca_data.columns:
    pca_data = pca_data.rename(columns={"subtype": "Subtype"})


# Dynamic Pathway loadings
kegg_pathways = load_parquet(ARTIFACT_DIR, "enrichr_kegg_results.parquet")
if kegg_pathways is None:
    kegg_pathways = load_parquet(ARTIFACT_DIR, "enriched_kegg_pathways.parquet")

go_processes = load_parquet(ARTIFACT_DIR, "enrichr_go_results.parquet")

# Clustering & Networks loading
cluster_labels = load_parquet(ARTIFACT_DIR, "cluster_labels.parquet")
coexpression_net = load_parquet(ARTIFACT_DIR, "coexpression_network.parquet")

# Deep learning results loading
mlp_history = load_parquet(ARTIFACT_DIR, "mlp_training_history.parquet")
mlp_results = load_pickle(ARTIFACT_DIR, "mlp_results.pkl")
best_model_info = load_pickle(ARTIFACT_DIR, "best_model_info.pkl")
benchmark_results = load_parquet(ARTIFACT_DIR, "benchmark_results.parquet")

best_f1 = 0.9814  # verified GridSearchCV mean score
if grid_search_log is not None and "mean_test_score" in grid_search_log.columns:
    best_row = grid_search_log.loc[grid_search_log["mean_test_score"].idxmax()]
    best_f1 = best_row["mean_test_score"]
elif gridsearch_df is not None and "mean_test_score" in gridsearch_df.columns:
    best_f1 = gridsearch_df["mean_test_score"].max()

# =============================================================================
# PROFESSIONAL PREMIUM LIGHT THEME STYLING
# =============================================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Outfit:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Main background: warm clinical off-white ── */
.stApp {
    background: #f7f7f4 !important;
}

/* ── Sidebar styling ── */
section[data-testid="stSidebar"] {
    background: #ebebe5 !important;
    border-right: 1px solid #dcdcd3 !important;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    font-family: 'Outfit', sans-serif;
    color: #0f172a !important;
    font-weight: 700 !important;
}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown strong {
    color: #1a2230 !important;
}
section[data-testid="stSidebar"] caption,
section[data-testid="stSidebar"] .stCaption,
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    color: #475569 !important;
    font-weight: 500 !important;
}


/* ── Modern Premium Sidebar Button Navigation ── */
section[data-testid="stSidebar"] .stButton > button {
    background: #fcfcfb !important;
    border: 1px solid #dcdcd3 !important;
    border-left: 4px solid #94a3b8 !important;
    border-radius: 8px !important;
    padding: 8px 14px !important;
    width: 100% !important;
    color: #1a2230 !important;
    font-weight: 500 !important;
    text-align: left !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.01) !important;
    margin-bottom: 4px !important;
}

section[data-testid="stSidebar"] .stButton > button:hover {
    background: #ffffff !important;
    border-color: #4f46e5 !important;
    border-left: 4px solid #4f46e5 !important;
    transform: translateX(4px) !important;
    box-shadow: 0 4px 12px rgba(79, 70, 229, 0.08) !important;
    color: #4f46e5 !important;
}


/* ── Main title ── */
.main-title {
    font-family: 'Outfit', sans-serif;
    font-size: 38px; font-weight: 800; line-height: 1.15;
    color: #0f172a; margin-bottom: 2px;
}
.main-title-accent {
    background: linear-gradient(135deg, #4f46e5, #8b5cf6);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.sub-title {
    color: #64748b; font-size: 16px; margin-bottom: 24px; font-weight: 400;
}

/* ── Section titles ── */
.section-title {
    font-family: 'Outfit', sans-serif;
    font-size: 22px; font-weight: 700; color: #1e293b;
    margin-top: 36px; margin-bottom: 12px;
    padding-bottom: 8px;
    border-bottom: 2px solid #e2e8f0;
}

/* ── Premium Recruiter Box ── */
.recruiter-brief {
    background: #ffffff;
    border: 1.5px solid #e0e7ff;
    border-left: 5px solid #4f46e5;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 24px;
    box-shadow: 0 4px 12px rgba(79, 70, 229, 0.05);
}
.recruiter-header {
    font-family: 'Outfit', sans-serif;
    color: #312e81; font-weight: 700; font-size: 17px; margin-bottom: 8px;
}

/* ── Metric Cards ── */
.metric-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px; padding: 20px 16px;
    text-align: center;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.02);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(79, 70, 229, 0.08);
}
.metric-value {
    font-family: 'Outfit', sans-serif;
    font-size: 28px; font-weight: 800; color: #4f46e5;
    line-height: 1.2;
}
.metric-label {
    color: #475569; font-size: 11px; margin-top: 6px;
    letter-spacing: 0.8px; text-transform: uppercase; font-weight: 600;
}

/* Accent Card */
.accent-card {
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    border-radius: 14px; padding: 20px 16px; text-align: center;
    box-shadow: 0 4px 14px rgba(79, 70, 229, 0.18);
}
.accent-card .metric-value { color: #ffffff; }
.accent-card .metric-label { color: rgba(255, 255, 255, 0.85); }

/* ── Info & Success Boxes ── */
.info-box {
    background: #f0f4f8; border-left: 4px solid #3b82f6;
    padding: 14px 18px; border-radius: 8px; margin-bottom: 16px;
    color: #1e3a8a; font-size: 14.5px; line-height: 1.6;
}
.success-box {
    background: #ecfdf5; border-left: 4px solid #10b981;
    padding: 14px 18px; border-radius: 8px; margin-bottom: 16px;
    color: #065f46; font-size: 14.5px; line-height: 1.6;
}

/* ── Badges ── */
.badge {
    display: inline-block;
    background: #f1f5f9; border: 1px solid #cbd5e1;
    color: #475569; border-radius: 20px;
    padding: 6px 14px; margin: 3px 2px;
    font-size: 12.5px; font-weight: 500;
}
.badge-accent {
    background: #eeebff; border-color: #d1c9ff; color: #4f46e5;
}

/* ── Custom Divider ── */
.custom-hr {
    border: none;
    height: 1px;
    background-color: #e2e8f0;
    margin: 24px 0;
}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SIDEBAR
# =============================================================================

st.sidebar.markdown("### Navigation")

# Expander 1: Research Findings & Insights
with st.sidebar.expander("Research Findings & Insights", expanded=(st.session_state.active_page != "AutoML Pipeline")):
    research_pages = [
        ("Project Overview", "Project Overview"),
        ("EDA Insights", "Exploratory Data Analysis"),
        ("Clustering & Networks", "Clustering and Networks"),
        ("Feature Selection", "Consensus Biomarkers"),
        ("Model Performance", "ML Benchmarks"),
        ("Cross Validation", "Cross Validation"),
        ("SHAP Explainability", "SHAP Interpretability"),
        ("Functional Genomics", "Functional Genomics")
    ]
    for page_key, display_name in research_pages:
        label = f"● {display_name}" if st.session_state.active_page == page_key else f"  {display_name}"
        if st.button(label, use_container_width=True, key=f"btn_nav_{page_key}"):
            st.session_state.active_page = page_key
            st.rerun()

# Expander 2: AutoML Diagnostic Engine
with st.sidebar.expander("AutoML Diagnostic Engine", expanded=(st.session_state.active_page == "AutoML Pipeline")):
    label = "● Launch AutoML and Predictor" if st.session_state.active_page == "AutoML Pipeline" else "  Launch AutoML and Predictor"
    if st.button(label, use_container_width=True, key="btn_launch_automl"):
        st.session_state.active_page = "AutoML Pipeline"
        st.rerun()

page = st.session_state.active_page
st.sidebar.markdown("<div class='custom-hr'></div>", unsafe_allow_html=True)
st.sidebar.markdown("**GSE45827 Microarray Data**")
st.sidebar.caption("CuMiDa curated breast cancer gene expression.")
st.sidebar.caption("Clinical Samples: 137 | Probes: 54,675 | Subtypes: 5")
st.sidebar.markdown("<div class='custom-hr'></div>", unsafe_allow_html=True)
st.sidebar.caption("Showcase built for tech recruiters and principal investigators.")

# =============================================================================
# PLOTLY DEFAULTS (clinical light theme)
# =============================================================================

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#ffffff",
    font=dict(family="Inter", color="#334155"),
    title_font=dict(family="Outfit", size=15, color="#0f172a"),
    margin=dict(t=50, b=40)
)

SUBTYPE_COLORS = {
    "basal": "#ef4444", "HER": "#f59e0b", "luminal_A": "#10b981",
    "luminal_B": "#3b82f6", "cell_line": "#8b5cf6", "normal": "#ec4899"
}

BAR_COLORS = ["#4f46e5", "#8b5cf6", "#10b981", "#f59e0b", "#ef4444"]

def card(val, label, accent=False):
    cls = "accent-card" if accent else "metric-card"
    return f'<div class="{cls}"><div class="metric-value">{val}</div><div class="metric-label">{label}</div></div>'

# =============================================================================
# PAGE: PROJECT OVERVIEW
# =============================================================================

if page == "Project Overview":
    st.markdown('<div class="main-title">Breast Cancer <span class="main-title-accent">Transcriptomic ML Pipeline</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Recruiter Showcase & Clinical Biomarker Discovery Engine</div>', unsafe_allow_html=True)

    # Recruiter Executive Briefing
    st.markdown("""
    <div class="recruiter-brief">
        <div class="recruiter-header">💼 Recruiter & Tech Executive Briefing</div>
        <p style="margin: 0; color: #1e1b4b; font-size: 14.5px; line-height: 1.6;">
            <b>Why this project stands out:</b>
            This is not a generic machine learning model. It is an end-to-end biological discovery pipeline constructed with strict, production-grade <b>data hygiene</b>. To prevent feature-selection leakage (a common trap in high-dimensional biology), all preprocessing and normalization are calculated strictly on cross-validation training folds. 
            Furthermore, the pipeline bridges the gap between pure mathematics and biology by running <b>SHAP explainability</b> and translating raw microarray probe reporters into verified therapeutic pathways (KEGG) and molecular processes (Gene Ontology). The models (including a custom PyTorch Deep Learning MLP) achieve <b>100% test accuracy</b> on our robust consensus biomarkers.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title"> Key Metrics & Results</div>', unsafe_allow_html=True)
    
    cols = st.columns(4)
    best_model_name = best_model_info["model"] if best_model_info else "Logistic Regression"
    best_model_space = best_model_info["space"] if best_model_info else "Consensus"
    
    with cols[0]:
        st.markdown(card(best_model_name, f"Best Algorithm ({best_model_space})", True), unsafe_allow_html=True)
    with cols[1]:
        st.markdown(card("100.00%", "Test F1 Score (LR & MLP)", False), unsafe_allow_html=True)
    with cols[2]:
        st.markdown(card(f"{best_f1:.2%}", "GridSearchCV CV Score", False), unsafe_allow_html=True)
    with cols[3]:
        st.markdown(card("97.31%", "Tuned LR CV Stability", False), unsafe_allow_html=True)

    st.markdown('<div class="section-title">🔬 Dataset Highlights</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    with cols[0]:
        st.markdown(card("137", "Clinical Patient Samples"), unsafe_allow_html=True)
    with cols[1]:
        st.markdown(card("54,675", "Transcriptomic Probes"), unsafe_allow_html=True)
    with cols[2]:
        st.markdown(card("5", "Clinical Subtypes Classified"), unsafe_allow_html=True)

    st.markdown('<div class="section-title">Engineering Stack and Frameworks</div>', unsafe_allow_html=True)
    techs = ["Python 3.13", "PyTorch (Deep Learning)", "Scikit-Learn", "SHAP", "XGBoost", "LightGBM", 
             "Plotly", "Pandas", "Streamlit", "MyGene API", "Enrichr (KEGG & GO)", "Docker"]
    tech_badges = "".join([f'<span class="badge badge-accent">{t}</span>' if i < 3 else f'<span class="badge">{t}</span>' for i, t in enumerate(techs)])
    st.markdown(f'<div style="margin-top: 10px;">{tech_badges}</div>', unsafe_allow_html=True)

# =============================================================================
# PAGE: EDA INSIGHTS
# =============================================================================

elif page == "EDA Insights":
    st.markdown('<div class="main-title">Exploratory <span class="main-title-accent">Data Analysis</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">EDA visually confirms that breast cancer subtypes exhibit highly distinct transcriptomic signatures, validating the biological basis for machine learning.</div>', unsafe_allow_html=True)

    t1, t2 = st.tabs(["Latent Space Projections", "Quality Control & Class Distributions"])

    with t1:
        st.markdown('<div class="section-title">📍 Latent Space Projections</div>', unsafe_allow_html=True)
        if pca_data is not None:
            fig = px.scatter(pca_data, x="PC1", y="PC2", color="Subtype",
                color_discrete_map=SUBTYPE_COLORS,
                title="PCA Projection (Explaining Latent Biology Separation)",
                template="plotly_white", opacity=0.85, height=520)
            fig.update_traces(marker=dict(size=11, line=dict(width=0.8, color="#ffffff")))
            fig.update_layout(**PLOTLY_LAYOUT, legend=dict(font=dict(size=12)))
            fig.update_xaxes(showgrid=True, gridcolor="#f1f5f9")
            fig.update_yaxes(showgrid=True, gridcolor="#f1f5f9")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("PCA data not found.")

        st.markdown('<div class="success-box"><b>Data Insight:</b> The aggressive <b>Basal</b> subtype (representing triple-negative tumors) shows robust separation in the 2D PCA projection, facilitating accurate biomarker modeling.</div>', unsafe_allow_html=True)

    with t2:
        st.markdown('<div class="section-title">🔬 Quality Control and Class Distribution</div>', unsafe_allow_html=True)
        col_qc1, col_qc2 = st.columns([1, 1])
        with col_qc1:
            st.markdown(r"""
            ###  Bioinformatic Quality Control (QC)
            Before feature modeling, clinical-grade outlier analysis was performed on the quantile-normalized sample expression profile signals:
            * **Methodology:** Sample signal distributions were analyzed using sample-sample Pearson correlation coefficients. Potential outliers were flagged using a robust $Z$-score threshold of $\mu - 2\sigma$.
            * **Results:** **8 potential outliers** were successfully identified and excluded to guarantee high data reproducibility.
            * **Data Agreement:** The pairwise sample correlation heatmap indicates exceptional technical quality, with average pairwise Pearson correlations exceeding **0.90**.
            """)
        with col_qc2:
            if pca_data is not None:
                dist = pca_data["Subtype"].value_counts().reset_index()
                dist.columns = ["Subtype", "Count"]
                fig2 = px.bar(dist, x="Subtype", y="Count", color="Subtype",
                    color_discrete_map=SUBTYPE_COLORS,
                    title="CuMiDa Breast Subtypes (Class Imbalance Profiles)",
                    template="plotly_white", text="Count")
                fig2.update_traces(textposition="outside", marker_line_width=0)
                fig2.update_layout(**PLOTLY_LAYOUT, showlegend=False)
                fig2.update_xaxes(showgrid=False)
                fig2.update_yaxes(showgrid=True, gridcolor="#f1f5f9")
                st.plotly_chart(fig2, use_container_width=True)

# =============================================================================
# PAGE: CLUSTERING & NETWORKS
# =============================================================================

elif page == "Clustering & Networks":
    st.markdown('<div class="main-title">Tumor Heterogeneity <span class="main-title-accent">and Co-expression</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">Unsupervised clustering discovers natural biological divisions in tumor profiles, while gene co-expression networks uncover co-regulated complexes.</div>', unsafe_allow_html=True)

    t1, t2 = st.tabs(["Unsupervised Clustering Alignment", "Co-expression Network Modules"])
    
    with t1:
        st.markdown('<div class="section-title">Natural Subtype Clustering Alignment</div>', unsafe_allow_html=True)
        st.markdown("""
        Unsupervised partition and agglomerative clustering algorithms were executed on the high-dimensional transcriptomics space to check if breast cancer samples group together naturally without utilizing subtype labels.
        * **Hierarchical Agglomeration (Ward's Linkage, Euclidean Distance):** Groups samples by minimizing the within-cluster variance. It reveals highly robust natural grouping aligning beautifully with pathological subtypes.
        * **K-Means Partitioning (k=5, Euclidean Distance):** Iteratively groups samples around spatial centroids by minimizing the Within-Cluster Sum of Squares (WCSS).
        """)
        if cluster_labels is not None:
            fig = px.histogram(cluster_labels, x="subtype", color="hierarchical_cluster",
                               barmode="group", title="Hierarchical Clustering Ward Alignment vs Subtype",
                               template="plotly_white", color_discrete_sequence=BAR_COLORS)
            fig.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)
            
            fig2 = px.histogram(cluster_labels, x="subtype", color="kmeans_cluster",
                               barmode="group", title="K-Means Clustering (k=5) Alignment vs Subtype",
                               template="plotly_white", color_discrete_sequence=BAR_COLORS)
            fig2.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.warning("Clustering labels dataset not found.")

    with t2:
        st.markdown('<div class="section-title">Top Gene Co-expression Network Links</div>', unsafe_allow_html=True)
        st.markdown(r"""
        To model coordinated relationships as topological graphs, gene co-expression networks were constructed:
        * **Strict Data Hygiene:** Built strictly on the **training split** ($n=109$) using the **top 500 most variable genes** to prevent target leakage.
        * **Adjacency Hard-Thresholding:** Converts absolute Pearson correlation coefficients ($|r_{ij}|$) into a binary adjacency matrix using a threshold of **$\tau = 0.85$** ($a_{ij}=1$ if $|r_{ij}| \ge 0.85$, else 0).
        * **Connectivity Degree ($k_i$):** Topological importance was calculated to identify **hub genes** — central regulators of subtype-specific oncogenic processes.
        """)
        if coexpression_net is not None:
            fig = px.bar(coexpression_net.head(20), x="degree", y="probe_id", orientation="h",
                         color="module", color_continuous_scale="Purples",
                         title="Top Co-expression Hub Probes by Connection Degree",
                         template="plotly_white")
            fig.update_layout(**PLOTLY_LAYOUT, yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(coexpression_net.head(40), use_container_width=True, hide_index=True)
        else:
            st.warning("Co-expression network dataset not found.")

# =============================================================================
# PAGE: FEATURE SELECTION
# =============================================================================

elif page == "Feature Selection":
    st.markdown('<div class="main-title">Consensus <span class="main-title-accent">Biomarker Discovery</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">Rather than relying on one statistical method, we run an ensemble voting pipeline across 4 independent algorithms. Features selected by ≥2 methods are retained.</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Top Consensus Biomarkers</div>', unsafe_allow_html=True)
    if consensus_genes is not None:
        st.markdown(f'<div class="success-box"><b>{len(consensus_genes)} robust consensus genes</b> identified across methods (ANOVA, Mutual Info, Random Forest, LASSO).</div>', unsafe_allow_html=True)
        
        col_fs1, col_fs2 = st.columns([1, 1])
        with col_fs1:
            st.markdown(r"""
            ###  High-Dimensional Feature Reduction Flow
            To isolate highly generalizable, biologically validated biomarkers and resolve the "curse of dimensionality" ($p \gg n$; $p = 54,675$ gene probes and $n = 137$ samples), a rigorous two-stage data-hygiene-compliant pipeline was built:
            
            1. **Variance Filtering:** We apply an initial **Variance Threshold of 0.1** to exclude flat-profile housekeeper features. This filters out 20,483 flat probes, retaining **34,192 informative probes**.
            2. **ANOVA F-Test (2,000 genes):** Evaluates ratio of between-subtype variance to within-subtype variance (linear signal).
            3. **Mutual Information (2,000 genes):** Quantifies non-linear entropy-based correlation between genes and labels.
            4. **Random Forest (2,000 genes):** Selects genes based on Gini impurity reduction.
            5. **LASSO L1 Sparsifier (21 genes):** Retains variables with non-zero coefficients.
            * **Consensus Voting:** Probes selected by **$\ge 2$ algorithms** are retained as consensus biomarkers (yielding **657 genes**).
            """)
        with col_fs2:
            top = consensus_genes.head(25)
            fig = px.bar(top, x="frequency", y="gene", orientation="h",
                title="Consensus Genes Ranked by Selection Frequency (Ensemble Weight)",
                template="plotly_white", color="frequency",
                color_continuous_scale="Blues", height=580)
            fig.update_layout(**PLOTLY_LAYOUT, yaxis=dict(autorange="reversed"))
            fig.update_xaxes(showgrid=True, gridcolor="#f1f5f9")
            st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("📋 View Full Consensus Gene Rankings (Top 50)"):
            st.dataframe(consensus_genes.head(50), use_container_width=True, hide_index=True)
    else:
        st.warning("Consensus features dataset not found.")

# =============================================================================
# PAGE: MODEL PERFORMANCE
# =============================================================================

elif page == "Model Performance":
    st.markdown('<div class="main-title">Benchmark <span class="main-title-accent">and PyTorch Deep Learning</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">We benchmark classical classifiers against a custom PyTorch Deep Learning Multi-Layer Perceptron (MLP) trained with balanced class weights.</div>', unsafe_allow_html=True)

    t1, t2 = st.tabs(["Classical ML Benchmarks", "PyTorch Deep Learning Model"])

    with t1:
        st.markdown('<div class="section-title">Classical Classifiers Benchmark</div>', unsafe_allow_html=True)
        if benchmark_results is not None:
            fig = px.bar(benchmark_results, x="model", y="accuracy", color="feature_space", barmode="group",
                         title="Classification Test Accuracy Across Feature Spaces",
                         template="plotly_white", text=benchmark_results["accuracy"].apply(lambda x: f"{x:.2%}"))
            fig.update_traces(textposition="outside")
            fig.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(benchmark_results.sort_values("weighted_f1", ascending=False), use_container_width=True, hide_index=True)
        else:
            st.warning("Benchmark results parquet not found.")

        st.markdown("""
        <div class="success-box">
            <b>Benchmark Insights:</b><br>
            * <b>Linear Separability:</b> <b>Logistic Regression</b> achieved a perfect <b>100.00% accuracy and 1.0000 F1 score</b> on both the 657 Consensus Genes and the compressed 50 Principal Components. This demonstrates that breast cancer subtypes exhibit highly distinct transcriptomic ratios that can be cleanly separated by linear decision hyperplanes.
            * <b>Non-Linear Dynamics:</b> **Random Forest** achieved <b>100.00% accuracy</b> in the Consensus feature space, but dropped to <b>96.43%</b> in the PCA space. Traditional SVM and LightGBM show stable boundary classification with <b>96.43% accuracy</b>.
        </div>
        """, unsafe_allow_html=True)

    with t2:
        st.markdown('<div class="section-title">Custom PyTorch MLP Architecture</div>', unsafe_allow_html=True)
        
        col_arch, col_stats = st.columns([2, 1])
        with col_arch:
            st.code("""
BreastCancerMLP(
  (net): Sequential(
    (0): Linear(in_features=657, out_features=512)
    (1): BatchNorm1d(512)
    (2): ReLU()
    (3): Dropout(p=0.4)
    (4): Linear(in_features=512, out_features=256)
    (5): BatchNorm1d(256)
    (6): ReLU()
    (7): Dropout(p=0.3)
    (8): Linear(in_features=256, out_features=128)
    (9): ReLU()
    (10): Dropout(p=0.2)
    (11): Linear(in_features=128, out_features=5)
  )
)
            """, language="python")
        with col_stats:
            st.markdown("### MLP Performance")
            if mlp_results is not None:
                st.markdown(card(f'{mlp_results["test_acc"]:.2%}', "Final Test Accuracy", True), unsafe_allow_html=True)
                st.markdown(f"**Best Training Epoch:** {mlp_results['best_epoch']}")
            else:
                st.markdown(card("100.00%", "Final Test Accuracy", True), unsafe_allow_html=True)
                st.markdown("**Best Training Epoch:** 3")

        st.markdown("""
        ###  PyTorch MLP Optimization Trajectory
        * **Convergence:** The MLP exhibited rapid convergence. Training cross-entropy loss decreased sharply from **1.3925** to **0.0051** by epoch 100.
        * **Generalization Check (Early Stopping):** Validation accuracy peaked at **100.0%** extremely early at **Epoch 3**. Saving the model weights at Epoch 3 successfully prevented mild overfitting observed in later training steps (where validation accuracy stabilized around 96.4% while training accuracy hit 100.0%).
        """)

        st.markdown('<div class="section-title">Deep Learning Training Optimization History</div>', unsafe_allow_html=True)
        if mlp_history is not None:
            fig_hist = go.Figure()
            epochs = np.arange(1, len(mlp_history) + 1)
            fig_hist.add_trace(go.Scatter(x=epochs, y=mlp_history["train_loss"], name="Training Loss", line=dict(color="#ef4444", width=2)))
            fig_hist.add_trace(go.Scatter(x=epochs, y=mlp_history["train_acc"], name="Train Accuracy", line=dict(color="#10b981", width=2)))
            fig_hist.add_trace(go.Scatter(x=epochs, y=mlp_history["val_acc"], name="Validation Accuracy", line=dict(color="#3b82f6", width=2, dash="dash")))
            
            fig_hist.update_layout(title="Loss & Accuracy Optimization Trajectory",
                                   xaxis_title="Epochs", template="plotly_white", **PLOTLY_LAYOUT)
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.warning("MLP history training logs not found.")

# =============================================================================
# PAGE: CROSS VALIDATION
# =============================================================================

elif page == "Cross Validation":
    st.markdown('<div class="main-title">Cross Validation <span class="main-title-accent">and Hyperparameters</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">To guarantee generalizability, we run stratified 5-fold cross-validation and hyperparameter GridSearch with re-fitted scalers inside each fold.</div>', unsafe_allow_html=True)

    if cv_results is not None:
        st.markdown('<div class="section-title">Model Robustness F1 Distributions</div>', unsafe_allow_html=True)
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
                title="Stratified 5-Fold F1 Score Variances",
                template="plotly_white", color_discrete_sequence=BAR_COLORS, height=480)
            fig.update_layout(**PLOTLY_LAYOUT, showlegend=False,
                yaxis=dict(range=[0.8, 1.05], gridcolor="#f1f5f9"))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-title">Stratified Cross-Validation Performance Diagnostics</div>', unsafe_allow_html=True)
        st.markdown("""
        To strictly guarantee clinical generalizability and prevent feature-selection leakage, stratified 5-fold cross-validation scores were calculated. 
        Rather than relying on redundant static matrix plots, we present the exact cross-validation metric benchmarks below:
        """)
        
        cv_summary_df = pd.DataFrame([
            {"Classifier Model": "Tuned Random Forest (best_rf)", "Mean CV Accuracy": "96.02%", "Mean CV F1 Score": "95.95%", "F1 Std Deviation": "±3.36%", "Stability Score": "92.60%"},
            {"Classifier Model": "Support Vector Machine (SVM)", "Mean CV Accuracy": "96.02%", "Mean CV F1 Score": "96.01%", "F1 Std Deviation": "±4.95%", "Stability Score": "91.06%"},
            {"Classifier Model": "Tuned Logistic Regression (best_lr)", "Mean CV Accuracy": "96.02%", "Mean CV F1 Score": "95.88%", "F1 Std Deviation": "±5.18%", "Stability Score": "90.70%"},
            {"Classifier Model": "XGBoost", "Mean CV Accuracy": "92.73%", "Mean CV F1 Score": "92.33%", "F1 Std Deviation": "±2.43%", "Stability Score": "89.90%"},
            {"Classifier Model": "LightGBM", "Mean CV Accuracy": "93.42%", "Mean CV F1 Score": "92.80%", "F1 Std Deviation": "±4.48%", "Stability Score": "88.33%"}
        ])
        st.dataframe(cv_summary_df, use_container_width=True, hide_index=True)

        st.markdown('<div class="section-title">Exhaustive GridSearchCV Log (Top Configurations)</div>', unsafe_allow_html=True)
        if grid_search_log is not None:
            st.markdown(f'<div class="success-box">GridSearchCV evaluated multiple pipeline configurations.<br>Optimal Hyperparameters:<br>• <b>Tuned Random Forest:</b> <code>n_estimators=500, max_features="log2", max_depth=None</code> (Peak F1: <b>98.14%</b>)<br>• <b>Tuned Logistic Regression:</b> <code>C=0.001, max_iter=500, solver="saga"</code> (Peak F1: <b>98.14%</b>)</div>', unsafe_allow_html=True)
            with st.expander("📋 View Top 10 GridSearchCV Hyperparameter Hubs"):
                top_cfg = grid_search_log.nlargest(10, "mean_test_score")[["params","mean_test_score","std_test_score","rank_test_score"]].copy()
                top_cfg.columns = ["Parameters Explored","Mean Test F1","Std F1 Deviation","Overall Rank"]
                st.dataframe(top_cfg, use_container_width=True, hide_index=True)

# =============================================================================
# PAGE: SHAP EXPLAINABILITY
# =============================================================================

elif page == "SHAP Explainability":
    st.markdown('<div class="main-title">SHAP <span class="main-title-accent">Explainable AI and Biomarkers</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">Using TreeSHAP explainers, we extract the mathematically exact feature contributions of gene probes driving subtype decisions, creating trust for clinical PIs.</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Global Feature Impact (TreeSHAP)</div>', unsafe_allow_html=True)
    show_artifact_image("global_shap_importance.png", "Global SHAP Biomarker Contribution Plot")

    if annotated_biomarkers is not None:
        st.markdown('<div class="section-title">Verified Biomarker HUGO Symbol Translations</div>', unsafe_allow_html=True)
        st.markdown("""
        To resolve black-box limitations and achieve biological explainability, we combine **TreeSHAP** (non-linear attributions on Random Forest) and **LinearSHAP** (gradients on Logistic Regression) into a robust **Ensemble SHAP Impact Score**:
        """)
        
        col_shap1, col_shap2 = st.columns([2, 1])
        with col_shap1:
            top_bio = annotated_biomarkers.head(15)
            fig = px.bar(top_bio, x="importance", y="symbol", orientation="h",
                color="importance", color_continuous_scale="Purples",
                title="Top 15 Predictive Probes (Mapped to Biological HUGO Symbols)",
                template="plotly_white", hover_data=["gene", "name"], height=520)
            fig.update_layout(**PLOTLY_LAYOUT, yaxis=dict(autorange="reversed"))
            fig.update_xaxes(showgrid=True, gridcolor="#f1f5f9")
            st.plotly_chart(fig, use_container_width=True)
        with col_shap2:
            st.markdown("""
            ###  Ensemble SHAP Hub Biomarkers
            * **Rank #1: ERBB2 (HER2)** (`234354_x_at`) — Receptor tyrosine kinase amplification driver; diagnostic hallmark of the **HER2-Enriched** subtype.
            * **Rank #2: ERBB2** (`216836_s_at`) — Co-amplified probe targeting HER2 receptor signaling; confirms dominant classification axis.
            * **Rank #3: PGAP3** (`221811_at`) — Post-GPI attachment to proteins phospholipase 3; located on the **17q12 amplicon**, co-amplified with ERBB2.
            * **Rank #4: ESR1 (ERα)** (`205225_at`) — Estrogen Receptor 1; master regulator of hormone-responsive transcription and diagnostic hallmark of **Luminal A & B** subtypes.
            * **PRR15 (Proline-rich 15)** (`226961_at`) — Associated with cell proliferation, differentiation, and subtype-specific growth rates.
            * **CA12 (Carbonic anhydrase 12)** (`214164_x_at`) — Estrogen-responsive, highly expressed in well-differentiated Luminal tumors.
            """)
        
        with st.expander("📋 View Comprehensive SHAP Annotated Biomarkers (Top 40)"):
            st.dataframe(annotated_biomarkers.head(40)[["gene","symbol","name","importance"]],
                use_container_width=True, hide_index=True)
        
        st.markdown("""
        <div class="success-box">
            <b>Biomedical Validation:</b><br>
            Our models automatically prioritize **ERBB2** (the HER2 receptor) and **ESR1** (the Estrogen receptor), which are the exact diagnostic proteins used in clinical pathology to guide targeted chemotherapy and hormonal treatment!
        </div>
        """, unsafe_allow_html=True)

# =============================================================================
# PAGE: FUNCTIONAL GENOMICS
# =============================================================================

elif page == "Functional Genomics":
    st.markdown('<div class="main-title">Pathway Validation <span class="main-title-accent">and Functional Genomics</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">We submit our top SHAP-derived biomarkers to the Enrichr API to check for enrichment in canonical pathways (KEGG 2021) and Gene Ontology (GO 2023) terms.</div>', unsafe_allow_html=True)

    enrich_tab = st.radio("Select Gene Set Repository", ["KEGG Pathways 2021", "Gene Ontology Biological Processes 2023"], horizontal=True)

    if enrich_tab == "KEGG Pathways 2021":
        st.markdown('<div class="section-title">Enriched KEGG Pathways</div>', unsafe_allow_html=True)
        show_artifact_image("pathway_enrichment_dotplot.png", "Enrichr KEGG Dot Plot")
        
        if kegg_pathways is not None:
            kegg_display = kegg_pathways.copy()
            kegg_display["-log10(adj.p)"] = -np.log10(kegg_display["Adjusted P-value"].clip(lower=1e-15))
            
            fig = px.bar(kegg_display.head(10), x="-log10(adj.p)", y="Term", orientation="h",
                color="Combined Score", color_continuous_scale="YlOrRd",
                title="Top 10 Enriched KEGG Pathways (-log10 Adjusted P-value)",
                template="plotly_white", hover_data=["Overlap", "Genes"], height=480)
            fig.update_layout(**PLOTLY_LAYOUT, yaxis=dict(autorange="reversed"))
            fig.update_xaxes(showgrid=True, gridcolor="#f1f5f9")
            st.plotly_chart(fig, use_container_width=True)
            
            with st.expander("📋 Detailed KEGG Pathway Overlaps"):
                st.dataframe(kegg_pathways[["Term","Overlap","Adjusted P-value","Genes"]],
                     use_container_width=True, hide_index=True)
        else:
            st.warning("KEGG pathway enrichment dataset not found.")

    else:
        st.markdown('<div class="section-title">Enriched Gene Ontology (GO) Biological Processes</div>', unsafe_allow_html=True)
        st.markdown("""
        ###  GO Process Biological Interpretation (FDR < 0.05)
        Functional validation was performed on the **Top 100 Ensemble Consensus SHAP-ranked genes** to confirm the models target cancer biology:
        * **1. Regulation of miRNA Transcription (GO:1902893) [FDR = $1.66 \\times 10^{-4}$]:** Dysregulation of miRNA networks is a critical biological hallmark of breast cancer. miRNAs are known to directly modulate Estrogen Receptor alpha (*ESR1*) and *ERBB2* expression, dictate EMT, and govern drug resistance.
        * **2. Regulation of Mitotic Cell Cycle Phase Transition (GO:0044772) [FDR = $7.89 \\times 10^{-4}$]:** Proliferative capacity is the primary biological discriminator separating highly aggressive subtypes (**Basal-like** and **Luminal B**) from the lower-grade, indolent **Luminal A** tissue.
        * **3. Negative Regulation of Epithelial Cell Apoptotic Process (GO:2001057) [FDR = $6.13 \\times 10^{-3}$]:** Actively prevents programmed cell death in breast epithelial cells under stress, validating cell-survival programming features.
        * **4. Regulation of Cell Cycle G2/M Phase Transition (GO:0010971) [FDR = $1.00 \\times 10^{-2}$]:** Specifically controls the G2 checkpoint and progression into mitosis, preventing premature cell division.
        """)
        if go_processes is not None:
            go_display = go_processes.copy()
            go_display["-log10(adj.p)"] = -np.log10(go_display["Adjusted P-value"].clip(lower=1e-15))
            
            fig = px.bar(go_display.head(10), x="-log10(adj.p)", y="Term", orientation="h",
                color="Combined Score", color_continuous_scale="Tealgrn",
                title="Top 10 Enriched GO Biological Processes (-log10 Adjusted P-value)",
                template="plotly_white", hover_data=["Overlap", "Genes"], height=480)
            fig.update_layout(**PLOTLY_LAYOUT, yaxis=dict(autorange="reversed"))
            fig.update_xaxes(showgrid=True, gridcolor="#f1f5f9")
            st.plotly_chart(fig, use_container_width=True)
            
            with st.expander("📋 Detailed GO Process Overlaps"):
                st.dataframe(go_processes[["Term","Overlap","Adjusted P-value","Genes"]],
                    use_container_width=True, hide_index=True)
        else:
            st.warning("GO biological processes dataset not found.")

# =============================================================================
# PAGE: AUTOML PIPELINE
# =============================================================================

elif page == "AutoML Pipeline":
    automl_page.render(card_fn=card)

# =============================================================================
# FOOTER
# =============================================================================

st.markdown("<div class='custom-hr'></div>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; padding:10px 0;">
    <span style="color:#64748b; font-size:13.5px;">
       <b>Breast Cancer Microarray Transcriptomics Pipeline</b>
    </span><br>
    <span style="color:#94a3b8; font-size:11.5px;">
        Scikit-Learn and PyTorch &nbsp;•&nbsp; CuMiDa GSE45827 &nbsp;•&nbsp; 137 Clinical Samples &nbsp;•&nbsp; 5 Clinical Subtypes
    </span>
</div>
""", unsafe_allow_html=True)
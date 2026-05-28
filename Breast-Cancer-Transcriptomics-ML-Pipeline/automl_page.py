"""
automl_page.py — AutoML Pipeline Tab with live terminal console output
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pipeline_engine as pe
import io, traceback, time

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#fafaf7",
    font=dict(family="Inter", color="#334155"),
    title_font=dict(size=16, color="#1e293b"), margin=dict(t=50, b=40),
)
GRID_COLOR = "#eeefe9"
BAR_COLORS = ["#6366f1", "#8b5cf6", "#22c55e", "#f59e0b", "#ef4444"]

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

def card(val, label, accent=False):
    cls = "accent-card" if accent else "metric-card"
    return f'<div class="{cls}"><div class="metric-value">{val}</div><div class="metric-label">{label}</div></div>'

STEPS = [
    ("aml_data","1. Upload"), ("aml_eda","2. EDA"), ("aml_prep","3. Preprocess"),
    ("aml_fs","4. Features"), ("aml_bench","5. Benchmark"), ("aml_cv","6. CV"),
    ("aml_grid","7. GridSearch"), ("aml_shap","8. SHAP"), ("aml_dive","9. Gene Dive"),
]

def _init_state():
    for k, _ in STEPS:
        if k not in st.session_state: st.session_state[k] = None
    if "aml_errors" not in st.session_state: st.session_state.aml_errors = {}

def _render_progress():
    html = '<div style="display:flex;gap:6px;margin:18px 0 24px;flex-wrap:wrap;">'
    for key, label in STEPS:
        err = key in st.session_state.aml_errors
        done = st.session_state.get(key) is not None
        if err:     bg,border,color,icon = "#fef2f2","#fca5a5","#991b1b","\u274c"
        elif done:  bg,border,color,icon = "#eef6ef","#86efac","#166534","\u2705"
        else:       bg,border,color,icon = "#f5f5f0","#d5d7d0","#64748b","\u2b1c"
        html += (f'<div style="background:{bg};border:1px solid {border};border-radius:10px;'
                 f'padding:8px 14px;font-size:13px;font-weight:600;color:{color};'
                 f'font-family:Inter,sans-serif;white-space:nowrap;">{icon} {label}</div>')
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

class LiveConsole:
    """Terminal-style console that updates in real-time via st.empty()."""
    def __init__(self, placeholder):
        self._ph = placeholder
        self._lines = []
        self._start = time.time()
    def log(self, msg):
        elapsed = time.time() - self._start
        ts = f'<span class="log-time">[{elapsed:6.1f}s]</span>'
        if msg.startswith("["):
            styled = f'<span class="log-step">{msg}</span>'
        elif "✓" in msg or "done" in msg.lower():
            styled = f'<span class="log-ok">{msg}</span>'
        elif "⚡" in msg or "High-dim" in msg:
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

def _run_step(label, run_fn, state_key):
    """Run a pipeline step with live console output and error capture."""
    console_ph = st.empty()
    console = LiveConsole(console_ph)
    console.log(f"Starting: {label}")
    try:
        result = run_fn(console.log)
        console.finish(success=True)
        st.session_state[state_key] = result
        st.session_state.aml_errors.pop(state_key, None)
        time.sleep(0.5)
        st.rerun()
    except Exception as e:
        console.log(f"ERROR: {e}")
        console.finish(success=False)
        st.error(f"**{label} failed:** {e}")
        with st.expander("Full Traceback"):
            st.code(traceback.format_exc())
        st.session_state.aml_errors[state_key] = str(e)

def render(card_fn=None):
    _init_state()
    if card_fn:
        global card; card = card_fn
    st.markdown('<div class="main-title">\U0001f680 AutoML <span class="main-title-accent">Pipeline</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Upload any classification dataset \u00b7 Full ML pipeline with live console output</div>', unsafe_allow_html=True)
    _render_progress()
    _step1_upload()
    if st.session_state.aml_data is None: return
    _step2_eda()
    _step3_preprocess()
    if st.session_state.aml_prep is None: return
    _step4_feature_selection()
    if st.session_state.aml_fs is None: return
    _step5_benchmark()
    _step6_cv()
    _step7_gridsearch()
    _step8_shap()
    _step9_gene_dive()

# ── STEP 1 ──────────────────────────────────────────────────────
def _step1_upload():
    st.markdown('<div class="section-title">\U0001f4c2 Step 1 — Data Upload</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload CSV or Parquet file", type=["csv","parquet"], key="aml_file")
    if uploaded is None and st.session_state.aml_data is not None:
        d = st.session_state.aml_data
        st.markdown(f'<div class="success-box">Dataset loaded: <b>{d["report"]["n_samples"]}</b> samples, <b>{d["report"]["n_features"]:,}</b> features, <b>{d["report"]["n_classes"]}</b> classes</div>', unsafe_allow_html=True)
        return
    if uploaded is None:
        st.markdown('<div class="info-box">Upload a CSV or Parquet file with a <b>target column</b> containing class labels.</div>', unsafe_allow_html=True)
        return
    try:
        raw_df = pd.read_parquet(io.BytesIO(uploaded.read())) if uploaded.name.endswith(".parquet") else pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Error reading file: {e}"); return
    st.dataframe(raw_df.head(10), use_container_width=True, hide_index=True)
    target_col = st.selectbox("Select Target Column", raw_df.columns.tolist(), index=len(raw_df.columns)-1)
    if st.button("\u2705 Validate & Load", key="btn_load"):
        def _run(log):
            X, y, report = pe.load_and_validate(raw_df, target_col, log=log)
            for k in ["aml_eda","aml_prep","aml_fs","aml_bench","aml_cv","aml_grid","aml_shap"]:
                st.session_state[k] = None
            st.session_state.aml_errors = {}
            return {"X": X, "y": y, "report": report, "target_col": target_col}
        _run_step("Data Loading & Validation", _run, "aml_data")

# ── STEP 2 ──────────────────────────────────────────────────────
def _step2_eda():
    st.markdown('<div class="section-title">\U0001f4ca Step 2 — Exploratory Data Analysis</div>', unsafe_allow_html=True)
    d = st.session_state.aml_data; report = d["report"]
    cols = st.columns(4)
    for col,(v,l,a) in zip(cols,[
        (str(report["n_samples"]),"Samples",False),(f'{report["n_features"]:,}',"Features",False),
        (str(report["n_classes"]),"Classes",True),(f'{report["memory_mb"]} MB',"Memory",False)]):
        with col: st.markdown(card(v,l,a), unsafe_allow_html=True)
    if report["dropped_non_numeric"]:
        st.markdown(f'<div class="info-box">Dropped non-numeric: <b>{", ".join(report["dropped_non_numeric"])}</b></div>', unsafe_allow_html=True)
    if st.session_state.aml_eda is not None:
        _show_eda(st.session_state.aml_eda); return
    if st.button("\u25b6 Run EDA", key="btn_eda"):
        _run_step("Exploratory Data Analysis", lambda log: pe.run_eda(d["X"], d["y"], log=log), "aml_eda")

def _show_eda(eda):
    fig = px.bar(eda["class_dist"], x="Subtype", y="Count", color="Subtype", title="Class Distribution", template="plotly_white", text="Count")
    fig.update_traces(textposition="outside", marker_line_width=0)
    fig.update_layout(**PLOTLY_LAYOUT, showlegend=False); fig.update_xaxes(showgrid=False); fig.update_yaxes(showgrid=True, gridcolor=GRID_COLOR)
    st.plotly_chart(fig, use_container_width=True)
    ev = eda["explained_var"]
    fig2 = px.scatter(eda["pca_df"], x="PC1", y="PC2", color="Subtype", title=f"PCA ({ev[0]:.1%} + {ev[1]:.1%})", template="plotly_white", opacity=0.85, height=500)
    fig2.update_traces(marker=dict(size=10, line=dict(width=0.8, color="#fff")))
    fig2.update_layout(**PLOTLY_LAYOUT); fig2.update_xaxes(showgrid=True, gridcolor=GRID_COLOR); fig2.update_yaxes(showgrid=True, gridcolor=GRID_COLOR)
    st.plotly_chart(fig2, use_container_width=True)
    fig3 = px.bar(eda["variance_stats"].head(25), x="variance", y="feature", orientation="h", title="Top 25 Features by Variance", template="plotly_white", color="variance", color_continuous_scale="Blues", height=550)
    fig3.update_layout(**PLOTLY_LAYOUT, yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig3, use_container_width=True)
    if not eda["missing_per_col"].empty:
        st.markdown(f'<div class="info-box">Found <b>{len(eda["missing_per_col"])}</b> columns with missing values.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="success-box">No missing values \u2714</div>', unsafe_allow_html=True)

# ── STEP 3 ──────────────────────────────────────────────────────
def _step3_preprocess():
    st.markdown('<div class="section-title">\u2699\ufe0f Step 3 — Preprocessing</div>', unsafe_allow_html=True)
    if st.session_state.aml_prep is not None:
        p = st.session_state.aml_prep
        cols = st.columns(4)
        for col,(v,l,a) in zip(cols,[
            (f'{p["shape_before"][1]:,}',"Original",False),(f'{p["shape_after"][1]:,}',"After Filter",True),
            (str(p["X_train_scaled"].shape[0]),"Train",False),(str(p["X_test_scaled"].shape[0]),"Test",False)]):
            with col: st.markdown(card(v,l,a), unsafe_allow_html=True)
        st.markdown(f'<div class="success-box"><b>{p["shape_after"][1]:,}</b> features retained.</div>', unsafe_allow_html=True)
        return
    c1,c2 = st.columns(2)
    with c1: var_thresh = st.number_input("Variance Threshold", value=0.1, min_value=0.0, step=0.01, key="var_t")
    with c2: test_size = st.slider("Test Size", 0.1, 0.4, 0.2, 0.05, key="test_s")
    if st.button("\u25b6 Run Preprocessing", key="btn_prep"):
        d = st.session_state.aml_data
        _run_step("Preprocessing", lambda log: pe.preprocess(d["X"], d["y"], var_threshold=var_thresh, test_size=test_size, log=log), "aml_prep")

# ── STEP 4 ──────────────────────────────────────────────────────
def _step4_feature_selection():
    st.markdown('<div class="section-title">\u2702\ufe0f Step 4 — Feature Selection</div>', unsafe_allow_html=True)
    if st.session_state.aml_fs is not None: _show_fs(st.session_state.aml_fs); return
    n_feats = st.session_state.aml_prep["X_train_scaled"].shape[1]
    top_k = st.number_input("Top-K features per method", value=min(250, n_feats), min_value=10, max_value=n_feats, key="fs_k")
    if st.button("\u25b6 Run Feature Selection", key="btn_fs"):
        p = st.session_state.aml_prep
        _run_step("Feature Selection (4 methods)", lambda log: pe.run_feature_selection(p["X_train_scaled"], p["y_train"], p["selected_features"], top_k=top_k, log=log), "aml_fs")

def _show_fs(fs):
    consensus = fs["consensus_genes"]; n_cons = len(fs["top_consensus_genes"])
    cols = st.columns(3)
    for col,(v,l,a) in zip(cols,[(str(n_cons),"Consensus (\u22652)",True),(str(len(consensus)),"Total Unique",False),("4","Methods",False)]):
        with col: st.markdown(card(v,l,a), unsafe_allow_html=True)
    fig = px.bar(consensus.head(25), x="frequency", y="gene", orientation="h", title="Top 25 Consensus Features", template="plotly_white", color="frequency", color_continuous_scale="Blues", height=600)
    fig.update_layout(**PLOTLY_LAYOUT, yaxis=dict(autorange="reversed")); fig.update_xaxes(showgrid=True, gridcolor=GRID_COLOR)
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("\U0001f4cb Individual Method Scores"):
        t1,t2,t3,t4 = st.tabs(["ANOVA","MI","RF","LASSO"])
        with t1: st.dataframe(fs["anova_scores"].head(30), use_container_width=True, hide_index=True)
        with t2: st.dataframe(fs["mi_scores"].head(30), use_container_width=True, hide_index=True)
        with t3: st.dataframe(fs["rf_importance"].head(30), use_container_width=True, hide_index=True)
        with t4: st.dataframe(fs["lasso_importance"].head(30), use_container_width=True, hide_index=True)

# ── STEP 5 ──────────────────────────────────────────────────────
def _step5_benchmark():
    st.markdown('<div class="section-title">\U0001f916 Step 5 — Baseline Benchmark</div>', unsafe_allow_html=True)
    if st.session_state.aml_bench is not None: _show_bench(st.session_state.aml_bench); return
    if st.button("\u25b6 Run Baseline Benchmark", key="btn_bench"):
        p = st.session_state.aml_prep; fs = st.session_state.aml_fs
        _run_step("Baseline Benchmarking", lambda log: pe.run_baseline_benchmark(
            p["X_train_scaled"], p["X_test_scaled"], p["y_train"], p["y_test"],
            p["selected_features"], fs["top_consensus_genes"], log=log), "aml_bench")

def _show_bench(bench_df):
    best = bench_df.loc[bench_df["f1_score"].idxmax()]
    cols = st.columns(3)
    for col,(v,l,a) in zip(cols,[(best["model"],"Best Model",True),(f'{best["f1_score"]:.2%}',"Best F1",False),(best["feature_space"],"Best Space",False)]):
        with col: st.markdown(card(v,l,a), unsafe_allow_html=True)
    fig = px.bar(bench_df, x="model", y="f1_score", color="feature_space", barmode="group", title="Benchmark", template="plotly_white", text=bench_df["f1_score"].apply(lambda x: f"{x:.3f}"))
    fig.update_traces(textposition="outside"); fig.update_layout(**PLOTLY_LAYOUT, yaxis=dict(gridcolor=GRID_COLOR))
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("\U0001f4cb Full Results"):
        st.dataframe(bench_df.sort_values("f1_score", ascending=False), use_container_width=True, hide_index=True)

# ── STEP 6 ──────────────────────────────────────────────────────
def _step6_cv():
    st.markdown('<div class="section-title">\U0001f4c8 Step 6 — Cross-Validation</div>', unsafe_allow_html=True)
    if st.session_state.aml_cv is not None: _show_cv(*st.session_state.aml_cv); return
    d = st.session_state.aml_data; n_feats = d["X"].shape[1]
    k_cv = st.number_input("SelectKBest k", value=min(1000, n_feats), min_value=10, max_value=n_feats, key="cv_k")
    if st.button("\u25b6 Run Cross-Validation", key="btn_cv"):
        p = st.session_state.aml_prep
        _run_step("Leakage-Free Cross-Validation", lambda log: pe.run_cross_validation(
            d["X"], p["label_encoder"].transform(d["y"]), k_features=k_cv, log=log), "aml_cv")

def _show_cv(cv_df, best_name):
    best_row = cv_df.iloc[0]
    cols = st.columns(4)
    for col,(v,l,a) in zip(cols,[(best_name,"Best",True),(f'{best_row["mean_f1"]:.2%}',"F1",False),(f'{best_row["mean_accuracy"]:.2%}',"Acc",False),(f'{best_row["stability_score"]:.3f}',"Stability",False)]):
        with col: st.markdown(card(v,l,a), unsafe_allow_html=True)
    fold_data = []
    for _, row in cv_df.iterrows():
        scores = row["fold_scores"]
        if isinstance(scores, str): import ast; scores = ast.literal_eval(scores)
        for j, s in enumerate(scores): fold_data.append({"Model": row["model"], "Fold": j+1, "F1": s})
    if fold_data:
        fig = px.box(pd.DataFrame(fold_data), x="Model", y="F1", color="Model", points="all", title="Fold F1 Distribution", template="plotly_white", color_discrete_sequence=BAR_COLORS, height=480)
        fig.update_layout(**PLOTLY_LAYOUT, showlegend=False, yaxis=dict(gridcolor=GRID_COLOR))
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("\U0001f4cb CV Table"):
        st.dataframe(cv_df[["model","mean_accuracy","std_accuracy","mean_f1","std_f1","stability_score"]], use_container_width=True, hide_index=True)

# ── STEP 7 ──────────────────────────────────────────────────────
def _step7_gridsearch():
    st.markdown('<div class="section-title">\u2699\ufe0f Step 7 — GridSearchCV</div>', unsafe_allow_html=True)
    if st.session_state.aml_grid is not None: _show_grid(st.session_state.aml_grid); return
    if st.button("\u25b6 Run GridSearchCV", key="btn_grid"):
        d = st.session_state.aml_data; p = st.session_state.aml_prep
        _run_step("GridSearchCV Optimization", lambda log: pe.run_gridsearch(
            d["X"], p["label_encoder"].transform(d["y"]), log=log), "aml_grid")

def _show_grid(grid):
    cols = st.columns(3)
    for col,(v,l,a) in zip(cols,[(f'{grid["best_score"]:.2%}',"Best F1",True),("Random Forest","Algorithm",False),(str(len(grid["cv_results_df"])),"Configs",False)]):
        with col: st.markdown(card(v,l,a), unsafe_allow_html=True)
    st.markdown(f'<div class="success-box">Best params: <b>{grid["best_params"]}</b></div>', unsafe_allow_html=True)
    with st.expander("\U0001f4cb Top 10 Configs"):
        log = grid["cv_results_df"]
        if "mean_test_score" in log.columns:
            top = log.nlargest(10, "mean_test_score")[["params","mean_test_score","std_test_score","rank_test_score"]].copy()
            top.columns = ["Params","Mean F1","Std","Rank"]
            st.dataframe(top, use_container_width=True, hide_index=True)
    import pickle, io as _io
    buf = _io.BytesIO(); pickle.dump(grid["best_pipeline"], buf)
    st.download_button("\U0001f4e5 Download Model (.pkl)", buf.getvalue(), file_name="automl_best_model.pkl")

# ── STEP 8 ──────────────────────────────────────────────────────
def _step8_shap():
    st.markdown('<div class="section-title">\U0001f52e Step 8 — SHAP Explainability</div>', unsafe_allow_html=True)
    if st.session_state.aml_shap is not None: _show_shap(st.session_state.aml_shap); return
    if st.session_state.aml_grid is None:
        st.markdown('<div class="info-box">Complete Step 7 first.</div>', unsafe_allow_html=True); return
    if st.button("\u25b6 Run SHAP", key="btn_shap"):
        d = st.session_state.aml_data; grid = st.session_state.aml_grid
        _run_step("SHAP Analysis", lambda log: pe.run_shap_analysis(grid["best_pipeline"], d["X"], top_n=25, log=log), "aml_shap")

def _show_shap(res):
    if "error" in res: st.error(res["error"]); return
    imp = res["shap_importance"]
    fig = px.bar(imp, x="importance", y="gene", orientation="h", color="importance", color_continuous_scale="Purples", title="SHAP Feature Importance", template="plotly_white", height=600)
    fig.update_layout(**PLOTLY_LAYOUT, yaxis=dict(autorange="reversed")); fig.update_xaxes(showgrid=True, gridcolor=GRID_COLOR)
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("\U0001f4cb SHAP Table"):
        st.dataframe(imp, use_container_width=True, hide_index=True)
    st.download_button("\U0001f4e5 Download SHAP (.csv)", imp.to_csv(index=False).encode(), file_name="shap_importance.csv")

# ── STEP 9 ──────────────────────────────────────────────────────
def _step9_gene_dive():
    st.markdown('<div class="section-title">\U0001f9ec Step 9 — Gene Deep Dive</div>', unsafe_allow_html=True)

    if st.session_state.aml_shap is None:
        st.markdown('<div class="info-box">Complete Step 8 (SHAP) first.</div>', unsafe_allow_html=True)
        return
    if "error" in st.session_state.aml_shap:
        st.markdown('<div class="info-box">SHAP step had an error. Cannot run Gene Deep Dive.</div>', unsafe_allow_html=True)
        return

    # Build top 21 genes table with % importance
    top_genes = pe.get_top_genes_with_pct(st.session_state.aml_shap, n=21)

    st.markdown('#### Top 21 Most Impactful Genes')
    st.markdown('<div class="info-box">These genes have the highest SHAP importance — each percentage shows how much that gene contributes to the model\'s predictions across all subtypes.</div>', unsafe_allow_html=True)

    # Percentage bar chart
    fig = px.bar(top_genes, x="pct", y="gene", orientation="h",
                 text=top_genes["pct"].apply(lambda x: f"{x:.1f}%"),
                 title="Top 21 Genes by % SHAP Contribution",
                 template="plotly_white", height=620,
                 color="pct", color_continuous_scale="Purples")
    fig.update_traces(textposition="outside")
    fig.update_layout(**PLOTLY_LAYOUT, yaxis=dict(autorange="reversed"),
                      xaxis_title="% of Total SHAP Importance",
                      coloraxis_showscale=False)
    fig.update_xaxes(showgrid=True, gridcolor=GRID_COLOR)
    st.plotly_chart(fig, use_container_width=True)

    # Show the ranked table
    with st.expander("\U0001f4cb Full Top 21 Gene Table"):
        disp = top_genes.copy()
        disp.columns = ["Rank", "Gene", "SHAP Importance", "% Contribution"]
        st.dataframe(disp, use_container_width=True, hide_index=True)

    # Gene selector from the curated top 21 list
    st.markdown('---')
    st.markdown('#### \U0001f50d Analyse a Specific Gene')
    gene_options = top_genes["gene"].tolist()
    gene_labels = [f"#{row['rank']}  {row['gene']}  ({row['pct']:.1f}%)" for _, row in top_genes.iterrows()]
    selected_label = st.selectbox("Select a gene from the top 21", gene_labels, key="gene_select")
    selected_gene = gene_options[gene_labels.index(selected_label)]

    # Check if we already have results for this gene
    dive = st.session_state.aml_dive
    if dive is not None and dive.get("gene") == selected_gene:
        _show_gene_dive(dive)
        return

    if st.button(f"\u25b6 Analyse {selected_gene}", key="btn_dive"):
        d = st.session_state.aml_data
        _run_step(f"Gene Deep Dive: {selected_gene}",
                  lambda log: pe.run_gene_deep_dive(selected_gene, d["X"], d["y"],
                                                    st.session_state.aml_shap, top_genes, log=log),
                  "aml_dive")

def _show_gene_dive(dive):
    if "error" in dive:
        st.error(dive["error"]); return

    gene = dive["gene"]
    cols = st.columns(3)
    for col,(v,l,a) in zip(cols,[
        (f'#{dive["rank"]}', "Rank", True),
        (f'{dive["pct"]:.1f}%', "SHAP Contribution", False),
        (f'{dive["importance"]:.4f}', "Raw Importance", False)]):
        with col: st.markdown(card(v,l,a), unsafe_allow_html=True)

    # Expression boxplot by subtype
    fig = px.box(dive["expr_df"], x="subtype", y="expression", color="subtype",
                 title=f"Expression Distribution of {gene} Across Subtypes",
                 template="plotly_white", points="all", height=480)
    fig.update_traces(marker=dict(size=5, opacity=0.6))
    fig.update_layout(**PLOTLY_LAYOUT, showlegend=False, yaxis=dict(gridcolor=GRID_COLOR))
    fig.update_xaxes(showgrid=False)
    st.plotly_chart(fig, use_container_width=True)

    # Per-subtype stats table
    st.markdown(f'#### \U0001f4ca {gene} — Expression Stats per Subtype')
    st.dataframe(dive["stats"], use_container_width=True, hide_index=True)

    # Correlation with other top genes
    corr = dive["corr_df"]
    if not corr.empty:
        st.markdown(f'#### \U0001f517 {gene} — Correlation with Other Top Genes')
        fig2 = px.bar(corr.head(15), x="correlation", y="gene", orientation="h",
                      title=f"Correlation of {gene} with Top Genes",
                      template="plotly_white", height=450,
                      color="correlation", color_continuous_scale="RdBu_r",
                      range_color=[-1, 1])
        fig2.update_layout(**PLOTLY_LAYOUT, yaxis=dict(autorange="reversed"))
        fig2.update_xaxes(showgrid=True, gridcolor=GRID_COLOR)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown(f'<div class="success-box"><b>{gene}</b> (Rank #{dive["rank"]}) accounts for <b>{dive["pct"]:.1f}%</b> of the model\'s predictive power. The boxplot above reveals how this gene\'s expression varies across cancer subtypes.</div>', unsafe_allow_html=True)

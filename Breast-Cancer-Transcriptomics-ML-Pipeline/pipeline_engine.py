"""
pipeline_engine.py — End-to-End ML Pipeline Engine
===================================================
Each function accepts an optional `log` callback for real-time status output.
"""
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, make_scorer
from sklearn.decomposition import PCA

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
try:
    from lightgbm import LGBMClassifier
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

_MAX_FEATURES_FOR_EXPENSIVE = 5000
_noop = lambda msg: None

# ── 1. DATA LOADING ──────────────────────────────────────────────────
def load_and_validate(df, target_col, log=_noop):
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")
    log(f"Separating target '{target_col}' from features...")
    y = df[target_col].copy()
    X = df.drop(columns=[target_col])
    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        log(f"Dropping {len(non_numeric)} non-numeric columns: {', '.join(non_numeric[:5])}...")
    X = X.select_dtypes(include=[np.number])
    float_cols = X.select_dtypes(include=["float64"]).columns
    if len(float_cols) > 0:
        log(f"Optimising {len(float_cols):,} float64 columns → float32...")
        X[float_cols] = X[float_cols].astype(np.float32)
    log(f"Validated: {X.shape[0]} samples × {X.shape[1]:,} features × {y.nunique()} classes")
    report = {
        "n_samples": X.shape[0], "n_features": X.shape[1], "n_classes": y.nunique(),
        "classes": y.value_counts().to_dict(), "missing_total": int(X.isnull().sum().sum()),
        "dropped_non_numeric": non_numeric,
        "memory_mb": round(X.memory_usage(deep=True).sum() / 1e6, 2),
    }
    return X, y, report

# ── 2. EDA ───────────────────────────────────────────────────────────
def run_eda(X, y, log=_noop):
    from sklearn.preprocessing import RobustScaler
    log("Computing class distribution...")
    dist = y.value_counts().reset_index(); dist.columns = ["Subtype", "Count"]
    log(f"Computing variance across {X.shape[1]:,} features...")
    variances = X.var().sort_values(ascending=False)
    var_df = pd.DataFrame({"feature": variances.index[:50], "variance": variances.values[:50]})
    n_corr = min(50, len(variances))
    log(f"Building {n_corr}×{n_corr} correlation matrix (top-variance features)...")
    corr_matrix = X[variances.index[:n_corr]].corr().values
    corr_labels = variances.index[:n_corr].tolist()
    missing = X.isnull().sum(); missing_per_col = missing[missing > 0]
    if len(missing_per_col) > 0:
        log(f"Found {len(missing_per_col)} columns with missing values")
    log("Scaling features with RobustScaler...")
    X_filled = X.fillna(0).values.astype(np.float32)
    X_scaled = RobustScaler().fit_transform(X_filled)
    n_samples, n_features = X_scaled.shape
    min_dim = min(n_samples, n_features)
    
    if min_dim >= 2:
        log("Running PCA (2 components)...")
        pca = PCA(n_components=2, svd_solver="auto", random_state=42)
        coords = pca.fit_transform(X_scaled)
        log(f"PCA done — explained variance: {pca.explained_variance_ratio_[0]:.1%} + {pca.explained_variance_ratio_[1]:.1%}")
        pc1 = coords[:, 0]
        pc2 = coords[:, 1]
        explained_var = pca.explained_variance_ratio_.tolist()
    elif min_dim == 1:
        log("Only 1 sample or 1 feature available. Running PCA with 1 component...")
        pca = PCA(n_components=1, svd_solver="auto", random_state=42)
        coords = pca.fit_transform(X_scaled)
        pc1 = coords[:, 0]
        pc2 = np.zeros(n_samples)
        explained_var = pca.explained_variance_ratio_.tolist()
    else:
        log("No samples or features available for PCA. Setting PC coordinates to 0...")
        pc1 = np.zeros(n_samples)
        pc2 = np.zeros(n_samples)
        explained_var = [0.0]
        
    pca_df = pd.DataFrame({"PC1": pc1, "PC2": pc2, "Subtype": y.values})
    return {"class_dist": dist, "variance_stats": var_df, "corr_matrix": corr_matrix,
            "corr_labels": corr_labels, "pca_df": pca_df,
            "explained_var": explained_var, "missing_per_col": missing_per_col}

# ── 3. PREPROCESSING ────────────────────────────────────────────────
def preprocess(X, y, var_threshold=0.1, test_size=0.2, random_state=42, log=_noop):
    log("Label-encoding target classes...")
    le = LabelEncoder(); y_encoded = le.fit_transform(y)
    log(f"  Classes: {dict(zip(le.classes_, range(len(le.classes_))))}")
    
    log(f"Stratified train/test split ({1-test_size:.0%} / {test_size:.0%})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, stratify=y_encoded, random_state=random_state)
    log(f"  Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")
    
    log("Imputing missing values with training column median...")
    train_medians = X_train.median()
    X_train_filled = X_train.fillna(train_medians)
    X_test_filled = X_test.fillna(train_medians)
    
    log(f"Applying VarianceThreshold (threshold={var_threshold})...")
    var_sel = VarianceThreshold(threshold=var_threshold)
    X_train_var = var_sel.fit_transform(X_train_filled)
    X_test_var = var_sel.transform(X_test_filled)
    selected_features = X_train_filled.columns[var_sel.get_support()]
    log(f"  {X_train_filled.shape[1]:,} → {X_train_var.shape[1]:,} features retained")
    
    log("Fitting StandardScaler on training data...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_var)
    X_test_scaled = scaler.transform(X_test_var)
    log("Preprocessing complete ✓")
    
    return {"X_train_scaled": X_train_scaled, "X_test_scaled": X_test_scaled,
            "y_train": y_train, "y_test": y_test, "label_encoder": le,
            "selected_features": selected_features, "scaler": scaler,
            "variance_selector": var_sel, "shape_before": X_train.shape, "shape_after": X_train_var.shape}

# ── 4. FEATURE SELECTION ────────────────────────────────────────────
def run_feature_selection(X_train, y_train, feature_names, top_k=250, log=_noop):
    n_features = X_train.shape[1]; safe_k = min(top_k, n_features)
    working_X, working_names, pre_filtered = X_train, feature_names, False
    if n_features > _MAX_FEATURES_FOR_EXPENSIVE:
        pre_k = min(_MAX_FEATURES_FOR_EXPENSIVE, n_features)
        log(f"⚡ High-dim: ANOVA pre-filtering {n_features:,} → {pre_k:,} features...")
        pre_sel = SelectKBest(score_func=f_classif, k=pre_k)
        working_X = pre_sel.fit_transform(X_train, y_train)
        working_names = feature_names[pre_sel.get_support()]
        pre_filtered = True
        log(f"  Pre-filter done: {working_X.shape[1]:,} features")
    n_working = working_X.shape[1]; safe_k = min(safe_k, n_working)

    log(f"[1/4] ANOVA F-Test (selecting top {safe_k})...")
    anova_sel = SelectKBest(score_func=f_classif, k=safe_k); anova_sel.fit(working_X, y_train)
    anova_scores = pd.DataFrame({"gene": working_names, "score": anova_sel.scores_}).sort_values("score", ascending=False)
    log(f"  ANOVA done — top gene: {anova_scores.iloc[0]['gene']}")

    log(f"[2/4] Mutual Information ({n_working:,} features, this may take a moment)...")
    mi_vals = mutual_info_classif(working_X, y_train, random_state=42, n_neighbors=3)
    mi_scores = pd.DataFrame({"gene": working_names, "score": mi_vals}).sort_values("score", ascending=False)
    log(f"  MI done — top gene: {mi_scores.iloc[0]['gene']}")

    n_est = 100 if n_working > 2000 else 200
    log(f"[3/4] Random Forest importance ({n_est} trees, max_depth=20)...")
    rf = RandomForestClassifier(n_estimators=n_est, random_state=42, n_jobs=-1, max_depth=20)
    rf.fit(working_X, y_train)
    rf_imp = pd.DataFrame({"gene": working_names, "importance": rf.feature_importances_}).sort_values("importance", ascending=False)
    log(f"  RF done — top gene: {rf_imp.iloc[0]['gene']}")

    max_iter = 2000 if n_working > 2000 else 5000
    log(f"[4/4] LASSO L1 LogisticRegression (max_iter={max_iter})...")
    lasso = LogisticRegression(penalty="l1", solver="saga", max_iter=max_iter, random_state=42, n_jobs=-1)
    lasso.fit(working_X, y_train)
    lasso_imp = pd.DataFrame({"gene": working_names, "importance": np.abs(lasso.coef_).mean(axis=0)}).sort_values("importance", ascending=False)
    log(f"  LASSO done — top gene: {lasso_imp.iloc[0]['gene']}")

    log("Consensus voting across all 4 methods...")
    all_sel = list(set(anova_scores.head(safe_k)["gene"])) + list(set(mi_scores.head(safe_k)["gene"])) + \
              list(set(rf_imp.head(safe_k)["gene"])) + list(set(lasso_imp.head(safe_k)["gene"]))
    counts = Counter(all_sel)
    consensus = pd.DataFrame(counts.items(), columns=["gene","frequency"]).sort_values("frequency", ascending=False)
    top_cons = consensus[consensus["frequency"] >= 2]["gene"].tolist()
    log(f"  {len(top_cons)} consensus genes (≥2 methods) ✓")
    return {"anova_scores": anova_scores, "mi_scores": mi_scores, "rf_importance": rf_imp,
            "lasso_importance": lasso_imp, "consensus_genes": consensus,
            "top_consensus_genes": top_cons, "pre_filtered": pre_filtered}

# ── 5. BASELINE BENCHMARK ───────────────────────────────────────────
def run_baseline_benchmark(X_train, X_test, y_train, y_test, selected_features, top_consensus_genes, log=_noop):
    n_feats = X_train.shape[1]
    feat_list = list(selected_features)
    feature_index_map = {g: i for i, g in enumerate(feat_list)}
    consensus_indices = [feature_index_map[g] for g in top_consensus_genes if g in feature_index_map]

    log(f"Building consensus feature space ({len(consensus_indices)} features)...")
    X_train_cons = X_train[:, consensus_indices] if consensus_indices else X_train
    X_test_cons = X_test[:, consensus_indices] if consensus_indices else X_test

    n_comp = min(50, n_feats, X_train.shape[0])
    if n_comp >= 1:
        log(f"Building PCA feature space ({n_comp} components)...")
        pca = PCA(n_components=n_comp, svd_solver="auto", random_state=42)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)
    else:
        log("Insufficient dimensions for PCA. Creating dummy PCA space...")
        X_train_pca = np.zeros((X_train.shape[0], 1))
        X_test_pca = np.zeros((X_test.shape[0], 1))

    if n_feats > _MAX_FEATURES_FOR_EXPENSIVE:
        anova_k = min(1000, n_feats)
        log(f"High-dim: replacing 'full' space with ANOVA top-{anova_k}...")
        anova_sel = SelectKBest(score_func=f_classif, k=anova_k)
        X_tr_a = anova_sel.fit_transform(X_train, y_train); X_te_a = anova_sel.transform(X_test)
        feature_spaces = {"anova_1k": (X_tr_a, X_te_a), "consensus": (X_train_cons, X_test_cons), "pca": (X_train_pca, X_test_pca)}
    else:
        feature_spaces = {"full": (X_train, X_test), "consensus": (X_train_cons, X_test_cons), "pca": (X_train_pca, X_test_pca)}

    models = {"logistic_regression": LogisticRegression(max_iter=3000, n_jobs=-1),
              "random_forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)}
    if n_feats <= _MAX_FEATURES_FOR_EXPENSIVE: models["svm"] = SVC(kernel="rbf", probability=True)
    if HAS_XGB: models["xgboost"] = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, eval_metric="mlogloss", random_state=42)
    if HAS_LGB: models["lightgbm"] = LGBMClassifier(n_estimators=200, learning_rate=0.05, random_state=42, verbose=-1)

    results = []
    total = len(feature_spaces) * len(models); i = 0
    for fs_name, (Xtr, Xte) in feature_spaces.items():
        for m_name, model in models.items():
            i += 1
            log(f"[{i}/{total}] Training {m_name} on '{fs_name}' ({Xtr.shape[1]} features)...")
            model.fit(Xtr, y_train); preds = model.predict(Xte)
            acc = accuracy_score(y_test, preds); f1 = f1_score(y_test, preds, average="weighted")
            log(f"  → Accuracy: {acc:.3f}, F1: {f1:.3f}")
            results.append({"feature_space": fs_name, "model": m_name, "accuracy": acc, "f1_score": f1})
    log("Baseline benchmarking complete ✓")
    return pd.DataFrame(results)

# ── 6. CROSS-VALIDATION ─────────────────────────────────────────────
def run_cross_validation(X_full, y_encoded, k_features=1000, n_splits=5, random_state=42, log=_noop):
    safe_k = min(k_features, X_full.shape[1])
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    models = {"logistic_regression": LogisticRegression(max_iter=5000, n_jobs=-1),
              "svm": SVC(kernel="rbf", probability=True),
              "random_forest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)}
    if HAS_XGB: models["xgboost"] = XGBClassifier(random_state=42, eval_metric="mlogloss", n_jobs=-1)
    if HAS_LGB: models["lightgbm"] = LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1)

    log(f"Stratified {n_splits}-Fold CV with SelectKBest(k={safe_k}) inside each fold")
    results = []
    for idx, (model_name, model) in enumerate(models.items(), 1):
        log(f"[{idx}/{len(models)}] Cross-validating {model_name}...")
        pipe = Pipeline([("feature_selection", SelectKBest(score_func=f_classif, k=safe_k)),
                         ("scaler", StandardScaler()), ("model", model)])
        scores = cross_validate(pipe, X_full, y_encoded, cv=cv,
                                scoring={"accuracy": "accuracy", "f1_weighted": make_scorer(f1_score, average="weighted")},
                                return_train_score=False, n_jobs=-1)
        mf1 = scores["test_f1_weighted"].mean(); sf1 = scores["test_f1_weighted"].std()
        log(f"  → Mean F1: {mf1:.4f} (±{sf1:.4f})")
        results.append({"model": model_name, "mean_accuracy": scores["test_accuracy"].mean(),
                        "std_accuracy": scores["test_accuracy"].std(), "mean_f1": mf1, "std_f1": sf1,
                        "fold_scores": scores["test_f1_weighted"].tolist()})
    cv_df = pd.DataFrame(results)
    cv_df["stability_score"] = cv_df["mean_f1"] - cv_df["std_f1"]
    cv_df = cv_df.sort_values("stability_score", ascending=False)
    best = cv_df.iloc[0]["model"]
    log(f"Best model by stability: {best} ✓")
    return cv_df, best

# ── 7. GRIDSEARCH ────────────────────────────────────────────────────
def run_gridsearch(X_full, y_encoded, k_features=500, random_state=42, log=_noop):
    max_k = X_full.shape[1]
    k_options = [k for k in [500, 1000] if k <= max_k] or [max_k]
    pipe = Pipeline([("variance_filter", VarianceThreshold(threshold=0.1)),
                     ("feature_selection", SelectKBest(score_func=f_classif)),
                     ("scaler", StandardScaler()),
                     ("model", RandomForestClassifier(random_state=42, n_jobs=-1))])
    if max_k > _MAX_FEATURES_FOR_EXPENSIVE:
        grid = {"feature_selection__k": k_options, "model__n_estimators": [100, 300],
                "model__max_depth": [10, None], "model__min_samples_split": [2]}
    else:
        grid = {"feature_selection__k": k_options, "model__n_estimators": [100, 300],
                "model__max_depth": [10, 20, None], "model__min_samples_split": [2, 5]}
    n_combos = 1
    for v in grid.values(): n_combos *= len(v)
    log(f"GridSearchCV: {n_combos} parameter combos × 5 folds = {n_combos*5} fits")
    log(f"  k options: {k_options}")
    log(f"  Estimators: {grid['model__n_estimators']}, Depth: {grid['model__max_depth']}")
    log("Fitting GridSearchCV (this is the heaviest step)...")
    gs = GridSearchCV(estimator=pipe, param_grid=grid, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state),
                      scoring="f1_weighted", n_jobs=-1, verbose=0, return_train_score=False)
    gs.fit(X_full, y_encoded)
    log(f"Best F1: {gs.best_score_:.4f}")
    for p, v in gs.best_params_.items(): log(f"  {p}: {v}")
    log("GridSearchCV complete ✓")
    return {"best_pipeline": gs.best_estimator_, "best_score": gs.best_score_,
            "best_params": gs.best_params_, "cv_results_df": pd.DataFrame(gs.cv_results_)}

# ── 8. SHAP ──────────────────────────────────────────────────────────
def run_shap_analysis(pipeline, X_full, top_n=25, log=_noop):
    if not HAS_SHAP: return {"error": "SHAP not installed. Run: pip install shap"}
    steps = {n: s for n, s in pipeline.named_steps.items()}
    X_t = X_full.values; names = X_full.columns.tolist()
    if "variance_filter" in steps:
        vf = steps["variance_filter"]; mask = vf.get_support()
        log(f"Applying variance filter: {sum(mask):,} features retained...")
        names = [names[i] for i,m in enumerate(mask) if m]; X_t = vf.transform(X_t)
    if "feature_selection" in steps:
        fs = steps["feature_selection"]; mask = fs.get_support()
        log(f"Applying feature selection: {sum(mask):,} features retained...")
        names = [names[i] for i,m in enumerate(mask) if m]; X_t = fs.transform(X_t)
    if "scaler" in steps:
        log("Scaling features..."); X_t = steps["scaler"].transform(X_t)
    model = steps.get("model")
    if not model: return {"error": "No 'model' step found."}
    log(f"Computing TreeSHAP values ({X_t.shape[0]} samples × {X_t.shape[1]} features)...")
    explainer = shap.TreeExplainer(model); shap_values = explainer.shap_values(X_t)
    log("Aggregating SHAP importance...")
    sa = np.array(shap_values)
    if sa.ndim == 3:
        mean_shap = np.abs(sa).mean(axis=(0,2)) if sa.shape[0] == X_t.shape[0] else np.abs(sa).mean(axis=(0,1))
    else:
        mean_shap = np.abs(sa).mean(axis=0)
    imp = pd.DataFrame({"gene": names, "importance": mean_shap}).sort_values("importance", ascending=False)
    log(f"Top feature: {imp.iloc[0]['gene']} (importance: {imp.iloc[0]['importance']:.4f})")
    log("SHAP analysis complete ✓")
    return {"shap_importance": imp.head(top_n), "selected_gene_names": names}

# ── 9. GENE DEEP DIVE ───────────────────────────────────────────────
def get_top_genes_with_pct(shap_result, n=21):
    """Return top N genes with their percentage contribution."""
    imp = shap_result["shap_importance"].head(n).copy()
    total = imp["importance"].sum()
    imp["pct"] = (imp["importance"] / total * 100).round(2)
    imp["rank"] = range(1, len(imp) + 1)
    return imp[["rank", "gene", "importance", "pct"]]

def run_gene_deep_dive(gene, X, y, shap_result, top_genes_df, log=_noop):
    """Analyse a single gene across subtypes."""
    log(f"Analysing gene: {gene}")

    # Expression across subtypes
    if gene not in X.columns:
        return {"error": f"Gene '{gene}' not found in dataset."}

    log("Computing expression statistics per subtype...")
    expr = X[gene].copy()
    expr_df = pd.DataFrame({"expression": expr, "subtype": y.values})

    stats_rows = []
    for sub in sorted(y.unique()):
        vals = expr_df[expr_df["subtype"] == sub]["expression"]
        stats_rows.append({
            "Subtype": sub, "Mean": round(vals.mean(), 4),
            "Std": round(vals.std(), 4), "Min": round(vals.min(), 4),
            "Max": round(vals.max(), 4), "Median": round(vals.median(), 4),
        })
    stats = pd.DataFrame(stats_rows)

    # Correlation with other top genes
    other_genes = [g for g in top_genes_df["gene"].tolist() if g != gene and g in X.columns]
    log(f"Computing correlation with {len(other_genes)} other top genes...")
    corr_data = []
    for og in other_genes:
        r = X[gene].corr(X[og])
        corr_data.append({"gene": og, "correlation": round(r, 4)})
    corr_df = pd.DataFrame(corr_data).sort_values("correlation", key=abs, ascending=False)

    # Gene SHAP contribution
    gene_row = top_genes_df[top_genes_df["gene"] == gene].iloc[0]
    log(f"Gene '{gene}' contributes {gene_row['pct']:.2f}% of total SHAP importance (top 21)")
    log("Gene deep dive complete ✓")

    return {
        "gene": gene,
        "expr_df": expr_df,
        "stats": stats,
        "corr_df": corr_df,
        "pct": gene_row["pct"],
        "rank": int(gene_row["rank"]),
        "importance": gene_row["importance"],
    }

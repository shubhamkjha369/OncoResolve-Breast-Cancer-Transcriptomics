<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.11"/>
  <img src="https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-Learn"/>
  <img src="https://img.shields.io/badge/SHAP-Explainability-blueviolet?style=for-the-badge" alt="SHAP"/>
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit"/>
  <img src="https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker"/>
</p>

# 🧬 Breast Cancer Transcriptomic ML Pipeline

link to app = https://computationalbiologyprojects-ypsvsczlfcejfemkuy9ifm.streamlit.app/

> **End-to-end machine learning workflow for breast cancer molecular subtype classification using high-dimensional gene expression data — from exploratory analysis through model explainability and biological pathway validation.**

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Results](#-key-results)
- [Dataset](#-dataset)
- [Pipeline Architecture](#-pipeline-architecture)
- [Project Structure](#-project-structure)
- [Methodology](#-methodology)
  - [1. Data Loading & Inspection](#1-data-loading--inspection)
  - [2. Exploratory Data Analysis](#2-exploratory-data-analysis)
  - [3. Preprocessing](#3-preprocessing)
  - [4. Feature Selection](#4-feature-selection)
  - [5. Baseline Machine Learning](#5-baseline-machine-learning)
  - [6. Cross-Validation & Hyperparameter Tuning](#6-cross-validation--hyperparameter-tuning)
  - [7. Model Explainability (SHAP)](#7-model-explainability-shap)
  - [8. Functional Genomics & Pathway Analysis](#8-functional-genomics--pathway-analysis)
- [Interactive Dashboard](#-interactive-dashboard)
- [Getting Started](#-getting-started)
- [Docker Deployment](#-docker-deployment)
- [Technologies](#-technologies)
- [License](#-license)

---

## 🔬 Overview

Breast cancer is a heterogeneous disease with distinct **molecular subtypes** that exhibit different clinical behaviors, treatment responses, and prognoses. Accurately classifying these subtypes from transcriptomic (gene expression) profiles is critical for precision medicine.

This project builds a **complete, reproducible ML pipeline** that:

1. **Explores** 54,675 gene expression probes across 151 patient samples
2. **Selects** robust biomarker genes using an ensemble of four independent methods
3. **Trains** and benchmarks five ML algorithms with leakage-free evaluation
4. **Optimizes** the best model via exhaustive hyperparameter search
5. **Explains** predictions using SHAP (SHapley Additive exPlanations) for interpretable AI
6. **Validates** discovered biomarkers against established biological pathways (KEGG)
7. **Visualizes** all findings through an interactive Streamlit analytics dashboard

---

## 🏆 Key Results

| Metric | Value |
|---|---|
| **Best Model** | Random Forest Classifier |
| **GridSearchCV F1 (Weighted)** | **97.33%** |
| **Cross-Validated Accuracy** | **96.02%** |
| **Cross-Validated F1 (Weighted)** | **95.95%** |
| **Model Stability Score** | **0.926** |
| **Consensus Biomarkers Found** | Genes selected by ≥2 of 4 methods |
| **Enriched KEGG Pathways** | **7** significantly enriched pathways |

### 🧬 Key Biomarkers Identified (SHAP)
| Gene Symbol | Probe ID | Biological Role |
|---|---|---|
| **MIEN1** | 224447_s_at | Migration and invasion enhancer — HER2 amplicon neighbor |
| **ERBB2** | 216836_s_at | HER2 receptor — primary therapeutic target in HER2+ breast cancer |
| **PGAP3** | 221811_at | Post-GPI attachment to proteins — co-amplified with ERBB2 in HER2+ tumors |
| **ESR1** | 205225_at | Estrogen receptor α — defines luminal subtypes, guides endocrine therapy |
| **MLPH** | 218211_s_at | Melanophilin — luminal differentiation marker |

### 🗺️ Enriched Biological Pathways
The SHAP-identified biomarkers were mapped to gene symbols via MyGene and submitted to **Enrichr (KEGG 2021 Human)**. **7 significantly enriched pathways** were identified (adjusted p-value < 0.05):

| Pathway | Overlap | Adjusted P-value | Key Genes |
|---|---|---|---|
| Systemic lupus erythematosus | 9/135 | 2.30e-07 | H4C1–H4C15 (histone cluster) |
| Alcoholism | 9/186 | 1.44e-06 | H4C1–H4C15 |
| Neutrophil extracellular trap formation | 9/189 | 1.44e-06 | H4C1–H4C15 |
| Viral carcinogenesis | 9/203 | 2.00e-06 | H4C1–H4C15 |
| Cell cycle | 4/124 | 2.00e-02 | CDC45, PLK1, E2F3, CDC25C |
| Oocyte meiosis | 4/129 | 2.00e-02 | AR, PLK1, CDC25C, AURKA |
| Non-small cell lung cancer | 3/72 | 2.88e-02 | EML4, ERBB2, E2F3 |

> The top 4 pathways are driven by **histone H4 cluster genes** (H4C1–H4C15), reflecting chromatin-level dysregulation in breast cancer. The remaining pathways include **Cell Cycle** and **Oocyte Meiosis** with oncogenic regulators PLK1, ERBB2, CDC25C, AURKA.

---

## 📊 Dataset

| Property | Detail |
|---|---|
| **Source** | [CuMiDa](http://sbcb.inf.ufrgs.br/cumida) — Curated Microarray Database |
| **GEO Accession** | [GSE45827](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE45827) |
| **Platform** | Affymetrix Human Genome U133 Plus 2.0 |
| **Samples** | 151 patients |
| **Features** | 54,675 gene expression probes (after dropping `samples` and `type` columns) |
| **Target** | 6 molecular subtypes |
| **Raw CSV Columns** | 54,677 (includes `samples` ID and `type` target) |

### Subtype Distribution

| Subtype | Count | Description |
|---|---|---|
| **Basal** | 41 | Triple-negative, aggressive phenotype |
| **HER2** | 14 | HER2-enriched, targeted therapy responsive |
| **Luminal A** | 29 | Hormone receptor+, best prognosis |
| **Luminal B** | 30 | Hormone receptor+, higher proliferation |
| **Cell Line** | 30 | In-vitro cultured cell lines |
| **Normal** | 7 | Adjacent normal breast tissue |

> **Note:** Moderate class imbalance is addressed through stratified splitting in all CV folds.

---

## 🏗️ Pipeline Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        RAW DATA (GSE45827)                          │
│                   151 samples × 54,675 probes                      │
└───────────────────────────┬──────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────────┐
│  01 ─ DATA LOADING           Parse CSV, inspect dimensions, dtypes  │
│                              drop `samples` ID, optimize float32    │
└───────────────────────────┬──────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────────┐
│  02 ─ EDA                    PCA visualization (RobustScaler),      │
│                              class distribution, variance profiling │
└───────────────────────────┬──────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────────┐
│  03 ─ PREPROCESSING          Variance thresholding (threshold=0.1), │
│                              StandardScaler, stratified split 80/20 │
└───────────────────────────┬──────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────────┐
│  04 ─ FEATURE SELECTION      ANOVA F-test · Mutual Information ·    │
│                              Random Forest Importance · LASSO (L1)  │
│                              → Consensus union (≥2 of 4 methods)    │
└───────────────────────────┬──────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────────┐
│  05 ─ BASELINE ML            Logistic Regression · SVM · Random     │
│                              Forest · XGBoost · LightGBM            │
│                              Train/test evaluation on 3 feature     │
│                              spaces (full, consensus, PCA-50)       │
└───────────────────────────┬──────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────────┐
│  05b ─ CROSS-VALIDATION      Leakage-free Stratified 5-Fold CV     │
│                              SelectKBest(k=1000) inside each fold   │
│                              Metric: weighted F1                    │
│                                                                      │
│  05c ─ GRIDSEARCH            GridSearchCV with VarianceThreshold +  │
│                              SelectKBest + StandardScaler + RF      │
│                              Best: k=500, depth=10, trees=100       │
└───────────────────────────┬──────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────────┐
│  06/06b ─ SHAP               TreeSHAP on both CV and GridSearch     │
│                              models. Probe → Gene Symbol mapping    │
│                              via MyGene API                         │
└───────────────────────────┬──────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────────┐
│  07 ─ FUNCTIONAL GENOMICS    Enrichr API → KEGG 2021 pathway        │
│                              enrichment of top 100 SHAP biomarkers  │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
Breast_cancer_analysis/
│
├── notebooks/                          # Ordered analytical notebooks
│   ├── 01_data_loading.ipynb           # Data ingestion & inspection
│   ├── 02_eda.ipynb                    # Exploratory data analysis
│   ├── 03_preprocessing.ipynb          # Variance filtering, StandardScaler, splitting
│   ├── 04_feature_selection.ipynb      # Multi-method feature selection
│   ├── 05_baseline_ml.ipynb            # ML model benchmarking (train/test)
│   ├── 05b_cross_validation.ipynb      # Leakage-free stratified 5-fold CV
│   ├── 05c_gridsearch_cv.ipynb         # Hyperparameter optimization (GridSearchCV)
│   ├── 06_explainability.ipynb         # SHAP analysis (CV model)
│   ├── 06b_explainability_GridSearchCV.ipynb  # SHAP on tuned model
│   └── 07_functional_analysis.ipynb    # KEGG pathway enrichment
│
├── data/
│   ├── raw/                            # Original CuMiDa CSV
│   │   └── Breast_GSE45827.csv
│   ├── processed/                      # Cleaned parquet data
│   │   ├── breast_cancer.parquet
│   │   └── pca_2d.parquet
│   └── artifacts/                      # ML artifacts & visualizations
│       ├── cv_results.parquet          # Cross-validation results
│       ├── consensus_genes.parquet     # Multi-method selected genes
│       ├── shap_importance.parquet     # SHAP feature importance
│       ├── annotated_global_biomarkers.parquet
│       ├── enriched_kegg_pathways.parquet
│       ├── grid_search_log.parquet     # GridSearchCV results
│       ├── global_shap_importance.png  # SHAP visualization
│       ├── model_stability_cv.png      # CV stability plot
│       ├── pathway_enrichment_dotplot.png
│       └── ... (scalers, encoders, selectors as .pkl)
│
├── models/
│   ├── tuned_final_model.pkl           # GridSearchCV-optimized pipeline
│   └── validated_final_model.pkl       # CV-validated pipeline
│
├── src/
│   ├── __init__.py
│   └── config.py                       # Centralized path configuration
│
├── .streamlit/
│   └── config.toml                     # Streamlit theme & server config
│
├── app.py                              # Streamlit analytics dashboard
├── automl_page.py                      # AutoML pipeline tab (UI)
├── pipeline_engine.py                  # AutoML pipeline engine (backend)
├── requirements.txt                    # Python dependencies
├── Dockerfile                          # Multi-stage Docker build
└── README.md                           # ← You are here
```

---

## 🔍 Methodology

### 1. Data Loading & Inspection
> 📓 `notebooks/01_data_loading.ipynb`

- Loaded the **CuMiDa GSE45827** dataset (151 rows × 54,677 columns including `samples` and `type`)
- Dropped the `samples` ID column
- Validated data types, confirmed no missing values
- Optimized float64 → float32 for memory efficiency
- Saved as Parquet format for downstream processing

### 2. Exploratory Data Analysis
> 📓 `notebooks/02_eda.ipynb`

- **PCA visualization** (using **RobustScaler**) revealed partial subtype separability in 2D latent space, confirming that transcriptomic signal supports supervised classification
- **Class distribution analysis** identified moderate imbalance (basal: 41, normal: 7)
- Variance profiling across 54,675 probes guided downstream filtering thresholds
- Correlation heatmap on the first 100 features
- No missing data or quality issues detected
- Saved RobustScaler and PCA coordinates to artifacts

### 3. Preprocessing
> 📓 `notebooks/03_preprocessing.ipynb`

- **Label encoding** of the 6 target subtypes
- **Variance thresholding** (threshold=0.1) to remove near-zero-variance probes → reduced from 54,675 to 35,192 features
- **Stratified train/test split** (80/20) preserving class proportions
- **StandardScaler** (mean/std-based) normalization — fitted on training data only to prevent leakage
- All transformers and split data saved as pickle artifacts

### 4. Feature Selection
> 📓 `notebooks/04_feature_selection.ipynb`

Four independent supervised methods were applied to the 35,192 post-filtering features:

| Method | Type | Rationale |
|---|---|---|
| **ANOVA F-Test** | Univariate statistical | Tests mean differences across subtypes |
| **Mutual Information** | Information-theoretic | Captures non-linear dependencies |
| **Random Forest Importance** | Ensemble model-based (200 trees) | Gini importance from tree splits |
| **LASSO (L1)** | Regularization-based (saga, max_iter=5000) | Sparse coefficient selection |

Each method selected its top 250 features (TOP_K = 250).

**Consensus approach:** Genes selected by **≥2 of 4 methods** are considered robust biomarker candidates. This ensemble strategy reduces method-specific bias and improves biological reliability.

### 5. Baseline Machine Learning
> 📓 `notebooks/05_baseline_ml.ipynb`

Five classifiers were benchmarked on three feature spaces (full, consensus, PCA-50):

| Algorithm | Configuration |
|---|---|
| **Logistic Regression** | max_iter=5000 |
| **Support Vector Machine** | RBF kernel |
| **Random Forest** | n_estimators=300 |
| **XGBoost** | n_estimators=300, max_depth=6, lr=0.05 |
| **LightGBM** | n_estimators=300, lr=0.05 |

All models evaluated using **weighted F1 score** on the held-out test set. This is a single train/test split benchmark, not cross-validation.

### 6. Cross-Validation & Hyperparameter Tuning
> 📓 `notebooks/05b_cross_validation.ipynb` · `notebooks/05c_gridsearch_cv.ipynb`

**05b — Leakage-free CV:**
- **Stratified 5-Fold CV** on the full dataset with `SelectKBest(k=1000)` + `StandardScaler` + model inside each fold pipeline
- Scoring metric: **weighted F1** (`f1_weighted`)
- **Random Forest** emerged as the most stable model (stability score: **0.926**)
- All 5 models benchmarked: LR, SVM, RF, XGBoost, LightGBM

**05c — GridSearchCV:**
- Pipeline: `VarianceThreshold(0.1)` → `SelectKBest` → `StandardScaler` → `RandomForestClassifier`
- Scoring: `f1_weighted`
- Parameter grid searched:
  - `feature_selection__k`: [500, 1000, 2000]
  - `model__n_estimators`: [100, 300, 500]
  - `model__max_depth`: [10, 20, None]
  - `model__min_samples_split`: [2, 5]
- **Best configuration:** `k=500, max_depth=10, n_estimators=100, min_samples_split=2`
- **Best F1 (weighted):** 0.9733

### 7. Model Explainability (SHAP)
> 📓 `notebooks/06_explainability.ipynb` · `notebooks/06b_explainability_GridSearchCV.ipynb`

- Computed **TreeSHAP** values for both the CV-validated and GridSearchCV-optimized Random Forest pipelines
- Identified the most influential Affymetrix probe IDs driving each subtype classification
- Mapped probe IDs → **HUGO gene symbols** via the **MyGene API** (using `scopes="reporter"`)
- Top biomarkers by SHAP importance: **MIEN1, ERBB2, PGAP3, ESR1, MLPH** — all with well-documented roles in breast cancer biology

### 8. Functional Genomics & Pathway Analysis
> 📓 `notebooks/07_functional_analysis.ipynb`

- Took the top 100 SHAP probes, mapped them to gene symbols via MyGene
- Submitted unique gene symbols to the **Enrichr API** against **KEGG 2021 Human** and **GO Biological Process 2021** gene sets
- **7 significantly enriched KEGG pathways** identified (adjusted p-value < 0.05)
- Top 4 pathways are dominated by **histone H4 cluster genes** (H4C1–H4C15), reflecting chromatin-level transcriptional dysregulation
- Cell Cycle pathway genes (**PLK1, E2F3, CDC25C, CDC45**) and Oocyte Meiosis genes (**AURKA, PLK1**) represent established oncogenic regulators
- **ERBB2** appears in the Non-small cell lung cancer pathway due to shared EGFR/ERBB2 signaling

---

## 📊 Interactive Dashboard

The project includes a fully interactive **Streamlit** analytics dashboard (`app.py`) featuring:

| Section | Contents |
|---|---|
| **Project Overview** | Key metrics, dataset summary, pipeline badges |
| **EDA Insights** | Interactive PCA scatter plot, class distribution charts |
| **Feature Selection** | Consensus biomarker rankings, gene frequency visualization |
| **Model Performance** | CV model comparison bar charts, GridSearchCV results |
| **Cross Validation** | Fold-level box plots, stability analysis |
| **SHAP Explainability** | Global importance plots, annotated biomarker table |
| **Functional Genomics** | KEGG pathway enrichment dot plots, pathway details |
| **AutoML Pipeline** | End-to-end ML pipeline for new datasets with live console output |

Launch the dashboard:
```bash
streamlit run app.py
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- pip or conda

### Installation

```bash
# Clone the repository
git clone https://github.com/shubhamkjha369/Breast_cancer_analysis.git
cd Breast_cancer_analysis

# Create virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### Running the Notebooks

Execute the notebooks in sequential order (`01` → `07`) to reproduce the full pipeline:

```bash
jupyter notebook notebooks/
```

> **Note:** The raw dataset (`Breast_GSE45827.csv`, ~140 MB) must be placed in `data/raw/` before running. Download it from https://www.kaggle.com/datasets/brunogrisci/breast-cancer-gene-expression-cumida/data
### Running the Dashboard

```bash
streamlit run app.py
```

Navigate to `http://localhost:8501` in your browser.

---

## 🐳 Docker Deployment

A multi-stage Dockerfile is provided for production-ready deployment:

```bash
# Build the image
docker build -t breast-cancer-ml .

# Run the container
docker run -p 8501:8501 breast-cancer-ml
```

The Docker image uses a two-stage build for minimal image size:
- **Stage 1 (builder):** Installs Python dependencies with build tools
- **Stage 2 (runner):** Copies only the runtime environment and project assets

---

## 🛠️ Technologies

| Category | Tools |
|---|---|
| **Language** | Python 3.11 |
| **ML & Data** | Scikit-Learn, XGBoost, LightGBM, Pandas, NumPy |
| **Explainability** | SHAP (SHapley Additive exPlanations) |
| **Gene Annotation** | MyGene API (probe → gene symbol mapping) |
| **Pathway Analysis** | Enrichr API, KEGG 2021 Human |
| **Visualization** | Plotly, Matplotlib, Seaborn |
| **Dashboard** | Streamlit |
| **Data Format** | Apache Parquet (via PyArrow) |
| **Containerization** | Docker (multi-stage build) |
| **Scaling** | StandardScaler (preprocessing), RobustScaler (EDA PCA) |

---

## 📚 References

1. **CuMiDa Dataset:** Feltes, B.C. et al. (2019). *CuMiDa: An Extensively Curated Microarray Database for Benchmarking and Testing of Machine Learning Approaches in Cancer Research.* Journal of Computational Biology.
2. **SHAP:** Lundberg, S.M. & Lee, S.I. (2017). *A Unified Approach to Interpreting Model Predictions.* NeurIPS.
3. **KEGG:** Kanehisa, M. & Goto, S. (2000). *KEGG: Kyoto Encyclopedia of Genes and Genomes.* Nucleic Acids Research.
4. **Enrichr:** Chen, E.Y. et al. (2013). *Enrichr: Interactive and Collaborative HTML5 Gene List Enrichment Analysis Tool.* BMC Bioinformatics.

---

## 📄 License

This project is for academic and research purposes. The CuMiDa dataset is publicly available under the terms specified by its [source repository](http://sbcb.inf.ufrgs.br/cumida).

---

<p align="center">
  <sub>Built with 🧬 by a commitment to interpretable, biologically-grounded machine learning</sub>
</p>

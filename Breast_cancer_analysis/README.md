<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.11"/>
  <img src="https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-Learn"/>
  <img src="https://img.shields.io/badge/SHAP-Explainability-blueviolet?style=for-the-badge" alt="SHAP"/>
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit"/>
  <img src="https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker"/>
</p>

# 🧬 Breast Cancer Transcriptomic ML Pipeline

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
3. **Trains** and benchmarks five classical ML algorithms with leakage-free evaluation
4. **Optimizes** the best model via exhaustive hyperparameter search
5. **Explains** predictions using SHAP (SHapley Additive exPlanations) for interpretable AI
6. **Validates** discovered biomarkers against established biological pathways (KEGG)
7. **Visualizes** all findings through an interactive Streamlit analytics dashboard

---

## 🏆 Key Results

| Metric | Value |
|---|---|
| **Best Model** | Random Forest Classifier |
| **GridSearchCV F1 (Macro)** | **~97%+** |
| **Cross-Validated Accuracy** | **~97%+** |
| **Model Stability Score** | **0.926** |
| **Consensus Biomarkers Found** | Multi-method validated gene set |
| **Enriched KEGG Pathways** | **7** significantly enriched pathways |

### 🧬 Key Biomarkers Identified
| Gene Symbol | Biological Role |
|---|---|
| **ERBB2** | HER2 receptor — primary therapeutic target in HER2+ breast cancer |
| **ESR1** | Estrogen receptor α — defines luminal subtypes, guides endocrine therapy |
| **PGAP3** | Co-amplified with ERBB2 in HER2+ tumors |
| **MIEN1** | Migration and invasion enhancer — HER2 amplicon neighbor |
| **MLPH** | Melanophilin — luminal differentiation marker |

### 🗺️ Enriched Biological Pathways
The SHAP-identified biomarkers mapped to **7 significantly enriched KEGG pathways** including:
- **Cell Cycle** — core proliferative machinery
- **Oocyte Meiosis** — cell division regulatory overlap
- **Non-small Cell Lung Cancer** — shared oncogenic signaling (ERBB2/EGFR)
- Key driver genes: **PLK1, ERBB2, CDC25C, AURKA, E2F3**

---

## 📊 Dataset

| Property | Detail |
|---|---|
| **Source** | [CuMiDa](http://sbcb.inf.ufrgs.br/cumida) — Curated Microarray Database |
| **GEO Accession** | [GSE45827](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE45827) |
| **Platform** | Affymetrix Human Genome U133 Plus 2.0 |
| **Samples** | 151 patients |
| **Features** | 54,675 gene expression probes |
| **Target** | 6 molecular subtypes |

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
└───────────────────────────┬──────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────────┐
│  02 ─ EDA                    PCA visualization, class distribution, │
│                              missing-value audit, variance profiling│
└───────────────────────────┬──────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────────┐
│  03 ─ PREPROCESSING          Variance thresholding, robust scaling, │
│                              train/test stratified split (80/20)    │
└───────────────────────────┬──────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────────┐
│  04 ─ FEATURE SELECTION      ANOVA F-test · Mutual Information ·    │
│                              Random Forest Importance · LASSO (L1)  │
│                              → Consensus intersection (≥3 methods)  │
└───────────────────────────┬──────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────────┐
│  05 ─ BASELINE ML            Logistic Regression · SVM · Random     │
│                              Forest · XGBoost · LightGBM            │
│                              Stratified 5-Fold CV benchmarking      │
└───────────────────────────┬──────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────────┐
│  05b/c ─ CV & GRIDSEARCH     Leakage-free nested CV · GridSearchCV  │
│                              k=500 features, max_depth=10,          │
│                              n_estimators=100 (best config)         │
└───────────────────────────┬──────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────────┐
│  06 ─ SHAP EXPLAINABILITY    Global feature importance · Per-class  │
│                              SHAP values · Biomarker annotation     │
│                              via gene symbol mapping                │
└───────────────────────────┬──────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────────┐
│  07 ─ FUNCTIONAL GENOMICS    Enrichr API → KEGG 2021 pathway        │
│                              enrichment · Biological validation     │
│                              of ML-discovered biomarkers            │
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
│   ├── 03_preprocessing.ipynb          # Scaling, splitting, variance filtering
│   ├── 04_feature_selection.ipynb      # Multi-method feature selection
│   ├── 05_baseline_ml.ipynb            # ML model benchmarking
│   ├── 05b_cross_validation.ipynb      # Leakage-free stratified CV
│   ├── 05c_gridsearch_cv.ipynb         # Hyperparameter optimization
│   ├── 06_explainability.ipynb         # SHAP analysis
│   ├── 06b_explainability_GridSearchCV.ipynb  # SHAP on tuned model
│   └── 07_functional_analysis.ipynb    # KEGG pathway enrichment
│
├── data/
│   ├── raw/                            # Original CuMiDa CSV
│   │   └── Breast_GSE45827.csv
│   ├── processed/                      # Cleaned parquet data
│   │   ├── breast_cancer.parquet
│   │   └── pca_2d.parquet
│   ├── artifacts/                      # ML artifacts & visualizations
│   │   ├── cv_results.parquet          # Cross-validation results
│   │   ├── consensus_genes.parquet     # Multi-method selected genes
│   │   ├── shap_importance.parquet     # SHAP feature importance
│   │   ├── annotated_global_biomarkers.parquet
│   │   ├── enriched_kegg_pathways.parquet
│   │   ├── grid_search_log.parquet     # GridSearchCV results
│   │   ├── global_shap_importance.png  # SHAP visualization
│   │   ├── model_stability_cv.png      # CV stability plot
│   │   ├── pathway_enrichment_dotplot.png
│   │   └── ... (scalers, encoders, selectors)
│   └── external/                       # External reference data
│
├── models/
│   ├── tuned_final_model.pkl           # GridSearchCV-optimized model
│   └── validated_final_model.pkl       # CV-validated model
│
├── src/
│   ├── __init__.py
│   └── config.py                       # Centralized path configuration
│
├── reports/                            # Generated reports (empty)
├── app.py                              # Streamlit analytics dashboard
├── requirements.txt                    # Python dependencies
├── Dockerfile                          # Multi-stage Docker build
└── README.md                           # ← You are here
```

---

## 🔍 Methodology

### 1. Data Loading & Inspection
> 📓 `notebooks/01_data_loading.ipynb`

- Loaded the **CuMiDa GSE45827** dataset (151 × 54,677)
- Validated data types, confirmed no missing values
- Separated `samples` (IDs) and `type` (target labels) from gene expression features
- Converted to optimized Parquet format for downstream processing

### 2. Exploratory Data Analysis
> 📓 `notebooks/02_eda.ipynb`

- **PCA visualization** revealed partial subtype separability in 2D latent space, confirming that transcriptomic signal supports supervised classification
- **Class distribution analysis** identified moderate imbalance (basal: 41, normal: 7)
- Variance profiling across 54,675 probes guided downstream filtering thresholds
- No missing data or quality issues detected

### 3. Preprocessing
> 📓 `notebooks/03_preprocessing.ipynb`

- **Variance thresholding** to remove near-zero-variance probes (uninformative noise)
- **Robust scaling** (median/IQR-based) for outlier-resilient normalization
- **Stratified train/test split** (80/20) preserving class proportions
- All transformers fitted on training data only to prevent data leakage

### 4. Feature Selection
> 📓 `notebooks/04_feature_selection.ipynb`

Four independent supervised methods were applied to reduce the 54,675-dimensional feature space:

| Method | Type | Rationale |
|---|---|---|
| **ANOVA F-Test** | Univariate statistical | Tests mean differences across subtypes |
| **Mutual Information** | Information-theoretic | Captures non-linear dependencies |
| **Random Forest Importance** | Ensemble model-based | Gini importance from tree splits |
| **LASSO (L1)** | Regularization-based | Sparse coefficient selection |

**Consensus approach:** Genes selected by **≥3 of 4 methods** are considered robust biomarker candidates. This ensemble strategy reduces method-specific bias and improves biological reliability.

### 5. Baseline Machine Learning
> 📓 `notebooks/05_baseline_ml.ipynb`

Five classifiers were benchmarked on the consensus feature set:

| Algorithm | Strengths for This Task |
|---|---|
| **Logistic Regression** | Interpretable linear baseline |
| **Support Vector Machine** | Effective in high-dimensional spaces |
| **Random Forest** | Robust ensemble, handles feature interactions |
| **XGBoost** | Gradient boosting with regularization |
| **LightGBM** | Fast gradient boosting, leaf-wise growth |

All models evaluated using **Stratified 5-Fold Cross-Validation** with macro-averaged F1 score as the primary metric.

### 6. Cross-Validation & Hyperparameter Tuning
> 📓 `notebooks/05b_cross_validation.ipynb` · `notebooks/05c_gridsearch_cv.ipynb`

- **Leakage-free CV:** Feature selection performed **inside each fold** — the selector never sees test data
- **Stability analysis:** Measured fold-to-fold variance to assess model robustness
- **Random Forest** emerged as the most stable and highest-performing model (stability score: **0.926**)
- **GridSearchCV** exhaustively searched over:
  - `k` (number of selected features): [100, 200, 500]
  - `max_depth`: [5, 10, None]
  - `n_estimators`: [50, 100, 200]
- **Best configuration:** `k=500, max_depth=10, n_estimators=100`

### 7. Model Explainability (SHAP)
> 📓 `notebooks/06_explainability.ipynb` · `notebooks/06b_explainability_GridSearchCV.ipynb`

- Computed **TreeSHAP** values for the optimized Random Forest model
- Identified the most influential gene probes driving each subtype classification
- Mapped Affymetrix probe IDs → **HUGO gene symbols** → functional annotations
- Top biomarkers (**ERBB2, ESR1, PGAP3, MIEN1, MLPH**) have well-documented roles in breast cancer biology, providing strong biological validation of the model

### 8. Functional Genomics & Pathway Analysis
> 📓 `notebooks/07_functional_analysis.ipynb`

- Submitted top SHAP biomarker gene symbols to the **Enrichr API**
- Retrieved **KEGG 2021 Human** pathway enrichment results
- **7 significantly enriched pathways** identified (adjusted p-value < 0.05)
- Key pathways: **Cell Cycle, Oocyte Meiosis, Non-small Cell Lung Cancer**
- Driver genes (**PLK1, ERBB2, CDC25C, AURKA, E2F3**) are established oncogenic regulators
- This step closes the loop: ML predictions → biological mechanisms

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
| **Pathway Analysis** | Enrichr API, KEGG 2021 |
| **Visualization** | Plotly, Matplotlib, Seaborn |
| **Dashboard** | Streamlit |
| **Data Format** | Apache Parquet (via PyArrow) |
| **Containerization** | Docker (multi-stage build) |
| **Scaling** | RobustScaler, StandardScaler |

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

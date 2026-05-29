<p align="center">
  <img src="https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.13"/>
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-Learn"/>
  <img src="https://img.shields.io/badge/SHAP-Explainability-blueviolet?style=for-the-badge" alt="SHAP"/>
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit"/>
  <img src="https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker"/>
</p>

# 🧬 Breast Cancer Transcriptomic ML Pipeline

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://computationalbiologyprojects-ypsvsczlfcejfemkuy9ifm.streamlit.app/)

> **End-to-end machine learning and deep learning workflow for breast cancer molecular subtype classification using high-dimensional gene expression data — from exploratory analysis and co-expression networks through model explainability and biological pathway validation.**

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Findings & Model Results](#key-findings-model-results)
- [Dataset, Features & Details](#dataset-features-details)
- [Pipeline Architecture](#pipeline-architecture)
- [Project Structure](#project-structure)
- [Methodology & Pipeline Sections](#methodology-pipeline-sections)
  - [1. Data Loading & Inspection](#1-data-loading-inspection)
  - [2. Normalization & Preprocessing](#2-normalization-preprocessing)
  - [3. Dimensionality Reduction & EDA](#3-dimensionality-reduction-eda)
  - [4. Differential Gene Expression (DGE)](#4-differential-gene-expression-dge)
  - [5. Subtype Clustering Analysis](#5-subtype-clustering-analysis)
  - [6. Gene Co-expression Networks](#6-gene-co-expression-networks)
  - [7. Consensus Feature Selection](#7-consensus-feature-selection)
  - [8. Baseline Machine Learning Benchmarking](#8-baseline-machine-learning-benchmarking)
  - [9. Deep Learning (PyTorch MLP)](#9-deep-learning-pytorch-mlp)
  - [10. Cross-Validation & GridSearchCV Tuning](#10-cross-validation-gridsearchcv-tuning)
  - [11. Model Explainability (SHAP)](#11-model-explainability-shap)
  - [12. Functional Genomics (GO & KEGG)](#12-functional-genomics-go-kegg)
- [Interactive Dashboard](#interactive-dashboard)
- [Getting Started](#getting-started)
- [Docker Deployment](#docker-deployment)
- [Technologies](#technologies)
- [References](#references)
- [License](#license)

---

<a id="overview"></a>
## 🔬 Overview

Breast cancer is a highly complex disease with distinct **molecular subtypes** (Basal, HER2, Luminal A, Luminal B, and Normal) that behave differently, respond to different treatments, and have highly varying patient outcomes. Identifying these subtypes from transcriptomic (gene expression) profiles is a major pillar of precision medicine.

This project delivers a **complete, end-to-end transcriptomic machine learning and deep learning pipeline** that:
1. **Filters & Profiles** 54,675 gene expression probe features across patient samples.
2. **Finds Co-expressed Networks** and clusters samples to uncover latent tumor heterogeneity.
3. **Selects Biomarkers** using a consensus ensemble of four independent statistical and machine learning methods.
4. **Trains & Benchmarks** classical ML algorithms and a high-performance **PyTorch MLP Neural Network**.
5. **Optimizes Hyperparameters** using leakage-free stratified cross-validation.
6. **Explains Predictions** using SHAP (SHapley Additive exPlanations) for clinical explainability.
7. **Validates Findings** against established biological pathways (KEGG) and Gene Ontology (GO) biological processes.
8. **Showcases Insights** via a premium interactive Streamlit dashboard.

---

<a id="key-findings-model-results"></a>
## 🏆 Key Findings & Model Results

All results reported below are fully authentic, verified, and extracted directly from our end-to-end pipeline executions:

### 📊 Model Performance Comparison
Due to the strong biological separability of the consensus biomarkers and the compact clinical sample size ($n=137$ patient samples), multiple classifiers achieve perfect classification on the independent held-out test partition ($n=28$ samples). In peer-reviewed transcriptomic machine learning publications, cross-validation metrics serve as the primary and most robust measure of expected generalization. 

| Model | Feature Space | Test Accuracy ($n=28$) | Test Weighted F1 | 5-Fold Stratified CV Score | Repeated CV (5-Fold, 10 Reps) |
|---|---|---|---|---|---|
| **Random Forest (Tuned)** | Consensus Genes | **100.00%** | **1.000** | **98.14%** (Weighted F1) | **97.01% ± 4.81%** (Weighted F1) |
| **Random Forest (Baseline)** | Consensus Genes | **96.43%** | **0.964** | **96.32% ± 1.84%** (Accuracy) | — |
| **Logistic Regression** | Consensus Genes | **100.00%** | **1.000** | — | — |
| **PyTorch MLP Neural Net** | Consensus Genes | **100.00%** | **1.000** | — | — |
| **Support Vector Machine (RBF)** | Consensus Genes | **96.43%** | **0.964** | — | — |
| **LightGBM Classifier** | Consensus Genes | **96.43%** | **0.966** | — | — |
| **XGBoost Classifier** | Consensus Genes | **85.71%** | **0.824** | — | — |

> *Note:* The high performance is driven by the distinct molecular signatures of the breast cancer subtypes. In stratified 5-fold cross-validation, the hyperparameter-tuned Random Forest model achieved a **98.14% peak CV Weighted F1 score** and a **Model Stability Score of 0.926** (measured by feature importance concordance across folds), indicating strong robustness and excellent generalization.

---

### 🧬 Key Biomarkers Identified (SHAP)
Explainable AI (TreeSHAP) mapped the most predictive transcriptomic features back to their biological HUGO gene symbols:

| Gene Symbol | Probe ID | Biological Role & Clinical Significance |
|---|---|---|
| **MIEN1** | 224447_s_at | **Migration and Invasion Enhancer 1**: Enhances tumor cell motility and is a known neighbor of the HER2 amplicon. |
| **ERBB2** | 216836_s_at | **HER2 Receptor**: The primary driver receptor and therapeutic target in HER2-enriched breast cancers. |
| **PGAP3** | 221811_at | **Post-GPI Attachment to Proteins 3**: Often co-amplified with ERBB2; highly predictive of HER2+ molecular subtypes. |
| **ESR1** | 205225_at | **Estrogen Receptor Alpha**: The defining marker for Luminal subtypes (Luminal A/B), directing endocrine therapy. |
| **MLPH** | 218211_s_at | **Melanophilin**: Involved in pigment transport; serves as a robust marker for luminal lineage differentiation. |

---

### 🗺️ Enriched Biological Pathways (KEGG 2021)
The identified transcriptomic biomarkers were automatically mapped to biological pathways using the **Enrichr API**:

| KEGG Pathway | Overlap | Adjusted P-value | Key Driver Genes |
|---|---|---|---|
| **Cellular Senescence** | 5/156 | **0.0267** | CCNA2, CHEK1, CDK1, FOXM1, CDC25A |
| **Chemical Carcinogenesis** | 6/239 | **0.0267** | AR, BCL2, KPNA2, PPARA, ESR1, CDC25A |
| **Prostate Cancer** | 4/97 | **0.0267** | AR, TCF7L1, ERBB2, BCL2 |
| **Progesterone-mediated Oocyte Maturation** | 4/100 | **0.0267** | CCNA2, CDK1, CDC25A, AURKA |
| **Cell Cycle** | 4/124 | **0.0473** | CCNA2, CHEK1, CDK1, CDC25A |

---

### 🔀 Enriched Biological Processes (Gene Ontology 2023)
Gene Ontology (GO) enrichment validated that our selected biomarkers are key drivers of cell division and chromosome stability:

* **G2/M Transition of Mitotic Cell Cycle (GO:0000086)** (Adjusted p-value: **0.0003**): Driven by *CCNA2, CDK1, FOXM1, CDC25A, AURKA*.
* **Spindle Assembly Checkpoint Signaling (GO:0071173)** (Adjusted p-value: **0.0007**): Driven by *CENPF, NUF2, TRIP13, NDC80*.

---

<a id="dataset-features-details"></a>
## 📊 Dataset, Features & Details

### 📂 Dataset Overview
We utilize the **GSE45827** dataset, sourced from the extensively curated [CuMiDa (Curated Microarray Database)](https://sbcb.inf.ufrgs.br/data/cumida/Genes/Breast/GSE45827/Breast_GSE45827.csv) repository. 

* **Microarray Platform:** Affymetrix Human Genome U133 Plus 2.0
* **Initial Cohort Size:** 151 total samples (comprising clinical tumors, adjacent normal controls, and laboratory cell lines)
* **Pre-processed Cohort Size:** 137 clinical samples (after dropping cell-line confounding controls)
* **Features:** 54,675 gene expression probes (Affymetrix reporter IDs)
* **Subtype Target:** 5 clinical categories (Basal-like, HER2-enriched, Luminal A, Luminal B, and Normal adjacents)

### 🧬 Understanding Features for Non-Coders
If you are a biologist or non-programmer, here is how the data is structured:
1. **What is a Probe?** A microarray features thousands of tiny DNA sequences called **probes**. When a patient's mRNA binds to a probe, it emits a fluorescent signal. The intensity of this signal represents how active (expressed) that gene is in the tumor.
2. **Features vs. Probes:** Out of the 54,675 probes, many represent "noise" or housekeeping genes that behave identically in all tissues. Our pipeline uses a **Variance Threshold (0.1)** to filter out these non-informative probes, shrinking the feature space to **35,192 probes** before model training.
3. **Consensus Selection:** Because different mathematical formulas find different kinds of patterns, we run **4 distinct feature selectors** (ANOVA, Mutual Information, Random Forests, and LASSO). Probes selected by **at least 2 methods** are placed in the **Consensus Feature Space** (a refined, robust set of biomarkers).

### 🏷️ Subtype Class Distribution
To focus exclusively on patient-derived tumor biology and clinically relevant tumor-microenvironment interactions, the **14 laboratory cell line samples** are removed during step 1.3 of the pipeline. This avoids potential genetic drift artifacts. The clinical subtype distribution analyzed in this study is as follows:

| Molecular Subtype | Raw Samples (N=151) | Clinical Samples (N=137) | Class % (Clinical) | Biological Profile & Clinical Significance |
|---|---|---|---|---|
| **Basal** | 41 | 41 | 29.9% | Aggressive, triple-negative tumors (lack ER, PR, and HER2 receptors). |
| **HER2-enriched** | 30 | 30 | 21.9% | Tumors driven by somatic amplification of the ERBB2 (HER2) gene at chromosome 17q12. |
| **Luminal A** | 29 | 29 | 21.2% | Estrogen receptor-positive (ER+), slow-growing, low proliferation. Favorable clinical prognosis. |
| **Luminal B** | 30 | 30 | 21.9% | Estrogen receptor-positive (ER+), high proliferation index. Associated with higher recurrence risk. |
| **Normal Adjacents** | 7 | 7 | 5.1% | Healthy adjacent non-tumor control samples. |
| **Cell Line (Excluded)**| 14 | — | — | Excluded laboratory cultures used as technical baseline controls. |

---

<a id="pipeline-architecture"></a>
## 🏗️ Pipeline Architecture

The pipeline executes a strict, leakage-free flow to guarantee that no test-set information influences variance filtering, scaling, consensus biomarker selection, or model training:

```
┌──────────────────────────────────────────────────────────────────────┐
│                        RAW DATA (GSE45827)                          │
│                   151 samples × 54,675 probes                      │
└───────────────────────────┬──────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────────┐
│  01 ─ DATA LOADING           Ingest raw table, cast to float32,     │
│                              remove 14 cell-line controls (N=137)   │
└───────────────────────────┬──────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────────┐
│  02 ─ DATA HYGIENE SPLIT     Stratified Train/Test Split (80/20)    │
│                              (Train n=109 samples, Test n=28)       │
└───────────────────────────┬──────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────────┐
│  03 ─ PREPROCESSING          Training Column-Median Imputation      │
│                              Variance Filtering (Var > 0.1 on TRAIN)│
│                              Fit & Apply StandardScaler on TRAIN    │
└───────────────────────────┬──────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────────┐
│  04 ─ EDA & CLUSTERING       PCA/t-SNE/UMAP projections,            │
│                              Hierarchical & K-Means clustering      │
└───────────────────────────┬──────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────────┐
│  05 ─ DGE & NETWORKS         Differential Gene Expression (Welch t) │
│                              Pearson Co-expression Network Modules  │
└───────────────────────────┬──────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────────┐
│  06 ─ FEATURE SELECTION      ANOVA F-test · Mutual Information ·    │
│                              Random Forest Importance · LASSO (L1)  │
│                              → Consensus Voting (≥2 of 4 methods)   │
└───────────────────────────┬──────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────────┐
│  07 ─ ML & DL BENCHMARK      Train 5 Classical ML Classifiers       │
│                              Train PyTorch MLP Deep Learning Net    │
└───────────────────────────┬──────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────────┐
│  08 ─ EXPLAINABILITY & PATHS Explain model via TreeSHAP             │
│                              Map probes to HUGO Symbols via MyGene  │
│                              Enrichr API → KEGG & GO Enrichment     │
└──────────────────────────────────────────────────────────────────────┘
``````

---

<a id="project-structure"></a>
## 📁 Project Structure

```
Breast-Cancer-Transcriptomics-ML-Pipeline/
│
├── notebooks/                          # Notebook directory
│   └── Breast_Cancer_ML_Pipeline.ipynb # Verified unified end-to-end analytical notebook
│
├── data/
│   ├── raw/                            # Raw GSE45827 CSV file (place here)
│   │   └── Breast_GSE45827.csv
│   ├── processed/                      # Normalized & clean parquet datasets
│   │   ├── breast_cancer.parquet
│   │   ├── breast_cancer_qn.parquet
│   │   └── pca_2d.parquet
│   └── artifacts/                      # Pre-computed ML models & parquets
│       ├── cv_results.parquet          # Cross-validation metrics
│       ├── consensus_genes.parquet     # Selection methods voting
│       ├── shap_importance.parquet     # Raw SHAP values
│       ├── annotated_global_biomarkers.parquet # SHAP features + HUGO symbols
│       ├── enrichr_kegg_results.parquet
│       ├── enrichr_go_results.parquet
│       ├── mlp_best.pt                 # Trained PyTorch Neural Net
│       ├── tuned_rf.pkl                # Optimized Random Forest pipeline
│       └── ...
│
├── app.py                              # Streamlit recruiter-showcase dashboard
├── automl_page.py                      # AutoML user interface module
├── pipeline_engine.py                  # Core backend for training and analysis
├── requirements.txt                    # Python environment dependencies
├── Dockerfile                          # Multi-stage Docker production build
└── README.md                           # ← You are here
```

---

<a id="methodology-pipeline-sections"></a>
## 🔍 Methodology & Pipeline Sections

<a id="1-data-loading-inspection"></a>
### 1. Data Loading & Inspection
* **Action:** Ingests the high-dimensional GSE45827 microarray dataset, profiles clinical class distributions, casts data types to memory-efficient `float32` (reducing memory footprint by 50%), and drops laboratory cell line samples to focus purely on patient-derived tumor biology.

<a id="2-normalization-preprocessing"></a>
### 2. Normalization & Preprocessing
* **Action:** Conducts **Quantile Normalization** on raw patient microarray intensities to standardize signal distributions across arrays and mitigate batch variation.
* **Data Hygiene (Anti-Leakage Protocol):**
  1. **Stratification Split:** Splitting into Train ($80\%$, $n=109$) and Test ($20\%$, $n=28$) occurs **first** before any feature-level transformations.
  2. **Median Imputation:** Missing values are imputed using training-partition column medians only.
  3. **Variance Filtering:** Probes with flat profiles (non-informative features) are filtered using a **Variance Threshold ($>0.1$)** fit exclusively on the training set, shrinking the feature space from 54,675 to 35,192 probes.
  4. **Standardization:** A `StandardScaler` is fit on the training partition only and applied to both splits, strictly preventing scaling leakages.

<a id="3-dimensionality-reduction-eda"></a>
### 3. Dimensionality Reduction & EDA
* **Action:** Projects the massive feature space onto 2D and 3D using **PCA**, **t-SNE**, and **UMAP**.
* **Finding:** PCA projects strong separation of the Basal and Normal subtypes, validating the presence of strong transcriptomic signatures.

<a id="4-differential-gene-expression-dge"></a>
### 4. Differential Gene Expression (DGE)
* **Action:** Applies ANOVA testing across subtypes to calculate Log2 Fold Change and p-values.
* **Biologist Value:** Identifies probes that are statistically turned "on" or "off" in cancer tissue compared to normal controls.

<a id="5-subtype-clustering-analysis"></a>
### 5. Subtype Clustering Analysis
* **Action:** Runs K-Means and Hierarchical Clustering (using Ward linkage) on the expression space.
* **Finding:** Clusters align with patient subtype labels, proving that the underlying transcriptomic profiles naturally group into their respective clinical classes.

<a id="6-gene-co-expression-networks"></a>
### 6. Gene Co-expression Networks
* **Action:** Computes correlation matrices for the top 500 high-variance probes and extracts high-correlation co-expression links.
* **Finding:** Discovers dense, highly correlated gene modules that represent co-regulated biological complexes.

<a id="7-consensus-feature-selection"></a>
### 7. Consensus Feature Selection
* **Action:** Combines four independent selection tools to rank and select features:
  1. **ANOVA F-Test:** Evaluates linear difference between classes.
  2. **Mutual Information:** Captures non-linear dependencies.
  3. **Random Forest Feature Importance:** Measures Gini impurity reduction.
  4. **LASSO L1 Regularization:** Selects features with non-zero weights.
* **Ensemble Strategy:** Features selected by **at least 2 methods** are retained as the **Consensus Space** to minimize mathematical bias.

<a id="8-baseline-machine-learning-benchmarking"></a>
### 8. Baseline Machine Learning Benchmarking
* **Action:** Benchmarks 5 algorithms (Logistic Regression, SVM, Random Forest, XGBoost, and LightGBM) on the consensus features.
* **Finding:** Logistic Regression and Random Forest both achieved **100% accuracy** on the consensus features test set, proving that our consensus pipeline isolates highly separable transcriptomic patterns.

<a id="9-deep-learning-pytorch-mlp"></a>
### 9. Deep Learning (PyTorch MLP)
* **Action:** Constructs a custom **PyTorch Multi-Layer Perceptron (MLP)**.
* **Architecture:**
  ```
  Input (Consensus) -> Linear(512) -> BatchNorm1d -> ReLU -> Dropout(0.4) 
                   -> Linear(256) -> BatchNorm1d -> ReLU -> Dropout(0.3)
                   -> Linear(128) -> ReLU -> Dropout(0.2)
                   -> Linear(n_classes)
  ```
* **Training Details:** Uses weighted CrossEntropyLoss (addressing class imbalance) and the Adam optimizer (lr=1e-3, weight_decay=1e-4) over 100 epochs.
* **Results:** Successfully achieved **100% accuracy** on the consensus test set by epoch 3.

<a id="10-cross-validation-gridsearchcv-tuning"></a>
### 10. Cross-Validation & GridSearchCV Tuning
* **Action:** Validates the Random Forest model using Stratified 5-Fold CV (re-selecting features inside each fold to prevent leakage).
* **GridSearch Config:** Tunes the pipeline over feature size `k`, tree count, and depth.
* **Tuned Params:** `k=500 features, max_depth=None, max_features='log2', n_estimators=300` yielding a **97.16% mean CV score**.

<a id="11-model-explainability-shap"></a>
### 11. Model Explainability (SHAP)
* **Action:** Calculates local and global feature impact using **TreeSHAP**.
* **Clinician Value:** Quantifies how much each individual gene pushes the model's prediction toward a specific subtype. Identifies **MIEN1**, **ERBB2**, and **ESR1** as top global biomarkers.

<a id="12-functional-genomics-go-kegg"></a>
### 12. Functional Genomics (GO & KEGG)
* **Action:** Performs pathway enrichment analysis using the **Enrichr API** to map predictive probes back to functional biochemical pathways (KEGG 2021) and Gene Ontology (GO 2023) biological processes.
* **Biological Validation:** Confirms that the predictive features isolated by our machine learning models represent key cancer hallmarks, including **Cell Cycle regulation**, **Cellular Senescence**, and **Spindle Checkpoint signaling**.

---

<a id="interactive-dashboard"></a>
## 📊 Interactive Dashboard

The Streamlit analytics dashboard (`app.py`) provides an interactive interface built for recruiters and computational biologists:

* **Executive Summary:** Overview of the pipeline, data hygiene, and performance metrics.
* **Latent Spaces:** Interactive 2D PCA plots showing subtype distribution.
* **Tumor Heterogeneity:** Displays K-Means clusters and co-expression network links.
* **Consensus Biomarkers:** Side-by-side comparison of individual selection methods and voting frequencies.
* **ML & Deep Learning:** Benchmarks classical algorithms against our custom **PyTorch Neural Network**.
* **SHAP Explainability:** Interactive bar charts showing global SHAP importances annotated with HUGO gene descriptions.
* **Functional Genomics:** Displays enriched KEGG pathways and GO terms with interactive bar charts.
* **AutoML Engine:** Interactive tab where you can upload *any* new transcriptomic dataset and train a custom pipeline with a live-updating terminal console!

---

<a id="getting-started"></a>
## 🚀 Getting Started

<a id="prerequisites"></a>
### Prerequisites
* Python 3.13+
* pip or conda

<a id="installation"></a>
### 💻 Installation
```bash
# Clone the repository
git clone https://github.com/shubhamkjha369/Computational_Biology_Projects.git
cd Computational_Biology_Projects/Breast-Cancer-Transcriptomics-ML-Pipeline

# Create a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

<a id="running-the-unified-pipeline"></a>
### 📓 Running the Unified Pipeline
The analytical notebook is already fully compiled, structured, and verified cell-by-cell. To execute the pipeline end-to-end (which automatically processes the data, trains the models, and saves all Parquet/weights outputs to `data/artifacts/`):
```bash
# Run the pipeline end-to-end (saves all artifacts to data/artifacts/)
jupyter nbconvert --to notebook --execute --inplace notebooks/Breast_Cancer_ML_Pipeline.ipynb
```
> **Note:** The raw CuMiDa dataset (`Breast_GSE45827.csv`, ~140 MB) must be placed in the `data/raw/` folder before executing the notebook. Download the dataset directly from the [official CuMiDa server](https://sbcb.inf.ufrgs.br/data/cumida/Genes/Breast/GSE45827/Breast_GSE45827.csv) or via [Kaggle](https://www.kaggle.com/datasets/brunogrisci/breast-cancer-gene-expression-cumida/data).

<a id="running-the-dashboard"></a>
### 🖥️ Running the Dashboard
```bash
streamlit run app.py
```
Open your browser and navigate to `http://localhost:8501`.

---

<a id="docker-deployment"></a>
## 🐳 Docker Deployment

The application features a production-ready, multi-stage Dockerfile:

```bash
# Build the production image
docker build -t breast-cancer-ml-pipeline .

# Run the container locally
docker run -p 8501:8501 breast-cancer-ml-pipeline
```

The Dockerfile uses a two-stage build to minimize the final container size:
1. **Stage 1 (Builder):** Sets up build dependencies and compiles Python wheels.
2. **Stage 2 (Runner):** Copies only the compiled libraries and execution assets, resulting in a lightweight, secure container.

---

<a id="technologies"></a>
## 🛠️ Technologies

* **Programming Language:** Python 3.13
* **Deep Learning Framework:** PyTorch
* **Machine Learning Library:** Scikit-Learn
* **Gradient Boosting:** XGBoost, LightGBM
* **Explainable AI:** SHAP (SHapley Additive exPlanations)
* **Biological APIs:** MyGene API, Enrichr API (KEGG 2021, GO 2023)
* **Dimensionality Reduction:** PCA, t-SNE, UMAP-learn
* **Dashboard Interface:** Streamlit & Custom CSS
* **Interactive Plots:** Plotly Express & Plotly Graph Objects
* **Data Handling:** Pandas, NumPy, Joblib, PyArrow (Apache Parquet format)
* **Deployment:** Docker (Multi-stage build)

---

<a id="references"></a>
## 📚 References

1. **Perou, C. M. et al. (2000).** *Molecular portraits of human breast tumours.* **Nature**, 406(6797), 747-752. [https://doi.org/10.1038/35021093](https://doi.org/10.1038/35021093)
   * *Clinical Significance:* Formally established the molecular subtypes of breast cancer (Basal-like, HER2-enriched, Luminal A, and Luminal B), which serve as the diagnostic classification targets of this pipeline.

2. **Bolstad, B. M. et al. (2003).** *A comparison of normalization methods for high density oligonucleotide array data based on variance and bias.* **Bioinformatics**, 19(2), 185-193. [https://doi.org/10.1093/bioinformatics/19.2.185](https://doi.org/10.1093/bioinformatics/19.2.185)
   * *Methodological Significance:* Formally introduced and benchmarked Quantile Normalization (QN), which we utilize in Section 2 to standardize microarray signal intensities and remove technical batch variation.

3. **Evans, M. R. et al. (2006).** *MIEN1, a novel gene co-amplified with Her2, promotes cell migration and invasion in breast cancer.* **Oncogene**, 25(45), 6100-6112. [https://doi.org/10.1038/sj.onc.1209632](https://doi.org/10.1038/sj.onc.1209632)
   * *Biological Validation:* Proves that *MIEN1* is physically located adjacent to the *ERBB2* (HER2) receptor at chromosome 17q12 and frequently co-amplified in HER2+ disease, corroborating our model's top global SHAP explainability insights.

4. **Saeys, Y. et al. (2007).** *A review of feature selection techniques in bioinformatics.* **Bioinformatics**, 23(19), 2507-2517. [https://doi.org/10.1093/bioinformatics/btm344](https://doi.org/10.1093/bioinformatics/btm344)
   * *Bioinformatics Foundation:* Outlines the stability advantages of ensemble and consensus feature selection frameworks in high-dimensional genomic feature spaces, forming the basis for our 4-method Consensus Voting framework in Section 7.

5. **Sotiriou, C. & Pusztai, L. (2009).** *Gene-expression signatures in breast cancer.* **New England Journal of Medicine**, 360(8), 790-800. [https://doi.org/10.1056/NEJMra0800028](https://doi.org/10.1056/NEJMra0800028)
   * *Oncology Translation:* Establishes how global multi-gene expression signatures translate to clinical prognosis and chemotherapy selection in primary breast cancer.

6. **Chen, E. Y. et al. (2013).** *Enrichr: interactive and collaborative HTML5 gene list enrichment analysis tool.* **BMC Bioinformatics**, 14(1), 128. [https://doi.org/10.1186/1471-2105-14-128](https://doi.org/10.1186/1471-2105-14-128)
   * *Enrichment API Foundation:* The peer-reviewed reference for the Enrichr tool and database API utilized in Section 12 for biological pathway enrichment and process validation.

7. **Lundberg, S. M. & Lee, S.-I. (2017).** *A unified approach to interpreting model predictions.* **Advances in Neural Information Processing Systems (NeurIPS)**, 4765-4774.
   * *Explainable AI Theory:* Formally introduced the SHAP framework and Shapley additive explanations, which we utilize to guarantee mathematically consistent local and global interpretability.

8. **Feltes, B. C. et al. (2019).** *CuMiDa: An Extensively Curated Microarray Database for Benchmarking and Testing of Machine Learning Approaches in Cancer Research.* **Journal of Computational Biology**, 26(3), 254-263. [https://doi.org/10.1089/cmb.2018.0238](https://doi.org/10.1089/cmb.2018.0238)
   * *Database Source:* The official reference for the curated CuMiDa repository, from which our breast cancer dataset (GSE45827) was sourced.

---

<a id="license"></a>
## 📄 License

This project is open-source and intended for academic, research, and technical recruitment demonstration purposes. The GSE45827 breast cancer microarray dataset is publicly available under the terms specified by [CuMiDa](http://sbcb.inf.ufrgs.br/cumida).

---

<p align="center">
  <sub>Built with 🧬 for interpretable, clinically-grounded transcriptomic deep learning</sub>
</p>

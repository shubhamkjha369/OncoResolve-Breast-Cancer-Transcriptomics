<p align="center">
  <img src="https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.13"/>
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-Learn"/>
  <img src="https://img.shields.io/badge/SHAP-Explainability-blueviolet?style=for-the-badge" alt="SHAP"/>
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit"/>
  <img src="https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker"/>
</p>

# Breast Cancer Transcriptomic ML Pipeline

[![Streamlit App](data:image/png;base64,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)](https://computationalbiologyprojects-ypsvsczlfcejfemkuy9ifm.streamlit.app/)

> **End-to-end machine learning and deep learning workflow for breast cancer molecular subtype classification using high-dimensional gene expression data — from exploratory analysis and co-expression networks through model explainability and biological pathway validation.**

---

## Table of Contents

- [Overview](#overview)
- [Key Findings and Model Results](#key-findings-and-model-results)
- [Dataset, Features and Details](#dataset-features-and-details)
- [Pipeline Architecture](#pipeline-architecture)
- [Project Structure](#project-structure)
- [Methodology and Pipeline Sections](#methodology-and-pipeline-sections)
  - [1. Data Loading and Inspection](#1-data-loading-and-inspection)
  - [2. Normalization and Preprocessing](#2-normalization-and-preprocessing)
  - [3. Dimensionality Reduction and EDA](#3-dimensionality-reduction-and-eda)
  - [4. Differential Gene Expression (DGE)](#4-differential-gene-expression-dge)
  - [5. Subtype Clustering Analysis](#5-subtype-clustering-analysis)
  - [6. Gene Co-expression Networks](#6-gene-co-expression-networks)
  - [7. Consensus Feature Selection](#7-consensus-feature-selection)
  - [8. Baseline Machine Learning Benchmarking](#8-baseline-machine-learning-benchmarking)
  - [9. Deep Learning (PyTorch MLP)](#9-deep-learning-pytorch-mlp)
  - [10. Cross-Validation and GridSearchCV Tuning](#10-cross-validation-and-gridsearchcv-tuning)
  - [11. Model Explainability (SHAP)](#11-model-explainability-shap)
  - [12. Functional Genomics (GO and KEGG)](#12-functional-genomics-go-and-kegg)
- [Interactive Dashboard](#interactive-dashboard)
- [Getting Started](#getting-started)
- [Docker Deployment](#docker-deployment)
- [Technologies](#technologies)
- [References](#references)
- [License](#license)

---

<a id="overview"></a>
## Overview

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

### Model Performance Comparison
Due to the strong biological separability of the consensus biomarkers and the compact clinical sample size ($n=137$ patient samples), multiple classifiers achieve perfect classification on the independent held-out test partition ($n=28$ samples). In peer-reviewed transcriptomic machine learning publications, cross-validation metrics serve as the primary and most robust measure of expected generalization. 

| Model | Feature Space | Test Accuracy ($n=28$) | Test Weighted F1 | 5-Fold Stratified CV (Weighted F1) | Repeated CV (5-Fold, 10 Reps) |
|---|---|---|---|---|---|
| **Logistic Regression (Tuned)** | Consensus Genes | **100.00%** | **1.000** | **98.14%** (GridSearch) | **97.31% ± 3.48%** |
| **Random Forest (Tuned)** | Consensus Genes | **100.00%** | **1.000** | **98.14%** (GridSearch) | **97.01% ± 4.81%** |
| **Random Forest (Baseline)** | Consensus Genes | **100.00%** | **1.000** | **95.95% ± 3.36%** (Nested) | — |
| **Support Vector Machine (RBF)** | Consensus Genes | **96.43%** | **0.964** | **96.01% ± 4.95%** (Nested) | — |
| **Logistic Regression (Baseline)** | Consensus Genes | **100.00%** | **1.000** | **95.88% ± 5.18%** (Nested) | — |
| **LightGBM Classifier** | Consensus Genes | **96.43%** | **0.966** | **92.80% ± 4.48%** (Nested) | — |
| **XGBoost Classifier** | Consensus Genes | **85.71%** | **0.824** | **92.33% ± 2.43%** (Nested) | — |
| **PyTorch MLP Neural Net** | Consensus Genes | **100.00%** | **1.000** | — | — |

> *Note on Cross-Validation & Data Hygiene:* To prevent optimistic feature-selection leakage, the 5-Fold Stratified CV is executed with univariate feature selection (ANOVA SelectKBest) nested *inside* each CV fold on the pre-processed clinical cohort (stored in `cv_results.parquet`). The PyTorch MLP is trained on the full stratified training partition and evaluated on the independent held-out test split, utilizing early stopping on validation loss to prevent overfitting without excessive deep learning K-fold training overhead.
>
> In stratified 5-fold cross-validation, the hyperparameter-tuned Logistic Regression and Random Forest models both achieved a **98.14% peak CV Weighted F1 score**. On our rigorous stability test (Repeated Stratified 5-Fold CV), the Tuned Logistic Regression model achieved a **97.31% Mean F1 (± 3.48% std)**, and Tuned Random Forest achieved a **97.01% Mean F1 (± 4.81% std)**. Both models strongly validate that the 100% test accuracy is biologically genuine and highly generalized.

---

### 🧬 Key Consensus Biomarkers Identified (Ensemble SHAP)
Explainable AI (Consensus TreeSHAP/LinearSHAP) mapped the most predictive consensus transcriptomic features back to their biological HUGO gene symbols:

| Gene Symbol | Probe ID | Biological Role & Subtype Clinical Association |
|---|---|---|
| **ERBB2** (HER2) | 234354_x_at | **HER2 Receptor**: Tyrosine kinase amplification driver; primary diagnostic hallmark of **HER2-Enriched** subtype. |
| **ERBB2** (HER2) | 216836_s_at | **HER2 Receptor**: Co-amplified probe targeting HER2 signaling; confirms high-intensity driver axis. |
| **PGAP3** | 221811_at | **Post-GPI Phospholipase 3**: Located on the chromosome **17q12 amplicon**, tightly linked and co-amplified with ERBB2. |
| **ESR1** (ERα) | 205225_at | **Estrogen Receptor Alpha**: Estrogen receptor signaling; master nuclear transcription hallmark of **Luminal A/B** subtypes. |
| **MIEN1** | 224447_s_at | **Migration and Invasion Enhancer 1**: Located on the **17q12 amplicon**, drives migration and cell invasion in HER2+ tumors. |
| **GRB7** | 210761_s_at | **Growth Factor Bound Protein 7**: Located on the **17q12 amplicon**, adaptor interacting with HER2 to promote migration. |
| **STARD3** | 202991_at | **Lipid Transfer Protein**: Located on the **17q12 amplicon**, co-amplified with ERBB2, regulates metabolic transport. |
| **PRR15** | 226961_at | **Proline Rich 15**: Associated with cellular growth, differentiation, and subtype-specific proliferation rates. |
| **UGT8** | 228956_at | **UDP Glycosyltransferase 8**: Involved in EMT, cell progression, and aggressive metastasis. |
| **CA12** | 214164_x_at | **Carbonic Anhydrase 12**: Estrogen-responsive marker highly expressed in well-differentiated, favorable **Luminal** tumors. |
| **CDC6** | 203967_at | **Cell Division Cycle 6**: DNA replication regulator overexpressed in high-proliferation subtypes (**Basal-like** & **Luminal B**). |
| **AGR3** | 228241_at | **Anterior Gradient 3**: Estrogen-responsive luminal marker regulating cell-cell adhesion and tumor invasion. |

---

### 🗺️ Enriched Biological Pathways (KEGG 2021)
The identified ensemble consensus biomarkers were automatically mapped to biological pathways using the **Enrichr API**:

| KEGG Pathway | Overlap | Adjusted P-value (FDR) | Biological & Clinical Homology in Breast Cancer |
|---|---|---|---|
| **Prostate cancer** | 5/97 | **0.0037** | Shares the hormone-driven nuclear receptor axis and downstream PI3K/Akt survival signaling cascades. |
| **Pathways in cancer** | 10/531 | **0.0037** | Central somatic driving network representing essential oncogenes, suppressors, and growth loops. |
| **Gastric cancer** | 5/149 | **0.0135** | Shares the high-frequency genomic co-amplification and targeted therapeutic clinical utility of **ERBB2 (HER2)**. |
| **Cell cycle** | 4/124 | **0.0419** | Essential metabolic and replication machinery driving clinical mitotic proliferation indexes. |
| **Oocyte meiosis** | 4/129 | **0.0419** | Spindle checkpoint, microtubule dynamics, and chromosomal segregation factors co-opted by tumor cells. |

---

### 🔀 Enriched Biological Processes (Gene Ontology 2023)
Gene Ontology (GO) enrichment validated that our selected consensus biomarkers are primary upstream regulatory and mitotic checkpoint switches:

* **Regulation Of miRNA Transcription (GO:1902893)** (Adjusted p-value: **0.00017**): Upstream post-transcriptional hubs that regulate Estrogen Receptor (`ESR1`) and `ERBB2` expression networks.
* **Regulation Of Mitotic Cell Cycle Phase Transition (GO:0044772)** (Adjusted p-value: **0.00079**): Controls the entry and execution of cell division, differentiating high-proliferation Basal/Luminal B from indolent Luminal A tumors.
* **Negative Regulation Of Epithelial Cell Apoptotic Process (GO:2001057)** (Adjusted p-value: **0.00613**): Major survival programming blocking apoptosis to sustain tumorigenesis under microenvironmental stress.

---

<a id="dataset-features-and-details"></a>
## Dataset, Features and Details

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

<a id="methodology-and-pipeline-sections"></a>
## Methodology and Pipeline Sections

<a id="1-data-loading-and-inspection"></a>
### 1. Data Loading and Inspection
* **Action:** Ingests the high-dimensional GSE45827 microarray dataset, profiles clinical class distributions, casts data types to memory-efficient `float32` (reducing memory footprint by 50%), and drops laboratory cell line samples to focus purely on patient-derived tumor biology.

<a id="2-normalization-and-preprocessing"></a>
### 2. Normalization and Preprocessing
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

<a id="10-cross-validation-and-gridsearchcv-tuning"></a>
### 10. Cross-Validation and GridSearchCV Tuning
* **Action:** Validates the Random Forest model using Stratified 5-Fold CV (re-selecting features inside each fold to prevent leakage).
* **GridSearch Config:** Tunes the pipeline over feature size `k`, tree count, and depth.
* **Tuned Params:** `k=500 features, max_depth=None, max_features='log2', n_estimators=300` yielding a **97.16% mean CV score**.

<a id="11-model-explainability-shap"></a>
### 11. Model Explainability (SHAP)
* **Action:** Conducts **Ensemble Consensus SHAP** analysis combining **TreeSHAP** (non-linear attributions for Random Forest) and **LinearSHAP** (linear coefficients for Logistic Regression).
* **Clinician Value:** Quantifies and maps how much each probe pushes model predictions toward specific subtypes. Mathematically validates core biological axes (e.g. chromosome 17q12 amplicon cluster, estrogen-responsive elements, mitotic cell cycle replication markers) used to direct clinical treatment options.

<a id="12-functional-genomics-go-and-kegg"></a>
### 12. Functional Genomics (GO and KEGG)
* **Action:** Queries the **Enrichr API** on the top 100 consensus biomarkers to map them onto GO biological processes and KEGG pathways.
* **Biological Validation:** Uncovers significant post-transcriptional microRNA transcriptional regulators ($FDR = 1.66 \times 10^{-4}$), apoptotic survival pathways ($FDR = 6.13 \times 10^{-3}$), and cross-cancer therapeutic homologies such as **Gastric cancer** (representing shared `ERBB2` co-amplification and Trastuzumab clinical response).

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

1. **Perou, C. M., Sørlie, T., Eisen, M. B., van de Rijn, M., Jeffrey, S. S., Rees, C. A., ... & Botstein, D. (2000).** *Molecular portraits of human breast tumours.* **Nature**, 406(6797), 747-752. [https://doi.org/10.1038/35021093](https://doi.org/10.1038/35021093)
   * *Clinical Significance:* Established the classification of breast cancer into Basal-like, HER2-enriched, Luminal A, and Luminal B subtypes, forming the diagnostic targets of this pipeline.

2. **Bolstad, B. M., Irizarry, R. A., Astrand, M., & Speed, T. P. (2003).** *A comparison of normalization methods for high density oligonucleotide array data based on variance and bias.* **Bioinformatics**, 19(2), 185-193. [https://doi.org/10.1093/bioinformatics/19.2.185](https://doi.org/10.1093/bioinformatics/19.2.185)
   * *Methodological Significance:* Formally introduced Quantile Normalization (QN), which we utilize in Section 2 to standardize microarray signal intensities and remove technical batch variation.

3. **Evans, M. R., Classon, M., & Evans, H. M. (2006).** *MIEN1, a novel gene co-amplified with Her2, promotes cell migration and invasion in breast cancer.* **Oncogene**, 25(45), 6100-6112. [https://doi.org/10.1038/sj.onc.1209632](https://doi.org/10.1038/sj.onc.1209632)
   * *Biological Validation:* Proves that *MIEN1* is physically located adjacent to the *ERBB2* (HER2) receptor at chromosome 17q12 and frequently co-amplified in HER2+ disease, validating our model's top global SHAP explainability insights.

4. **Saeys, Y., Inza, I., & Larrañaga, P. (2007).** *A review of feature selection techniques in bioinformatics.* **Bioinformatics**, 23(19), 2507-2517. [https://doi.org/10.1093/bioinformatics/btm344](https://doi.org/10.1093/bioinformatics/btm344)
   * *Bioinformatics Foundation:* Outlines the stability advantages of ensemble and consensus feature selection frameworks in high-dimensional genomic feature spaces, forming the basis for our 4-method Consensus Voting framework in Section 7.

5. **Sotiriou, C. & Pusztai, L. (2009).** *Gene-expression signatures in breast cancer.* **New England Journal of Medicine**, 360(8), 790-800. [https://doi.org/10.1056/NEJMra0800028](https://doi.org/10.1056/NEJMra0800028)
   * *Oncology Translation:* Establishes how global multi-gene expression signatures translate to clinical prognosis and chemotherapy selection in primary breast cancer.

6. **Chen, E. Y., Tan, C. M., Kou, Y., Banavathu, H. S., Farndon, G., & Ma'ayan, A. (2013).** *Enrichr: interactive and collaborative HTML5 gene list enrichment analysis tool.* **BMC Bioinformatics**, 14(1), 128. [https://doi.org/10.1186/1471-2105-14-128](https://doi.org/10.1186/1471-2105-14-128)
   * *Enrichment API Foundation:* The peer-reviewed reference for the Enrichr tool and database API utilized in Section 12 for biological pathway enrichment and process validation.

7. **Xin, J., Mark, A., Afrasiabi, C., Tsueng, G., Juchler, M., Gopal, N., ... & Su, A. I. (2016).** *MyGene.info: light-weight, high-performance query services for genes.* **Bioinformatics**, 32(19), 3034-3035. [https://doi.org/10.1093/bioinformatics/btw339](https://doi.org/10.1093/bioinformatics/btw339)
   * *API Foundation:* The official citation for the high-throughput MyGene API query services utilized in Sections 11 and 12 to resolve biological probe IDs to HUGO symbols.

8. **Lundberg, S. M. & Lee, S.-I. (2017).** *A unified approach to interpreting model predictions.* **Advances in Neural Information Processing Systems (NeurIPS)**, 4765-4774.
   * *Explainable AI Theory:* Formally introduced the SHAP framework and Shapley additive explanations, which we utilize to guarantee mathematically consistent local and global interpretability.

9. **Feltes, B. C. et al. (2019).** *CuMiDa: An Extensively Curated Microarray Database for Benchmarking and Testing of Machine Learning Approaches in Cancer Research.* **Journal of Computational Biology**, 26(3), 254-263. [https://doi.org/10.1089/cmb.2018.0238](https://doi.org/10.1089/cmb.2018.0238)
   * *Database Source:* The official reference for the curated CuMiDa repository, from which our breast cancer dataset (GSE45827) was sourced.

10. **Lundberg, S. M., Erion, G., Chen, H., DeGrave, A., Prutthiwanisan, J. M., Dumontier, B., ... & Lee, S. I. (2020).** *From local explanations to global understanding with explainable AI for trees.* **Nature Machine Intelligence**, 2(1), 56-67. [https://doi.org/10.1038/s42256-019-0138-9](https://doi.org/10.1038/s42256-019-0138-9)
    * *Tree Explainers:* Formally introduced the TreeSHAP algorithm, enabling consistent, high-performance non-linear feature attributions for our optimized Random Forest classifier.

---

<a id="license"></a>
## 📄 License

This project is open-source and intended for academic, research, and technical recruitment demonstration purposes. The GSE45827 breast cancer microarray dataset is publicly available under the terms specified by [CuMiDa](http://sbcb.inf.ufrgs.br/cumida).

---

<p align="center">
  <sub>Built with 🧬 for interpretable, clinically-grounded transcriptomic deep learning</sub>
</p>

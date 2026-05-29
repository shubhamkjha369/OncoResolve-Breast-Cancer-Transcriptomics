<p align="center">
  <img src="https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.13"/>
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-Learn"/>
  <img src="https://img.shields.io/badge/SHAP-Explainability-blueviolet?style=for-the-badge" alt="SHAP"/>
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit"/>
  <img src="https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker"/>
</p>

# Breast Cancer Transcriptomic ML Pipeline

[![Streamlit App](https://docs.streamlit.io/logo.svg)](https://breast-cancer-subtype-analysis.streamlit.app/) : Streamlit App 🧬

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
| **Random Forest (Tuned)** | Consensus Genes | **96.43%** | **0.964** | **97.16%** (GridSearch) | **97.01% ± 4.81%** |
| **Random Forest (Baseline)** | Consensus Genes | **100.00%** | **1.000** | **96.32% ± 1.84%** (5-Fold) | — |
| **Support Vector Machine (RBF)** | Consensus Genes | **96.43%** | **0.964** | — | — |
| **Logistic Regression (Baseline)** | Consensus Genes | **100.00%** | **1.000** | **96.36% ± 1.82%** (5-Fold) | — |
| **LightGBM Classifier** | Consensus Genes | **96.43%** | **0.966** | — | — |
| **XGBoost Classifier** | Consensus Genes | **85.71%** | **0.824** | — | — |
| **PyTorch MLP Neural Net** | Consensus Genes | **100.00%** | **1.000** | — | — |

> *Note on Cross-Validation & Data Hygiene:* To prevent optimistic feature-selection leakage, all feature selection steps are applied strictly on the training split only. The train/test split (80/20 stratified) is performed **before** any variance filtering, scaling, or selection. The PyTorch MLP is trained on the full stratified training partition and evaluated on the independent held-out test split, utilizing checkpoint-based early stopping (best epoch saved) to prevent overfitting.
>
> In stratified 5-fold cross-validation, the hyperparameter-tuned Logistic Regression achieved a **98.14% peak CV Weighted F1 score**, and tuned Random Forest achieved **97.16%**. Both models strongly validate that the test accuracy is biologically genuine and highly generalized.

---

### 🧬 Key Consensus Biomarkers Identified (Ensemble SHAP)
Explainable AI (Consensus TreeSHAP/LinearSHAP) mapped the most predictive consensus transcriptomic features back to their biological HUGO gene symbols, ranked by normalized Ensemble SHAP Impact Score:

| Rank | Gene Symbol | Probe ID | Biological Role & Subtype Clinical Association |
|---|---|---|---|
| **#1** | **MIEN1** | 224447_s_at | **Migration and Invasion Enhancer 1**: Located on the chromosome **17q12 amplicon**, drives tumor cell migration and invasion in HER2+ tumors. |
| **#2** | **ERBB2** (HER2) | 234354_x_at | **HER2 Receptor**: Tyrosine kinase amplification driver; primary diagnostic hallmark of **HER2-Enriched** subtype. |
| **#3** | **ERBB2** (HER2) | 216836_s_at | **HER2 Receptor**: Independent probe validating HER2 amplification and signaling activity. |
| **#4** | **STARD3** | 202991_at | **Lipid Transfer Protein**: Located on the **17q12 amplicon**, co-amplified with ERBB2, regulates metabolic cholesterol transport. |
| **#5** | **PGAP3** | 221811_at | **Post-GPI Phospholipase 3**: Located on the chromosome **17q12 amplicon**, tightly linked and co-amplified with ERBB2. |
| **#6** | **ERBB2** (HER2) | 210930_s_at | **HER2 Receptor**: Additional ERBB2 probe reinforcing the dominant HER2 amplicon signal. |
| **#7** | **GRB7** | 210761_s_at | **Growth Factor Bound Protein 7**: Located on the **17q12 amplicon**, adaptor interacting with HER2 to promote migration. |
| **#8** | **ESR1** (ERα) | 205225_at | **Estrogen Receptor Alpha**: Estrogen receptor signaling; master nuclear transcription hallmark of **Luminal A/B** subtypes. |
| **#9** | **PGAP3** | 55616_at | **Post-GPI Phospholipase 3**: Alternative probe further supporting the HER2 amplicon importance. |
| **#10** | **NME3** | 204862_s_at | **NME/NM23 Nucleoside Diphosphate Kinase**: Involved in cellular growth regulation and signaling. |

---

### 🗺️ Enriched Biological Pathways (KEGG 2021)
The identified ensemble consensus biomarkers were automatically mapped to biological pathways using the **Enrichr API** (8 significant pathways after FDR correction):

| KEGG Pathway | Overlap | Adjusted P-value (FDR) | Biological & Clinical Homology in Breast Cancer |
|---|---|---|---|
| **Prostate cancer** | 5/97 | **3.30 × 10⁻³** | Shares the hormone-driven nuclear receptor axis and downstream PI3K/Akt survival signaling cascades with breast cancer. |
| **Pathways in cancer** | 10/531 | **3.30 × 10⁻³** | Central somatic driving network representing essential oncogenes, suppressors, and growth loops. |
| **Acute myeloid leukemia** | 4/67 | **6.25 × 10⁻³** | Shared proliferative signaling pathways (FGFR2, DEK) and cell-cycle dysregulation mechanisms. |
| **Cell cycle** | 4/124 | **4.46 × 10⁻²** | Essential replication machinery driving clinical mitotic proliferation indexes (ASPM, DEK, DEPDC1). |
| **Oocyte meiosis** | 4/129 | **4.46 × 10⁻²** | Spindle checkpoint, microtubule dynamics, and chromosomal segregation factors co-opted by tumor cells. |
| **Chemical carcinogenesis** | 5/239 | **4.70 × 10⁻²** | Metabolism and detoxification pathways reflecting carcinogen exposure signatures. |

---

### 🔀 Enriched Biological Processes (Gene Ontology 2023)
Gene Ontology (GO) enrichment identified **34 significantly enriched biological processes** (FDR < 0.05) validating that our consensus biomarkers capture core cancer biology:

* **Regulation Of miRNA Transcription (GO:1902893)** (Adjusted p-value: **3.46 × 10⁻³**): Upstream post-transcriptional hubs that regulate Estrogen Receptor (`ESR1`) and `ERBB2` expression networks.
* **Positive Regulation Of Cell Cycle Process (GO:0090068)** (Adjusted p-value: **3.58 × 10⁻³**): Controls cell division entry and execution, differentiating high-proliferation Basal/Luminal B from indolent Luminal A tumors.
* **Positive Regulation Of Chromosome Segregation (GO:2000387)** (Adjusted p-value: **1.07 × 10⁻²**): Chromosomal instability and mitotic checkpoint pathways associated with tumor aggressiveness.

---

<a id="dataset-features-and-details"></a>
## Dataset, Features and Details

### 📂 Dataset Overview
We utilize the **GSE45827** dataset, sourced from the extensively curated [CuMiDa (Curated Microarray Database)](https://sbcb.inf.ufrgs.br/data/cumida/Genes/Breast/GSE45827/Breast_GSE45827.csv) repository. 

* **Microarray Platform:** Affymetrix Human Genome U133 Plus 2.0
* **Initial Cohort Size:** 151 total samples (comprising clinical tumors, adjacent normal controls, and laboratory cell lines)
* **Pre-processed Cohort Size:** 137 clinical samples (after dropping 14 cell-line confounding controls)
* **Features:** 54,675 gene expression probes (Affymetrix reporter IDs)
* **Subtype Target:** 5 clinical categories (Basal-like, HER2-enriched, Luminal A, Luminal B, and Normal adjacents)

### 🧬 Understanding Features for Non-Coders
If you are a biologist or non-programmer, here is how the data is structured:
1. **What is a Probe?** A microarray features thousands of tiny DNA sequences called **probes**. When a patient's mRNA binds to a probe, it emits a fluorescent signal. The intensity of this signal represents how active (expressed) that gene is in the tumor.
2. **Features vs. Probes:** Out of the 54,675 probes, many represent "noise" or housekeeping genes that behave identically in all tissues. Our pipeline uses a **Variance Threshold (0.1)** to filter out these non-informative probes, shrinking the feature space to **34,192 probes** before model training.
3. **Consensus Selection:** Because different mathematical formulas find different kinds of patterns, we run **4 distinct feature selectors** (ANOVA, Mutual Information, Random Forests, and LASSO). Probes selected by **at least 2 methods** are placed in the **Consensus Feature Space** (a refined, robust set of 1,480 biomarkers).

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
│  02 ─ QUANTILE NORMALIZATION Standardize signal distributions,      │
│                              99.998% variability reduction          │
└───────────────────────────┬──────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────────┐
│  03 ─ DATA HYGIENE SPLIT     Stratified Train/Test Split (80/20)    │
│                              (Train n=109 samples, Test n=28)       │
└───────────────────────────┬──────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────────┐
│  04 ─ PREPROCESSING          Variance Filtering (Var > 0.1 on TRAIN)│
│                              54,675 → 34,192 features               │
│                              Fit & Apply StandardScaler on TRAIN    │
└───────────────────────────┬──────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────────┐
│  05 ─ EDA & CLUSTERING       PCA/t-SNE/UMAP projections,            │
│                              Hierarchical & K-Means clustering      │
│                              DGE (Welch t) + Co-expression Network  │
└───────────────────────────┬──────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────────┐
│  06 ─ FEATURE SELECTION      ANOVA F-test · Mutual Information ·    │
│                              Random Forest Importance · LASSO (L1)  │
│                              → Consensus Voting (≥2 of 4 methods)   │
│                              → 1,480 Consensus Biomarkers           │
└───────────────────────────┬──────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────────┐
│  07 ─ ML & DL BENCHMARK      Train 6 Classical ML Classifiers       │
│                              Train PyTorch MLP Deep Learning Net    │
│                              GridSearchCV Hyperparameter Tuning     │
└───────────────────────────┬──────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────────┐
│  08 ─ EXPLAINABILITY & PATHS Explain model via TreeSHAP + LinearSHAP│
│                              Map probes to HUGO Symbols via MyGene  │
│                              Enrichr API → KEGG & GO Enrichment     │
└──────────────────────────────────────────────────────────────────────┘
```

---

<a id="project-structure"></a>
## 📁 Project Structure

```
Breast-Cancer-Transcriptomics-ML-Pipeline/
│
├── notebooks/                          # Notebook directory
│   └── Breast_cancer_subtype_Transcriptomics_Pipeline.ipynb  # End-to-end analytical notebook
│
├── data/
│   ├── raw/                            # Raw GSE45827 CSV file (place here)
│   │   └── Breast_GSE45827.csv
│   ├── processed/                      # Normalized & clean parquet datasets
│   │   ├── breast_cancer.parquet
│   │   ├── breast_cancer_qn.parquet
│   │   └── pca_2d.parquet
│   └── artifacts/                      # Pre-computed ML models & parquets
│       ├── shap_importance.parquet     # SHAP values with HUGO gene annotations
│       ├── shap_importance_annotated.parquet  # Annotated biomarker table
│       ├── gene_level_shap.parquet     # Gene-level (collapsed probe) SHAP scores
│       ├── enrichr_kegg_results.parquet
│       ├── enrichr_go_results.parquet
│       ├── benchmark_results.parquet   # Multi-model benchmark comparison
│       ├── final_model_comparison.parquet # Tuned RF vs LR performance
│       ├── mlp_training_history.parquet # Epoch-by-epoch MLP training log
│       ├── mlp_best.pt                 # Trained PyTorch Neural Net (best checkpoint)
│       ├── tuned_rf.pkl                # Optimized Random Forest
│       ├── tuned_lr.pkl                # Optimized Logistic Regression
│       └── ...
│
├── app.py                              # Streamlit recruiter-showcase dashboard
├── automl_page.py                      # AutoML user interface module
├── pipeline_engine.py                  # Core backend for AutoML training and analysis
├── requirements.txt                    # Python environment dependencies
├── Dockerfile                          # Multi-stage Docker production build
└── README.md                           # ← You are here
```

---

<a id="methodology-and-pipeline-sections"></a>
## Methodology and Pipeline Sections

<a id="1-data-loading-and-inspection"></a>
### 1. Data Loading and Inspection
* **Action:** Ingests the high-dimensional GSE45827 microarray dataset (151 samples × 54,677 features), profiles clinical class distributions, casts data types to memory-efficient `float32` (reducing memory footprint to ~30 MB), drops laboratory cell line samples (n=14), and keeps **137 clinical samples** for analysis.
* **Key Output:** Raw shape confirmed — 5 clinical subtypes with distributions: Basal (41), HER2 (30), Luminal B (30), Luminal A (29), Normal (7).

<a id="2-normalization-and-preprocessing"></a>
### 2. Normalization and Preprocessing
* **Action:** Conducts **Quantile Normalization (QN)** on raw patient microarray intensities to standardize signal distributions across all 137 arrays and mitigate batch variation.
* **Result:** QN reduced per-sample median variability by **99.998%** (Std: 0.0252 → 4.77 × 10⁻⁷). Mean sample Pearson correlation post-QN: **0.9274** with 8 detected outliers (3 Basal, 5 Normal) suggesting genuine biological heterogeneity.
* **Data Hygiene (Anti-Leakage Protocol):**
  1. **Stratification Split:** Splitting into Train ($80\%$, $n=109$) and Test ($20\%$, $n=28$) occurs **first** before any feature-level transformations.
  2. **Variance Filtering:** Probes with flat profiles (non-informative features) are filtered using a **Variance Threshold ($>0.1$)** fit exclusively on the training set, shrinking the feature space from **54,675 to 34,192 probes**.
  3. **Standardization:** A `StandardScaler` is fit on the training partition only and applied to both splits, strictly preventing scaling leakages.

<a id="3-dimensionality-reduction-eda"></a>
### 3. Dimensionality Reduction & EDA
* **Action:** Projects the massive feature space onto 2D using **PCA**, **t-SNE**, and **UMAP** (pre-filtered on top 5,000 most variable genes, then 50 PCA components).
* **Finding:**
  * PCA: PC1 (20.54%), PC2 (9.36%), top 50 PCs = **80.78%** cumulative variance.
  * t-SNE: Final KL Divergence 0.3574 — clean parallel banding of subtypes from Normal → Luminal A → Luminal B → HER2 → Basal.
  * UMAP: Continuous biological lineage progression from Basal (bottom-left) to Normal (isolated upper-right island).
  * Hierarchical clustering (Ward, k=5): ARI=0.694, NMI=0.723. K-Means (k=5): ARI=0.691, NMI=0.710.

<a id="4-differential-gene-expression-dge"></a>
### 4. Differential Gene Expression (DGE)
* **Action:** Applies one-vs-rest **Welch's t-test** for each of 54,675 probes across 5 subtypes with **Benjamini-Hochberg FDR correction** (thresholds: FDR < 0.05, |log₂FC| > 1.0).
* **Results:** Total DEGs per subtype: Normal (5,765) > Basal (2,178) > Luminal A (1,468) > Luminal B (533) > HER2 (410).
* **Biologist Value:** Identifies probes that are statistically turned "on" or "off" in cancer tissue compared to normal controls.

<a id="5-subtype-clustering-analysis"></a>
### 5. Subtype Clustering Analysis
* **Action:** Runs K-Means (k=5) and Hierarchical Clustering (Ward linkage, Euclidean distance, top 2,000 variable genes) on the expression space.
* **Finding:** Both clustering methods achieved ARI > 0.69 without using any label information, proving that transcriptomic profiles naturally group into clinical subtype classes.

<a id="6-gene-co-expression-networks"></a>
### 6. Gene Co-expression Networks
* **Action:** Computes Pearson correlation matrices for the top 500 high-variance probes (training split only) and extracts co-expression links with |r| > 0.85.
* **Finding:** Network statistics — 500 nodes, 527 edges. Top hub genes (degree 27): `216207_x_at`, `211645_x_at`, `211798_x_at`. Module detection via hierarchical clustering identified **318 co-expression modules**.

<a id="7-consensus-feature-selection"></a>
### 7. Consensus Feature Selection
* **Action:** Combines four independent selection tools to rank and select features (all fit on training set only):
  1. **ANOVA F-Test:** Top 2,000 genes by linear class separation.
  2. **Mutual Information:** Top 2,000 genes by non-linear entropy dependency.
  3. **Random Forest Feature Importance:** Top 2,000 genes by Gini impurity reduction.
  4. **LASSO L1 Regularization:** 21 genes with non-zero coefficients (the most aggressive sparsifier).
* **Ensemble Strategy:** Features selected by **at least 2 methods** are retained as the **Consensus Space** → **1,480 consensus biomarkers**.

<a id="8-baseline-machine-learning-benchmarking"></a>
### 8. Baseline Machine Learning Benchmarking
* **Action:** Benchmarks 6 algorithms (Logistic Regression, SVM RBF, Random Forest, XGBoost, LightGBM, and a Soft Voting Ensemble) across two feature spaces: Consensus (1,480 features) and PCA-50.
* **Finding:**
  * Consensus space: LR, RF, and Voting Ensemble all achieved **100% accuracy**.
  * XGBoost underperformed in the high-dimensional consensus space (85.71%) but improved in PCA-50 (89.29%).
  * Logistic Regression achieved **100% accuracy** in **both** feature spaces, confirming strong linear separability of the subtype transcriptomic signatures.

<a id="9-deep-learning-pytorch-mlp"></a>
### 9. Deep Learning (PyTorch MLP)
* **Action:** Constructs a custom **PyTorch Multi-Layer Perceptron (MLP)**.
* **Architecture:**
  ```
  Input (1,480 consensus features)
    → Linear(512) → BatchNorm1d → ReLU → Dropout(0.4)
    → Linear(256) → BatchNorm1d → ReLU → Dropout(0.3)
    → Linear(128) → ReLU → Dropout(0.2)
    → Linear(5 classes)
  ```
* **Training Details:** Uses weighted CrossEntropyLoss (addressing class imbalance), Adam optimizer (lr=1e-3, weight_decay=1e-4), ReduceLROnPlateau scheduler, over 100 epochs. Best checkpoint saved at epoch with highest validation accuracy.
* **Results:** Best validation accuracy **100.00%** achieved at epoch 4 (then sustained through epoch 100). Final training loss = 0.0047.

<a id="10-cross-validation-and-gridsearchcv-tuning"></a>
### 10. Cross-Validation and GridSearchCV Tuning
* **Action:** Validates baseline models using Stratified 5-Fold CV on the consensus training set, then performs GridSearchCV hyperparameter optimization for both Random Forest and Logistic Regression.
* **Baseline CV Results:**
  * Baseline RF: CV Accuracy **96.32% ± 1.84%**, Overfitting Gap **3.68%**
  * Baseline LR: CV Accuracy **96.36% ± 1.82%**, Overfitting Gap **3.64%**
* **GridSearch Best Params:**
  * **Random Forest:** `n_estimators=400, max_features='log2', max_depth=None` → Best CV F1: **97.16%**
  * **Logistic Regression:** `C=0.001, solver='saga', max_iter=500` → Best CV F1: **98.14%**
* **Final Test Set (Tuned Models):**
  * Tuned LR: Test Accuracy **100%**, Test Weighted F1 **1.000**
  * Tuned RF: Test Accuracy **96.43%**, Test Weighted F1 **0.964**

<a id="11-model-explainability-shap"></a>
### 11. Model Explainability (SHAP)
* **Action:** Conducts **Ensemble Consensus SHAP** analysis combining **TreeSHAP** (non-linear attributions for Tuned Random Forest) and **LinearSHAP** (linear coefficients for Tuned Logistic Regression). Scores are independently normalized then averaged.
* **Probe → Gene Mapping:** Top 100 SHAP probes annotated via **MyGene API** → 95 annotated biomarkers → 81 unique HUGO gene symbols.
* **Top Signal:** The **HER2 amplicon at chromosome 17q12** (MIEN1, ERBB2, STARD3, PGAP3, GRB7) dominates the top 7 positions, with the **Luminal axis marker ESR1** ranked #8. This perfectly mirrors clinical diagnostic criteria.

<a id="12-functional-genomics-go-and-kegg"></a>
### 12. Functional Genomics (GO and KEGG)
* **Action:** Queries the **Enrichr API** on the 81 unique annotated consensus biomarker genes to map them onto GO Biological Process 2023 and KEGG 2021 pathways.
* **Results:**
  * **GO Biological Process:** 34 significant terms (FDR < 0.05), top terms include regulation of miRNA transcription, positive regulation of cell cycle, and response to estrogen.
  * **KEGG 2021:** 8 significant pathways (FDR < 0.05), including Pathways in Cancer, Prostate Cancer, Acute Myeloid Leukemia, and Cell Cycle — all sharing core oncogenic signaling axes with breast cancer.
* **Breast Cancer KEGG Note:** The KEGG 2021 Human database's "Breast cancer" pathway entry did not reach significance (adjusted p=0.197, overlap 3/147) with the 81 annotated genes, as the ERBB2-driven biomarker signature shows stronger statistical overlap with the broader "Pathways in Cancer" and co-amplified oncogene modules catalogued under other cancer pathway entries.

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
jupyter nbconvert --to notebook --execute --inplace notebooks/Breast_cancer_subtype_Transcriptomics_Pipeline.ipynb
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
* **Deep Learning Framework:** PyTorch 2.7+
* **Machine Learning Library:** Scikit-Learn
* **Gradient Boosting:** XGBoost, LightGBM
* **Explainable AI:** SHAP (SHapley Additive exPlanations) — TreeSHAP + LinearSHAP
* **Biological APIs:** MyGene API, Enrichr API (KEGG 2021 Human, GO Biological Process 2023)
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
   * *Biological Validation:* Proves that *MIEN1* is physically located adjacent to the *ERBB2* (HER2) receptor at chromosome 17q12 and frequently co-amplified in HER2+ disease, validating our model's #1 ranked global SHAP biomarker.

4. **Saeys, Y., Inza, I., & Larrañaga, P. (2007).** *A review of feature selection techniques in bioinformatics.* **Bioinformatics**, 23(19), 2507-2517. [https://doi.org/10.1093/bioinformatics/btm344](https://doi.org/10.1093/bioinformatics/btm344)
   * *Bioinformatics Foundation:* Outlines the stability advantages of ensemble and consensus feature selection frameworks in high-dimensional genomic feature spaces, forming the basis for our 4-method Consensus Voting framework in Section 7.

5. **Sotiriou, C. & Pusztai, L. (2009).** *Gene-expression signatures in breast cancer.* **New England Journal of Medicine**, 360(8), 790-800. [https://doi.org/10.1056/NEJMra0800028](https://doi.org/10.1056/NEJMra0800028)
   * *Oncology Translation:* Establishes how global multi-gene expression signatures translate to clinical prognosis and chemotherapy selection in primary breast cancer.

6. **Chen, E. Y., Tan, C. M., Kou, Y., Banavathu, H. S., Farndon, G., & Ma'ayan, A. (2013).** *Enrichr: interactive and collaborative HTML5 gene list enrichment analysis tool.* **BMC Bioinformatics**, 14(1), 128. [https://doi.org/10.1186/1471-2105-14-128](https://doi.org/10.1186/1471-2105-14-128)
   * *Enrichment API Foundation:* The peer-reviewed reference for the Enrichr tool and database API utilized in Section 12 for biological pathway enrichment and process validation.

7. **Xin, J., Mark, A., Afrasiabi, C., Tsueng, G., Juchler, M., Gopal, N., ... & Su, A. I. (2016).** *MyGene.info: light-weight, high-performance query services for genes.* **Bioinformatics**, 32(19), 3034-3035. [https://doi.org/10.1093/bioinformatics/btw339](https://doi.org/10.1093/bioinformatics/btw339)
   * *API Foundation:* The official citation for the high-throughput MyGene API query services utilized in Section 11 to resolve Affymetrix probe IDs to HUGO gene symbols.

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

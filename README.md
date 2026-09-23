# OncoResolve: Breast Cancer Transcriptomics & Precision Molecular Subtyping Pipeline

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status: Publication-Grade](https://img.shields.io/badge/Status-Publication--Grade-green.svg)](#)

## Overview

**OncoResolve** is a research-grade computational biology and machine learning framework designed for breast cancer molecular subtyping (PAM50), biomarker discovery, explainability (SHAP), patient-level uniqueness scoring (WPR-CUS), systems biology co-expression networks, and multi-cohort external validation.

The pipeline integrates transcriptomic profiling across discovery ($N=784$), holdout ($N=197$), and three independent external validation cohorts ($N=2,263$ total):
- **SMC 2018 Cohort** ($N=168$, Samsung Medical Center RNA-seq)
- **SCAN-B / GSE96058 Cohort** ($N=339$, Sweden Cancerome Analysis Network)
- **METABRIC Cohort** ($N=1,756$, Microarray Validation Dataset)

---

## Key Methodological Innovations

1. **Full 2,000-Gene Candidate Model Feature Space**:
   - Models are trained on the top 2,000 differentially expressed genes (DEGs) selected leakage-free from the discovery cohort.
   - Dual Entrez Gene ID $\leftrightarrow$ HUGO Symbol resolution ensures 97–100% feature mapping across external platforms.

2. **Cross-Platform Z-Score Standardization & Scale Alignment**:
   - Eliminates platform-specific intensity and count baseline shifts without artificial metric manipulation:
     $$X_{\text{aligned}, j} = \mu_{\text{TCGA}, j} + \left(\frac{X_{\text{ext}, j} - \mu_{\text{ext}, j}}{\sigma_{\text{ext}, j}}\right) \cdot \sigma_{\text{TCGA}, j}$$
   - Unmapped features receive neutral training mean imputation ($\\mu_{\\text{TCGA}, j}$), scaling naturally to $0.0$ standard deviations.

3. **Systems Biology Gene Co-expression Network (GCN)**:
   - Self-contained NetworkX graph construction using consensus biomarkers, degree centrality, and Louvain modular community detection.
   - Identifies major hub genes (**AGR3**, **ESR1**, **PPP1R14C**, **FOXA1**, **CA12**, **SFRP1**, **VGLL1**).

4. **WPR-CUS: Weighted Projected Residual Composite Uniqueness Scoring**:
   - Quantifies N-of-1 patient heterogeneity and reconstruction residuals using out-of-fold (OOF) cross-patient Ridge regression.

---

## Model Generalization & External Validation Benchmarks

### 1. External Cohort Performance Summary

| Cohort | Model | Accuracy | Macro F1 | Weighted F1 | N Samples |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **SMC 2018** | **Random Forest** | **80.95%** | **0.8610** | **0.8050** | 168 |
| **SMC 2018** | **Logistic Regression** | **76.19%** | **0.7646** | **0.7470** | 168 |
| **SCAN-B (GSE96058)** | **Logistic Regression** | **89.38%** | **0.8385** | **0.8878** | 339 |
| **SCAN-B (GSE96058)** | **Random Forest** | **85.55%** | **0.8058** | **0.8488** | 339 |
| **METABRIC** | **Random Forest** | **76.42%** | **0.6747** | **0.7432** | 1,756 |
| **METABRIC** | **Logistic Regression** | **75.00%** | **0.6791** | **0.7329** | 1,756 |

### 2. Per-PAM50 Subtype Accuracy Highlights

- **Basal-like Subtype**: Precision **96.4–100.0%**, Recall **75.1–100.0%**, F1 **0.848–0.986** across all external cohorts.
- **HER2-Enriched Subtype**: F1 **0.675–0.849** across external cohorts.
- **Luminal A / Luminal B**: F1 **0.718–0.925** across external cohorts.

---

## Repository Structure

```
OncoResolve-Breast-Cancer-Transcriptomics/
├── notebooks/
│   ├── OncoResolve_Subtyping_and_Precision_Profiling.ipynb       # Main pipeline notebook (Sections 0-17)
│   ├── External_cohort_data_preparation_analysis.ipynb          # External cohort ingestion & cleaning
│   ├── OncoResolve_Model_Training_Validation_Pipeline.ipynb     # Model training & CV benchmark
│   └── OncoResolve_Temporal_and_Longitudinal_Analysis.ipynb    # Prognostic & survival analysis
├── data/
│   ├── processed/          # Cleaned & mapped parquet files
│   └── artifacts/          # Serialized models, scalers, and network graphs
├── results/                # Exported CSV performance audit tables
└── README.md
```

---

## Execution Guide

To execute the main subtyping pipeline:

```bash
# Clone repository
git clone https://github.com/shubhamkjha369/OncoResolve-Breast-Cancer-Transcriptomics.git
cd OncoResolve-Breast-Cancer-Transcriptomics

# Run main notebook via Jupyter or command line
jupyter nbconvert --to notebook --execute notebooks/OncoResolve_Subtyping_and_Precision_Profiling.ipynb
```

---

## Citation & License

This project is licensed under the MIT License.

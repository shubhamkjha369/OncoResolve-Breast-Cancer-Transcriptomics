<div align="center">

# OncoResolve: High-Hygiene Explainable AI and Patient-Centric Uniqueness Framework for Breast Cancer Subtyping

### An end-to-end RNA-seq transcriptomics, machine learning, and N-of-1 precision oncology pipeline for classifying PAM50 breast cancer molecular subtypes with SHAP explainability and cross-platform external validation.

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn 1.4+](https://img.shields.io/badge/Scikit--Learn-1.4+-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-Primary_Classifier-green?style=flat)](#)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-blueviolet?style=flat)](#)
[![Preprint PDF](https://img.shields.io/badge/Preprint-PDF-red?style=flat&logo=adobeacrobatreader&logoColor=white)](pre-print/OncoResolve.pdf)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Shubham K. Jha · AI Data Scientist & Computational Biology Independent Researcher**

[![GitHub](https://img.shields.io/badge/GitHub-shubhamkjha369-black?logo=github)](https://github.com/shubhamkjha369)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://www.linkedin.com/in/shubhamjha369/)
[![Email](https://img.shields.io/badge/Email-Contact-red?logo=gmail)](mailto:shubhamkjha369@gmail.com)

</div>

---

## Table of Contents

- [Abstract](#abstract)
- [📄 Pre-Print Paper & Manuscript](#preprint-paper)
- [📌 Literature Gap Analysis Matrix](#literature-matrix)
- [🛡️ Codebase Quality & System Audit](#system-audit)
- [Project Aim](#project-aim)
- [Pipeline Workflow & Architecture](#pipeline-workflow)
- [1. Patient Cohorts & Dataset Specifications](#patient-cohorts)
- [2. Biomarker Discovery: 211 Consensus Signature](#biomarker-discovery)
- [3. Subtype Predictors & Multi-Cohort Performance](#subtype-predictors)
- [4. Explainable AI (SHAP)](#explainable-ai)
- [5. N-of-1 Personal Profiling: Baseline CUS vs. WPR-CUS](#personal-profiling)
- [6. Pathway Divergence & Precision Signatures](#pathway-divergence)
- [7. Prognosis & Outcomes: Consensus Ridge Risk Score (CRS)](#prognosis-outcomes)
- [8. Biological Validation (CRISPR & LINCS)](#biological-validation)
- [🐳 Reproducibility & Quick Start Guide](#reproducibility)
- [References & Citation](#references)

---

<a id="abstract"></a>
## Abstract

Breast cancer exhibits high molecular heterogeneity, traditionally stratified into five PAM50 intrinsic molecular subtypes (Basal-like, HER2-enriched, Luminal A, Luminal B, and Normal-like) that dictate therapeutic response and clinical management. While transcriptomics-based machine learning classifiers show promise in precision oncology, high-dimensional dataset challenges—particularly feature selection leakage and cross-platform domain shifts—frequently inflate reported performances and hinder external generalizability.

**OncoResolve (v3.4.0)** introduces an end-to-end, high-hygiene computational framework designed to resolve these challenges. Using the **TCGA-BRCA Pan-Cancer Atlas** dataset ($N=981$ post-QC RNA-seq profiles retrieved via NCI Genomic Data Commons with STAR-based quantification and GRCh38 assembly), OncoResolve enforces a strict **Anti-Leakage Protocol (ALP)** where all scaling, missing value imputation, and feature selection are performed strictly inside cross-validation training partitions. A tri-method ensemble selector locks a **211-gene consensus biomarker signature** (211 unique HUGO gene symbols mapped from Entrez IDs).

Evaluated across four independent international cohorts ($N=5,960$ total patient profiles), OncoResolve achieves **88.83%** accuracy (**0.8520** Macro F1, **0.9714** ROC-AUC) for LightGBM and **85.79%** accuracy (**0.8182** Macro F1, **0.9773** ROC-AUC) for Linear SVM on the locked TCGA holdout split ($N=197$). On independent external cohorts with zero model retraining, OncoResolve yields **80.59%** accuracy on SCAN-B ($N=3,055$, RNA-seq), **80.36%** accuracy on SMC 2018 ($N=168$, RNA-seq), and **72.78%** accuracy on METABRIC ($N=1,756$, Illumina HT-12 microarray). 

Beyond categorical classification, OncoResolve formulates the **Weighted Projected Residual Composite Uniqueness Score (WPR-CUS)**, an N-of-1 patient profiling metric that combines out-of-column analytical Leave-One-Out (LOO) leverage-corrected residuals with eigen-projected directional distance (WPRD) and Patient Similarity Network (PSN) graph topology. WPR-CUS expands the Luminal A vs. Luminal B uniqueness distance gap by **36.3-fold** ($\Delta = 0.6798$ vs. $0.0187$) over baseline CUS, identifies transcriptomic outliers with $0.9745$ AUC, and predicts independent overall survival risk ($p = 0.0378$). Furthermore, pathway divergence analysis reveals **zero Jaccard overlap ($0.0000$)** between private outlier pathways and population baselines, proving that patient-specific biology diverges fundamentally from global subtype profiles.

---

<a id="preprint-paper"></a>
## 📄 Pre-Print Paper & Manuscript

The complete research manuscript detailing theoretical foundations, mathematical formulations, Anti-Leakage Protocol (ALP), dual-architecture explainability, WPR-CUS formulation, and multi-cohort validation results is available in the [`pre-print`](pre-print/) directory:

- 📄 **Manuscript PDF**: [`pre-print/OncoResolve.pdf`](pre-print/OncoResolve.pdf)
- 📝 **LaTeX Source Code**: [`pre-print/OncoResolve.tex`](pre-print/OncoResolve.tex)

### Key Methodological Highlights:
1. **Dataset Rigor**: Primary TCGA-BRCA dataset built on NCI GDC STAR-based quantification aligned to GRCh38 ($N=981$ post-QC; $N=784$ discovery, $N=197$ holdout).
2. **Anti-Leakage Protocol (ALP)**: Enforces 100% fold-contained Z-score scaling and feature selection inside cross-validation training loops, completely eliminating target leakage.
3. **Dual-Architecture Explainable AI**: Attributions independently audited across Linear Support Vector Machine ($C=0.1$) and LightGBM ($n\_estimators=100, learning\_rate=0.05$), selecting LightGBM as primary classifier via leak-free CV outer-fold mean Macro F1.
4. **WPR-CUS Framework**: Fuses out-of-column analytical LOO leverage-corrected residuals ($e_{i,g} = \frac{x_{i,g} - \hat{x}_{i,g}}{1 - h_{ii,g}}$) with Weighted Projected Residual Distance (WPRD) across eigen-axes, expanding LumA vs LumB separation gap to **0.6798** (36.3-fold expansion).
5. **Pathway Divergence ($0.0000$ Jaccard Overlap)**: Demonstrates zero pathway overlap between private outlier signatures and global subtype pathways.
6. **Prognostic Cox Risk Score (CRS)**: Continuous survival risk score evaluated on TCGA-BRCA ($C\text{-index}=0.7266$), SCAN-B ($C\text{-index}=0.6401$), and METABRIC ($C\text{-index}=0.5551$).

---

<a id="literature-matrix"></a>
## 📌 Literature Gap Analysis Matrix

| Landmark Reference | Prior Art & Standard Findings | Unaddressed Research Gap | Exact OncoResolve Solution & Results |
| :--- | :--- | :--- | :--- |
| **Perou et al. / Parker et al.** (*J Clin Oncol* 2009) | PAM50 nearest-centroid Spearman rank subtyping. | **Rigid Categorical Stratification**: Ignores within-subtype heterogeneity and N-of-1 biological outliers. | **WPR-CUS Framework**<br>Fuses analytical LOO residuals with WPRD distance.<br>• Expands LumA vs LumB gap by **36.3-fold** ($\Delta=0.6798$). |
| **Kapoor & Narayanan** (*Patterns* 2022) | Exposed widespread data leakage in genomic ML. | **Benchmark Inflation**: Global feature selection before CV inflates accuracy (>95%) which collapses on external test data. | **Anti-Leakage Protocol (ALP)**<br>100% fold-contained scaling and feature selection.<br>• Holdout Acc: **88.83%** (LightGBM), **85.79%** (SVM). |
| **Curtis et al.** (*Nature* 2012 / METABRIC) | Identified genomic subclonal clusters in 2,000 tumors. | **Platform Domain Shift**: Direct transfer from RNA-seq to microarray suffers from probe loss and cross-platform shift. | **Local Z-Score Standardization**<br>Zero-retraining transfer across 3 independent cohorts.<br>• METABRIC (Microarray): **72.78%** Acc (LightGBM). |
| **Sammut et al.** (*Nature* 2022) | Multi-omic ML predicting neoadjuvant response. | **Prospective N-of-1 Barrier**: Joint co-normalization across full datasets precludes prospective single-sample subtyping. | **Independent Cohort Standardizer**<br>Standardizes test observations zero-retraining.<br>• SCAN-B ($N=3,055$): **80.59%** Acc.<br>• SMC 2018 ($N=168$): **80.36%** Acc. |
| **Haibe-Kains et al.** (*Bioinformatics* 2020) | Evaluated single-sample subtyping tools (`Genefu`). | **Decoupled Risk Modeling**: Subtyping models remain decoupled from continuous survival risk scoring. | **Consensus Ridge Cox Risk Score (CRS)**<br>Ridge-regularized survival predictor on 211 genes.<br>• TCGA $C\text{-index} = 0.7266$ ($p < 0.001$). |

---

<a id="patient-cohorts"></a>
## 1. Patient Cohorts & Dataset Specifications

We evaluated OncoResolve across 4 primary clinical cohorts spanning **5,960 total patients**:

| Cohort Name | Platform Type | Total Patients ($N$) | Mapped Genes | Quantification / Assembly | Primary Role |
| :--- | :--- | :---: | :---: | :--- | :--- |
| **TCGA-BRCA** | Illumina HiSeq RNA-seq | **981** (784 train / 197 test) | 211 / 211 | STAR Counts / GRCh38 (NCI GDC) | Discovery & Locked Holdout |
| **SCAN-B (GSE96058)** | Illumina NextSeq RNA-seq | **3,055** | 211 / 211 | FPKM / GRCh38 | External Independent RNA-seq |
| **SMC 2018** | Illumina HiSeq RNA-seq | **168** | 211 / 211 | RSEM / GRCh38 | External Asian Cohort |
| **METABRIC** | Illumina HT-12 Microarray | **1,756** | 178 / 211 (84.4%) | Log2 Intensity / HT-12 v3 | External Microarray Domain Shift |

---

<a id="biomarker-discovery"></a>
## 2. Biomarker Discovery: 211 Consensus Signature

Out of ~18,000 protein-coding transcripts, our **tri-method ensemble selector** isolates genes nominated by $\ge 2/3$ orthogonal methods inside cross-validation loops:
- **ANOVA F-test**: Parametric variance ratio across subtype classes.
- **LASSO L1 Logistic Regression**: Sparse L1 feature penalty ($C=0.1, \text{SAGA solver}$).
- **Random Forest Gini Importance**: Non-linear tree impurity reduction ($N_{\text{trees}}=200$).

The locked **211 consensus biomarker signature** maps 211 Entrez IDs to 211 unique HUGO gene symbols (e.g., *ESR1*, *ERBB2*, *KRT5*, *KRT17*, *FOXA1*, *MKI67*, *GRB7*, *GATA3*, *TFF1*, *SOX10*, *KLK6*, *CASP14*, *S100A7A*). Selection stability was verified via fold-contained bootstrap resampled CV ($B=50$, Jaccard Stability Index JSI = $0.5869 \pm 0.0347, p < 0.001$).

---

<a id="subtype-predictors"></a>
## 3. Subtype Predictors & Multi-Cohort Performance

On the unseen **TCGA Holdout** split ($N=197$), **LightGBM (Primary Classifier)** achieved **88.83%** accuracy (**0.8520** Macro F1, **0.9714** OvR ROC-AUC) and **Linear SVM** achieved **85.79%** accuracy (**0.8182** Macro F1, **0.9773** OvR ROC-AUC).

### Multi-Cohort Diagnostic Benchmark Table

| Cohort Partition | $N$ | Mapped Features | Classifier Model | Accuracy | Macro F1 | OvR ROC-AUC | Status |
| :--- | :---: | :---: | :--- | :---: | :---: | :---: | :--- |
| **TCGA Holdout** | 197 | 211 / 211 | **LightGBM (Primary)** | **88.83%** | **0.8520** | **0.9714** | Unseen Holdout |
| **TCGA Holdout** | 197 | 211 / 211 | **Linear SVM** | **85.79%** | **0.8182** | **0.9773** | Unseen Holdout |
| **SMC 2018** | 168 | 211 / 211 | **LightGBM (Primary)** | **80.36%** | **0.8104** | **0.9410** | External RNA-seq |
| **SCAN-B** | 3,055 | 211 / 211 | **LightGBM (Primary)** | **80.59%** | **0.7717** | **0.9450** | External RNA-seq |
| **METABRIC** | 1,756 | 178 / 211 | **LightGBM (Primary)** | **72.78%** | **0.5843** | **0.8650** | External Microarray |
| **METABRIC** | 1,756 | 178 / 211 | **Linear SVM** | **70.79%** | **0.5970** | **0.8810** | External Microarray |

---

<a id="explainable-ai"></a>
## 4. Explainable AI (SHAP)

SHAP attributions computed across LightGBM and Linear SVM confirm model-agnostic biological driver consensus:
- **ESR1**: Primary positive driver for Luminal A and Luminal B subtyping.
- **ERBB2 & GRB7**: Co-amplified 17q12 drivers for HER2-enriched subtyping.
- **KRT5, KRT17, CASP14, KLK6**: Structural cytokeratins and basal drivers for Basal-like (TNBC) subtyping.
- **FOXA1 & GATA3**: Pioneer transcription factors for ER-positive luminal subtyping.

Per-subtype top driver barplots with exact mapped HUGO gene symbols are generated and archived in `pre-print/images/shap_drivers_*.png`.

---

<a id="personal-profiling"></a>
## 5. N-of-1 Personal Profiling: Baseline CUS vs. WPR-CUS

We audit the evolution from the **Baseline CUS** formulation to the advanced **Weighted Projected Residual Composite Uniqueness Score (WPR-CUS)**.

### Mathematical Formulation Comparison

#### 1. Baseline (Standard) CUS Formulation
$$\text{Baseline CUS: } u_{i, \text{Baseline}} = \frac{1}{3}\bar{u}_{i, \text{MSE}} + \frac{1}{3}\bar{u}_{i, \text{PSN Isolation}} + \frac{1}{3}\bar{u}_{i, \text{Centroid Distance}}$$
where raw LOO MSE is $\bar{u}_{i, \text{MSE}} = \frac{1}{P} \sum_{g=1}^P (x_{i,g} - \hat{x}_{i,g})^2$ and centroid distance is $d_{i, \text{Centroid}} = \|\mathbf{x}_i - \boldsymbol{\mu}_k\|_2$.
- *Drawback*: Fails to adjust for leverage diagonals $h_{ii,g}$, assumes isotropic Euclidean variance, and collapses LumA vs LumB distance gap to $\Delta = 0.0187$.

#### 2. Weighted Projected Residual CUS (WPR-CUS) Formulation
$$\text{WPR-CUS: } u_{i, \text{WPR-CUS}} = \frac{1}{3}\bar{u}_{i, \text{Attrib}} + \frac{1}{3}\bar{u}_{i, \text{WPRD}} + \frac{1}{3}\bar{u}_{i, \text{Manifold}}$$

Where:
- **Leverage-Corrected Analytical LOO Residuals**:
  $$e_{i,g} = \frac{x_{i,g} - \hat{x}_{i,g}^{\setminus g}}{1 - h_{ii,g}}$$
  (Matches brute-force $N \times P$ refitting with max error $< 3.05 \times 10^{-13}, r = 1.000000$).
- **Weighted Projected Residual Distance (WPRD)**:
  $$D_{\boldsymbol{\Theta}}(i, j) = \sum_{l=1}^L w_l^{\text{Eigen}} \cdot \left| \boldsymbol{\theta}_l^T (\mathbf{e}_i - \mathbf{e}_j) \right|$$

### Empirical Benchmark Summary: Baseline CUS vs. WPR-CUS

| Metric Category | Existing Baseline CUS | **WPR-CUS (Fold-Safe OOF)** | Verified Empirical Verdict |
| :--- | :---: | :---: | :--- |
| **LOO Formula Accuracy** | Approximated | **Machine Precision ($3.05 \times 10^{-13}$ Error)** | Verified against brute-force refitting ($r = 1.0000$) |
| **LumA vs. LumB OOF Gap ($\Delta$)** | $0.0187$ | **$0.6798$** | **36.3-fold gap expansion** over baseline |
| **ANOVA $F$-Statistic ($p$-value)** | $27.85$ ($p = 1.23 \times 10^{-21}$) | **$95.99$ ($p = 2.13 \times 10^{-66}$)** | Unbiased out-of-sample parametric group separation |
| **Permutation Test $p_F$ (1000x)** | $p_{\text{perm}} < 0.001$ | **$p_{\text{perm}} = 0.000999$ ($<0.001$)** | Observed $F=95.99$ vastly exceeds max null $F_{\text{null}}=4.64$ |
| **Kruskal-Wallis $H$-Statistic** | $97.08$ ($p = 4.12 \times 10^{-20}$) | **$259.91$ ($p = 4.78 \times 10^{-55}$)** | Unbiased non-parametric subtype separation |
| **Synthetic Outlier Patient AUC** | N/A | **$0.9745$** | Near-perfect anomaly patient identification |
| **Synthetic Spike Gene Recovery** | N/A | **$1.0000$** | 100% true spike feature attribution recovery |

---

<a id="pathway-divergence"></a>
## 6. Pathway Divergence & Precision Signatures

To test whether patient-specific uniqueness residuals capture genuine individualized biology rather than noise, we execute pathway enrichment on **Uniqueness Residuals** ($\mathbf{E} = \mathbf{X} - \hat{\mathbf{X}}$) versus global population subtype DEGs.

- **Empirical Result**: **Jaccard Overlap Index = $0.0000$** (0 shared pathways out of 6 enriched private outlier pathways).
- **Enriched Private Outlier Pathways**: KRAS Signaling Dn ($p = 5.25 \times 10^{-6}$), Estrogen Response Early, Defective GALNT3, Defective GALNT12, Defective C1GALT1C1, Termination of O-glycan Biosynthesis.
- **Scientific Conclusion**: **Population Biology (Subtypes) $\neq$ Patient-Specific Biology (Outliers)**. WPR-CUS isolates orthogonal, actionable patient-specific drug targets beyond standard subtype labels.

---

<a id="prognosis-outcomes"></a>
## 7. Prognosis & Outcomes: Consensus Ridge Cox Risk Score (CRS)

An L2-regularized **Ridge Cox Proportional Hazards** model trained on the 211 consensus genes yields a continuous prognostic risk score (CRS):

- **TCGA-BRCA OS Harrell C-index**: **0.7266** ($p = 3.65 \times 10^{-6}$)
- **SCAN-B OS Harrell C-index**: **0.6401** ($p = 2.14 \times 10^{-5}$)
- **METABRIC OS Harrell C-index**: **0.5551** ($p = 0.0210$)

*Schoenfeld residual testing revealed non-proportional hazards in Basal-like tumors ($p = 0.0162$), reflecting early 36-month mortality risk characteristic of triple-negative breast cancer.*

---

<a id="biological-validation"></a>
## 8. Biological Validation (CRISPR & LINCS)

Validation against Broad DepMap CRISPR-Cas9 essentiality (Chronos scores) confirms essentiality for consensus driver genes in subtype-matched cell lines (*ERBB2* in HER2-amplified lines, *ESR1* and *FOXA1* in ER-positive luminal lines). LINCS L1000 connectivity analysis identifies targeted compounds (trastuzumab, tamoxifen, fulvestrant, palbociclib) with strong tau reversal scores.

---

<a id="reproducibility"></a>
## 🐳 Reproducibility & Quick Start Guide

### Step 1 — Download All Raw Datasets
```bash
python data/external_cohort/download_external_cohorts.py
```

### Step 2 — Run Main Subtyping & Precision Profiling Notebook
Open and run all sections of:
```
notebooks/OncoResolve_Subtyping_and_Precision_Profiling.ipynb
```

### Step 3 — Run Model Training & Validation Pipeline Notebook
Open and run all sections of:
```
notebooks/OncoResolve_Model_Training_Validation_Pipeline.ipynb
```

---

<a id="references"></a>
## References & Citation

```bibtex
@article{jha2026oncoresolve_preprint,
  author       = {Shubham K. Jha},
  title        = {OncoResolve: High-Hygiene Explainable AI and Patient-Centric Uniqueness Framework for Breast Cancer Subtyping},
  journal      = {Preprint},
  year         = {2026},
  note         = {Available at: pre-print/OncoResolve.pdf},
  url          = {https://github.com/shubhamkjha369/OncoResolve-Breast-Cancer-Transcriptomics/blob/main/pre-print/OncoResolve.pdf}
}
```

## License
Licensed under the [MIT License](LICENSE).

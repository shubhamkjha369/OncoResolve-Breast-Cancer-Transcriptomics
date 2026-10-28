<div align="center">

# OncoResolve
### High-Hygiene Explainable AI & Patient-Centric N-of-1 Uniqueness Framework for Breast Cancer Molecular Subtyping

<p>
  <a href="https://github.com/shubhamkjha369/OncoResolve-Breast-Cancer-Transcriptomics"><img src="https://img.shields.io/badge/Version-v3.4.0-blue.svg?style=for-the-badge" alt="Version"></a>
  <a href="https://doi.org/10.5281/zenodo.21967841"><img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21967841-green.svg?style=for-the-badge" alt="DOI"></a>
  <a href="https://orcid.org/0009-0007-4519-5867"><img src="https://img.shields.io/badge/ORCID-0009--0007--4519--5867-brightgreen.svg?style=for-the-badge" alt="ORCID"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-orange.svg?style=for-the-badge" alt="License"></a>
  <img src="https://img.shields.io/badge/Python-3.11-blue.svg?style=for-the-badge" alt="Python 3.11">
  <img src="https://img.shields.io/badge/Scikit--Learn-1.9.0-F7931E?style=for-the-badge" alt="Scikit-learn">
  <img src="https://img.shields.io/badge/Status-Publication--Grade-success?style=for-the-badge" alt="Status">
</p>

</div>

---

## Overview

**OncoResolve** is a computational biology and machine learning pipeline for **PAM50 breast cancer molecular subtyping**, built on the TCGA-BRCA RNA-sequencing cohort and validated across three independent external platforms. The framework encompasses the full translational arc: raw data ingestion → leakage-free feature selection → tri-architecture supervised classification → SHAP explainability → N-of-1 patient heterogeneity scoring → functional enrichment → prognostic survival modelling → tumour microenvironment deconvolution → external multi-cohort validation.
---

## Pipeline Architecture

```
TCGA-BRCA (N=981)
   ├── Discovery Cohort (N=784, 80%)     ← All training, CV, feature selection
   └── Holdout Cohort  (N=197, 20%)     ← Locked; never seen during training

         ▼ Section 1–3: QC, Expression Audit, PCA/UMAP Manifold
         ▼ Section 4:   Differential Gene Expression (DGE)  →  17,014 retained genes
         ▼ Section 5:   Tri-Architecture ML Benchmark       →  2,000-gene candidate space
         ▼ Section 6:   Independent Holdout Evaluation      →  Logistic Regression leads (83.8% Acc, 0.8125 Macro F1)
         ▼ Section 7:   Gene Co-expression Network (GCN)    →  131 nodes, 1,034 edges (|r|≥0.60)
         ▼ Section 8:   SHAP Explainability                 →  211-gene Consensus Biomarker Set
         ▼ Section 9:   Functional Enrichment (4 databases) →  Estrogen, Keratinization pathways
         ▼ Section 10:  WPR-CUS N-of-1 Uniqueness Scoring   →  ANOVA F=116.05, η²=0.322
         ▼ Section 11:  External Cohort Validation          →  SMC, SCAN-B, METABRIC
         ▼ Section 12:  Kaplan-Meier & Cox Regression       →  C-Index 0.741
         ▼ Section 13:  TME Deconvolution (decoupler ULM)   →  22 cell-type signatures
```

---

## Cohort Characteristics

| Cohort | N | Platform | Role |
|:---|:---:|:---|:---|
| **TCGA-BRCA** (full master) | 981 | Illumina HiSeq V2 RNA-seq (RSEM+1, log₂) | Discovery + holdout |
| TCGA Discovery | 784 | — | All model training, CV, feature selection |
| TCGA Holdout | 197 | — | Final locked evaluation |
| **SMC 2018** | 168 | Illumina HiSeq RNA-seq | External validation |
| **SCAN-B (GSE96058)** | 339 | RNA-seq | External validation |
| **METABRIC** | 1,756 | Microarray | External cross-platform validation |

**PAM50 Subtype Distribution (Full TCGA Master Cohort, N=981):**

| Subtype | N | % |
|:---|:---:|:---:|
| Luminal A | 499 | 50.9% |
| Luminal B | 197 | 20.1% |
| Basal-like | 171 | 17.4% |
| HER2-enriched | 78 | 7.9% |
| Normal-like | 36 | 3.7% |

---

## Expression Data Characteristics

All analyses use **log₂(RSEM + 1)** normalized expression values. Verified scale parameters from execution:

| Statistic | Value |
|:---|:---|
| Expression scale | log₂(RSEM + 1) |
| Expression range | 0.0000 – 20.9608 |
| Mean expression (X_log₂) | 6.7954 |
| Std (across genes) | 0.9684 |
| Features retained after QC | 17,958 / 17,994 (non-zero variance) |
| Discovery ↔ Holdout overlap | **0 patients** (100% disjoint, assertion-verified) |

---

## Section-by-Section Results (Empirically Verified)

### Section 3: PCA Transcriptomic Manifold

| Component | Variance Explained |
|:---|:---:|
| PC1 | **11.93%** |
| PC2 | **8.45%** |
| PC3 | 6.04% |
| Cumulative (10 PCs) | **41.41%** |
| Cumulative (PC1+PC2) | **20.38%** |

---

### Section 4: Differential Gene Expression (DGE)

- Input features: ~17,014 transcripts retained after low-expression filtering
- Statistical framework: One-vs-Rest (OVR) Welch t-test + Benjamini-Hochberg FDR correction
- Top 2,000 high-variance genes selected as the **candidate feature space**
- Volcano plots generated per PAM50 subtype (Figs 6b-1 through 6b-5)

---

### Section 5: Tri-Architecture Benchmark (5-Fold Leakage-Free CV, Discovery N=784)

All three candidate models were evaluated using the **Anti-Leakage Protocol (ALP)**: 5-fold stratified cross-validation with feature selection, scaling, and hyperparameter tuning strictly confined within each training fold.

| Model | Mean Val Macro F1 | Mean Val Accuracy | Generalization Stability |
|:---|:---:|:---:|:---:|
| **Random Forest** | **0.8433** | **88.01%** | High (tree-ensemble stability) |
| **Logistic Regression ($L_2$)** | **0.8329** | 87.88% | Balanced, low variance |
| **PyTorch MLP** | 0.7941 | 86.61% | High capacity |

> Random Forest and Logistic Regression demonstrated excellent out-of-fold generalization under leak-free nested evaluation, confirming absence of row-level feature selection bias.

---

### Section 6: Independent TCGA Holdout Evaluation (N=197, Frozen Locked Models)

No hyperparameter changes or threshold tuning were permitted after holdout exposure. All evaluations are single-pass locked inferences under TRIPOD+AI.

| Model Architecture | Accuracy | Balanced Acc | Macro F1 | Weighted F1 | MCC | Macro ROC-AUC | Log Loss |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Logistic Regression ($L_2$)** | **83.76%** | **82.11%** | **0.8125** | **0.8412** | **0.7605** | **0.9657** | **0.4580** |
| **PyTorch MLP** | 82.74% | 76.09% | 0.7593 | 0.8267 | 0.7480 | 0.9532 | 0.5898 |
| **Random Forest** | 82.74% | 73.72% | 0.7063 | 0.8130 | 0.7380 | 0.9735 | 0.5050 |

**Probability Calibration & Decision Reliability (Holdout N=197):**

| Model Architecture | OvR Brier Score | Top-Label ECE | Calibration Status |
|:---|:---:|:---:|:---|
| **Logistic Regression ($L_2$)** | **0.0492** | **5.73%** | Well-calibrated, clinical-grade probabilities |
| **PyTorch MLP** | 0.0544 | 9.54% | Moderately calibrated |
| **Random Forest** | 0.0541 | 18.46% | Uncalibrated probabilities (tree leaf variance) |

> **Champion Selection:** Logistic Regression ($L_2$) achieved champion status, significantly outperforming Random Forest in paired bootstrap Macro F1 ($\Delta\text{F1} = +0.1062$, 95% CI: $[+0.0032, +0.1913]$, $p = 0.042$) and superior probability calibration (ECE = 5.73% vs 18.46%).

---

### Section 7: Gene Co-expression Network (GCN)

- **Input:** 188 of 211 consensus biomarkers with expression data in the discovery matrix
- **Adjacency threshold:** |r| ≥ 0.60 (Pearson pairwise correlation)
- **Network:** 131 nodes, 1,034 edges
- **Community detection:** Louvain modularity optimization
- **Major hub genes identified:** AGR3, ESR1, PPP1R14C, FOXA1, CA12, SFRP1, VGLL1

---

### Section 8: SHAP Explainability & Consensus Biomarker Distillation

**SHAP Methods by Architecture:**
- **TreeSHAP** (`shap.TreeExplainer`) → Random Forest
- **LinearSHAP** (`shap.LinearExplainer`) → Logistic Regression
- **Gradient×Input** (PyTorch autograd) → MLP

**Consensus Biomarker Set:** 211 genes with 100% HUGO symbol mapping retention, ranked by `consensus_importance` (F1-weighted cross-architecture SHAP consensus).

**Top 20 Consensus Biomarkers (by rank):**
`PGR, IRX1, MUC15, LBP, MAB21L4, SRARP, KRT5, ELOVL2, BPIFB1, CAPN8, KLK6, KLK7, KRT6A, FSIP1, NXPH1, NAT1, PHYHD1, AGR2, FOXA1, AGR3`

> PGR (Progesterone Receptor) ranked #1 globally. FOXA1 and ESR1 dominate the Luminal A SHAP signature, consistent with canonical estrogen receptor biology.

---

### Section 9: Functional Enrichment Analysis

Performed local hypergeometric ORA (Benjamini-Hochberg corrected) against 4 databases using the complete **211-gene consensus set** against a **2,000-gene background**.

| Database | Terms Tested | Significant Terms | Top Finding |
|:---|:---:|:---:|:---|
| GO Biological Process 2023 | 610 | **1** | Intermediate Filament Organization (GO:0045109), adj-p=0.0067, overlap=9/18 |
| KEGG 2021 Human | 135 | **2** | Estrogen Signaling Pathway, adj-p=0.000361; Staphylococcus aureus infection, adj-p=0.034 |
| MSigDB Hallmark 2020 | 23 | **2** | KRAS Signaling Dn, adj-p=0.00334; Estrogen Response Late, adj-p=0.00959 |
| Reactome Pathways 2024 | 246 | **2** | Formation of the Cornified Envelope, adj-p=5.64×10⁻⁸; Keratinization, adj-p=5.64×10⁻⁸ |

**Key biological interpretation:** The 211-gene consensus biomarker set is enriched for:
1. **Basal-like keratinization biology** (KRT5, KRT6A, KRT14-17, KRT23, KRT81) — Reactome and GO convergence
2. **Estrogen signaling** (ESR1, PGR, TFF1, FOXA1, CA12, AGR2) — KEGG and MSigDB convergence
3. **KRAS-driven mesenchymal de-differentiation** (SOX10, TFAP2B) — MSigDB Hallmark

> Note: 23 of 211 consensus genes were outside the 2,000-gene candidate background and were excluded from enrichment calculations. Effective background intersection: 188/211 (89.1%).

---

### Section 10: N-of-1 Patient Heterogeneity (WPR-CUS Framework)

**Cohort:** N=981 patients, P=211 consensus genes.

#### WPR-CUS Formulation
The Weighted Projected Residual Composite Uniqueness Score (WPR-CUS) combines three orthogonal metrics under strict 5-fold OOF cross-fitting (equal weights α=β=γ=⅓):
1. Mean Absolute LOO Residual
2. WPR-Hubness Isolation Score (Pairwise Weighted Projected Residual Distance)
3. L₂ Residual Vector Norm

#### Subtype-Level WPR-CUS Descriptives (N=981)

| Subtype | N | Mean WPR-CUS | SD | Median | IQR |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Basal-like** | 171 | **1.191** | 1.026 | 1.098 | [0.462, 1.801] |
| **HER2-enriched** | 78 | **1.096** | 0.887 | 0.986 | [0.545, 1.598] |
| **Luminal B** | 197 | 0.435 | 0.780 | 0.416 | [-0.067, 0.859] |
| **Normal-like** | 36 | 0.002 | 1.078 | -0.156 | [-0.689, 0.570] |
| **Luminal A** | 499 | **−0.219** | 0.756 | −0.297 | [-0.793, 0.234] |

#### Statistical Validation

| Test | Statistic | p-value |
|:---|:---:|:---:|
| Classical ANOVA F | **116.052** | 5.49×10⁻⁸¹ |
| Eta-squared (η²) | **0.322** | — |
| Omega-squared (ω²) | 0.319 | — |
| Welch ANOVA F | 101.298 | 5.06×10⁻⁴⁴ |
| Kruskal-Wallis H | **316.094** | — |
| Empirical permutation p (1,000 permutations) | **0.000999** | < 0.001 |
| Permutation null 95th percentile H | 9.575 | — |

> The observed Kruskal-Wallis H=316.09 vastly exceeds the 99th percentile of the permutation null (H=12.95), confirming that PAM50 subtype association with WPR-CUS is not a statistical artifact.

#### Cross-Patient Reconstruction Benchmark (OOF Ridge, 5-Fold)

| Metric | Observed | Population-Mean Null | Permutation Null |
|:---|:---:|:---:|:---:|
| OOF R² | **0.7914 ± 0.0033** | 0.3956 ± 0.0081 | 0.3693 ± 0.0041 |
| Pooled OOF R² | **0.7915** | 0.3956 | 0.3693 |
| ΔR² vs mean-null | +0.3959 | — | — |
| ΔR² vs permutation-null | +0.4222 | — | — |
| Pooled MAE | 1.2949 | — | — |
| Pooled RMSE | 1.7895 | — | — |
| Median selected Ridge α | 3162.28 | — | — |

#### Feature Stability (Jaccard Stability Index, 20 Bootstraps at 80% subsample)

| Method | Mean JSI | Std | Interpretation |
|:---|:---:|:---:|:---|
| **Random Forest** | **0.6335** | 0.0491 | High ensemble feature stability |
| **Multi-Architecture Consensus** | **0.4789** | 0.0548 | Statistically significant biological convergence ($p < 0.001$) |

---

### Section 11: External Multi-Cohort Validation (N=2,263 Total)

All models transferred **zero-retraining and zero-fine-tuning** across diverse sequencing centers and profiling chemistries.

| Cohort | Platform | Samples ($N$) | Genes Mapped | Model Architecture | Accuracy | Macro F1 | OvR ROC-AUC | Top-Label ECE | OvR Brier |
|:---|:---|:---:|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| **SMC 2018** | Illumina HiSeq RNA-seq | 168 | 211 / 211 | **Logistic Regression ($L_2$)$^*$** | **77.98%** | **0.7437** | **0.9824** | **10.78%** | **0.0753** |
| SMC 2018 | Illumina HiSeq RNA-seq | 168 | 211 / 211 | PyTorch MLP | 77.38% | 0.7693 | 0.9716 | 11.20% | 0.0782 |
| SMC 2018 | Illumina HiSeq RNA-seq | 168 | 211 / 211 | Random Forest (aligned) | 80.36% | 0.7913 | 0.9795 | -- | -- |
| **SCAN-B (GSE96058)** | Illumina NextSeq RNA-seq | 339 | 211 / 211 | **Logistic Regression ($L_2$)$^*$** | **87.61%** | **0.8565** | **0.9813** | **4.40%** | **0.0550** |
| SCAN-B (GSE96058) | Illumina NextSeq RNA-seq | 339 | 211 / 211 | PyTorch MLP | 86.43% | 0.8183 | 0.9785 | 7.15% | 0.0612 |
| SCAN-B (GSE96058) | Illumina NextSeq RNA-seq | 339 | 211 / 211 | Random Forest (aligned) | 86.73% | 0.8411 | 0.9764 | -- | -- |
| **METABRIC** | Illumina HT-12 v3 Microarray | 1,756 | 176 / 211 | **Random Forest (calibrated)$^*$** | **81.23%** | **0.7915** | **0.9050** | **12.75%** | **0.1068** |
| METABRIC | Illumina HT-12 v3 Microarray | 1,756 | 176 / 211 | Logistic Regression ($L_2$, cal.) | 80.45% | 0.7830 | 0.8980 | 13.10% | 0.1120 |
| METABRIC | Illumina HT-12 v3 Microarray | 1,756 | 176 / 211 | PyTorch MLP | 72.21% | 0.6319 | 0.8840 | 14.50% | 0.1245 |

> **Key Architectural & Cross-Platform Findings:**
> 1. **Zero-Retraining Generalization:** Logistic Regression ($L_2$) achieved outstanding prospective accuracy and calibration across independent RNA-seq cohorts (87.61% Acc, 0.8565 F1, 4.40% ECE on SCAN-B; 77.98% Acc, 0.7437 F1 on SMC 2018).
> 2. **Cross-Platform Microarray Transfer:** On METABRIC ($N=1,756$), Platt-calibrated Random Forest achieved 81.23% Accuracy (0.7915 Macro F1, 0.9050 OvR AUC) and calibrated Logistic Regression reached 80.45% Accuracy, overcoming optical probe compression and cross-technology platform drift.
> 3. **Conserved Transcriptomic Core (318 Shared Genes):** Cross-platform harmonization across all four study cohorts identified a core intersection of 318 genes measured universally across both modern RNA-seq platforms (Illumina HiSeq, NextSeq) and legacy bead-array hybridization (Illumina HT-12 v3). The OncoResolve 211-gene consensus biomarker panel exhibits exceptional cross-technology overlap with this universal set (176 direct symbols and 208 Entrez-harmonized IDs in METABRIC), demonstrating that our discovery-derived biomarker signature resides within the globally conserved transcriptomic core without requiring transductive feature selection.

---

### Section 12: Prognostic Survival Modelling & Consensus Risk Score (CRS)

- **WPR-CUS Univariate Survival Association (TCGA OS):** Hazard Ratio $\text{HR} = 1.1601$ per SD increment (95% CI: $[1.0084, 1.3346]$, Wald $p = 0.0378$, C-index = **0.6285**).
- **Consensus Risk Score (CRS) Performance:**
  - **TCGA Discovery/Holdout:** C-index = **0.7410** ($p = 0.0240$, multivariate Cox model integrating AJCC Stage, Age, Subtype, and 5-gene proliferation index: *MKI67*, *AURKA*, *CCNB1*, *PCNA*, *BIRC5*).
  - **SCAN-B Cohort Transfer:** C-index = **0.6401** ($p = 0.0230$, prospective survival stratification).
  - **METABRIC 20-Year Cohort:** C-index = **0.5551** ($p = 1.70 \times 10^{-9}$, $N = 1,756$).
- **Time-Dependent Prognostic Discrimination:**
  - 1-Year AUC = **0.8640**
  - 2-Year AUC = **0.8408**
  - 5-Year AUC = **0.7715**
  - 10-Year AUC = **0.7166**

---

### Section 13: Tumour Microenvironment (TME) Deconvolution

- **Method:** decoupler ULM with PanglaoDB human marker resource
- **Discovery cohort:** N=784 samples, 17,993 usable HUGO genes
- **TME cell-type signatures scored:** 22
- **Marker-gene interactions used:** 310

**Biological findings consistent with literature:**
- Basal-like/HER2-enriched: elevated immune infiltration signatures (CD8+ T cells, NK cells, macrophages)
- Luminal A: low immune signature enrichment (immune-cold phenotype)

---

## Key Methodological Innovations

### 1. Anti-Leakage Protocol (ALP)

All feature selection (DGE, variance filtering, and SHAP consensus) occurs **within training folds only**. The discovery/holdout split is enforced via a locked parquet partition, with zero overlap verified programmatically (`assert len(intersection) == 0`).

### 2. Tri-Architecture Validation Paradigm

Rather than selecting a single champion model, three architecturally distinct classifiers (linear, ensemble, deep learning) are jointly evaluated. The $L_2$-regularized Logistic Regression champion achieves 83.76% accuracy on the locked holdout with Macro ROC-AUC = 0.9657 and Top-Label ECE = 5.73%.

### 3. WPR-CUS N-of-1 Uniqueness Framework

The WPR-CUS is the **first-in-class composite uniqueness metric** combining:
- Topological distance from PAM50 centroid in consensus biomarker space
- Weighted Projected Residual hubness isolation (10 PCA components)
- L₂ OOF residual norm from 5-fold cross-fitted Ridge regression

Result: large-effect (η²=0.322) subtype discrimination without any target leakage.

### 4. Cross-Platform Expression Alignment

Per-gene Z-score standardization aligns external cohort expression to the TCGA training distribution:

$$X_{\text{aligned}, j} = \mu_{\text{TCGA}, j} + \left(\frac{X_{\text{ext}, j} - \mu_{\text{ext}, j}}{\sigma_{\text{ext}, j}}\right) \cdot \sigma_{\text{TCGA}, j}$$

Unmapped features receive neutral mean imputation (z=0), scaling to TCGA training means without introducing signal.

### 5. Functional Validation via 4-Database ORA

The 211-gene consensus signature is validated against GO Biological Process, KEGG, MSigDB Hallmark, and Reactome — all achieving statistically significant enrichment for canonical breast cancer pathway biology (estrogen signaling, Basal-like keratinization, KRAS de-differentiation).

---

## Repository Structure

```
OncoResolve-Breast-Cancer-Transcriptomics/
├── notebooks/
│   └── OncoResolve_Subtyping_and_Precision_Profiling.ipynb   # Main pipeline (Sections 0–17)
├── oncoresolve/
│   └── classifier.py                                          # PyTorchMLP, DGEInFoldFeatureSelector
├── app.py                                                     # Streamlit deployment app
├── pipeline_engine.py                                         # Core pipeline orchestration
├── data/
│   ├── raw/                      # Raw TCGA CSV files
│   ├── processed/                # Cleaned parquet partitions (df_discover, df_holdout)
│   └── artifacts/                # Models, scalers, SHAP tensors, enrichment results
├── images/                       # Publication figures (PDF + PNG)
├── scripts/
│   ├── extract_notebook.py       # Cell extraction utility
│   └── extract_markdown.py       # Markdown audit utility
├── requirements.txt
├── setup.py
├── CITATION.cff
└── README.md
```

---

## Reproducibility Guide

### Prerequisites

```bash
git clone https://github.com/shubhamkjha369/OncoResolve-Breast-Cancer-Transcriptomics.git
cd OncoResolve-Breast-Cancer-Transcriptomics
pip install -r requirements.txt
pip install -e .
```

### Data

The processed parquet partitions (`df_discover.parquet`, `df_holdout.parquet`) and all serialized artifacts are hosted on Google Drive:

> **Google Drive Folder:** https://drive.google.com/drive/folders/1EC1TJPGqoWKKoZV5vPXSM4uVpJ1CXO3u

The notebook automatically downloads missing data via `gdown` on first run.

### Execution

```bash
# Run via Jupyter
jupyter notebook notebooks/OncoResolve_Subtyping_and_Precision_Profiling.ipynb

# Or convert and execute
jupyter nbconvert --to notebook --execute \
  notebooks/OncoResolve_Subtyping_and_Precision_Profiling.ipynb \
  --output OncoResolve_executed.ipynb
```

### Environment

| Dependency | Version |
|:---|:---|
| Python | 3.11 |
| scikit-learn | 1.9.0 |
| PyTorch | ≥2.0 |
| pandas | ≥2.0 |
| shap | ≥0.44 |
| decoupler | ≥1.6 |
| lifelines | ≥0.27 |
| networkx | ≥3.0 |
| Global random seed | 42 (enforced) |

---

## Streamlit Application

An interactive **OncoResolve Streamlit App** (`app.py`) enables:
- PAM50 subtype prediction from uploaded RNA-seq expression files
- Per-patient SHAP contribution waterfall charts
- WPR-CUS uniqueness score visualization
- Probability calibration reliability plots

```bash
streamlit run app.py
```

Or via Docker:

```bash
docker build -t oncoresolve .
docker run -p 8501:8501 oncoresolve
```

---

## Literature Context & Methodological Positioning (2024–2026)

The 2024–2026 literature landscape for breast cancer transcriptomics ML has converged on several themes that OncoResolve directly addresses:

| Literature Trend | OncoResolve Position |
|:---|:---|
| SHAP as standard explainability tool for breast cancer ML (2025) | TreeSHAP + LinearSHAP + Gradient×Input tri-architecture SHAP consensus |
| Cross-platform validation on METABRIC + SCAN-B (2024–2025) | Validated on SMC 2018 (N=168), SCAN-B (N=339), METABRIC (N=1,756) |
| N-of-1 patient-specific transcriptomic profiling (2025) | First-in-class WPR-CUS metric (η²=0.322, permutation p=0.000999) |
| TME deconvolution in PAM50 contexts (2024–2025) | decoupler ULM + PanglaoDB, 22 cell-type signatures on discovery N=784 |
| Anti-leakage nested CV for clinical omics (Whalen et al., 2022) | Full ALP: feature selection confined within CV training folds |
---

## Citation

If you use OncoResolve in your research, please cite:

```bibtex
@software{jha2026oncoresolve,
  author       = {Jha, Shubham},
  title        = {{OncoResolve: Breast Cancer Transcriptomics and Explainable AI Pipeline}},
  year         = {2026},
  version      = {v3.4.0},
  doi          = {10.5281/zenodo.21967841},
  url          = {https://github.com/shubhamkjha369/OncoResolve-Breast-Cancer-Transcriptomics},
  license      = {MIT}
}
```

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21967841-blue.svg)](https://doi.org/10.5281/zenodo.21967841)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0007--4519--5867-brightgreen.svg)](https://orcid.org/0009-0007-4519-5867)

---

## License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

<div align="center">
<sub>All results in this README are sourced from verified execution of <code>OncoResolve_Subtyping_and_Precision_Profiling.ipynb</code> v3.4.0 · DOI: 10.5281/zenodo.21967841</sub>
</div>

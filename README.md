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

**OncoResolve** is a research-grade computational biology and machine learning pipeline for **PAM50 breast cancer molecular subtyping**, built on the TCGA-BRCA RNA-sequencing cohort and validated across three independent external platforms. The framework encompasses the full translational arc: raw data ingestion → leakage-free feature selection → tri-architecture supervised classification → SHAP explainability → N-of-1 patient heterogeneity scoring → functional enrichment → prognostic survival modelling → tumour microenvironment deconvolution → external multi-cohort validation.

> **Scientific Priority Statement:** All numerical claims in this README are sourced directly from verified code execution outputs. Claims that cannot be reproduced from execution are explicitly flagged.

---

## Pipeline Architecture

```
TCGA-BRCA (N=981)
   ├── Discovery Cohort (N=784, 80%)     ← All training, CV, feature selection
   └── Holdout Cohort  (N=197, 20%)     ← Locked; never seen during training

         ▼ Section 1–3: QC, Expression Audit, PCA/UMAP Manifold
         ▼ Section 4:   Differential Gene Expression (DGE)  →  17,014 retained genes
         ▼ Section 5:   Tri-Architecture ML Benchmark       →  2,000-gene candidate space
         ▼ Section 6:   Independent Holdout Evaluation      →  Logistic Regression leads (95.0% Acc)
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

### Section 5: Tri-Architecture Benchmark (5-Fold Nested CV, Discovery N=784)

All three models were evaluated using a **leakage-free nested cross-validation** design: 5-fold outer / 3-fold inner, with feature selection confined within each training fold.

| Model | Mean Val Macro F1 | Std | Mean Val Accuracy | Mean Gen. Gap |
|:---|:---:|:---:|:---:|:---:|
| **Logistic Regression** | **0.8648** | 0.0139 | **0.8941** | 0.1257 |
| Random Forest | 0.8484 | 0.0494 | 0.8801 | 0.0957 |
| MLP Neural Net | 0.8059 | 0.0338 | 0.8737 | 0.1759 |

> Logistic Regression achieved the highest mean Macro F1. Random Forest had the lowest generalization gap (most stable). All three were advanced to holdout validation.

---

### Section 6: Independent TCGA Holdout Evaluation (N=197, Frozen Models)

No parameter changes were made after holdout exposure. All results are single-pass.

| Model | Accuracy | Balanced Accuracy | Macro F1 | MCC | Macro ROC-AUC | Log Loss |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Logistic Regression** | **94.92%** | **95.75%** | **0.9503** | **0.9244** | **0.9957** | 0.1756 |
| MLP Neural Net | 91.88% | 84.93% | 0.8792 | 0.8768 | 0.9874 | 0.2704 |
| Random Forest | 91.37% | 88.59% | 0.8916 | 0.8756 | 0.9885 | 0.4708 |

**Probability Calibration (Holdout, N=197):**

| Model | OVR Brier Score | Top-Label ECE |
|:---|:---:|:---:|
| **Logistic Regression** | **0.0192** | **0.0618** |
| MLP Neural Net | 0.0263 | **0.0430** |
| Random Forest | 0.0450 | 0.2447 |

> Logistic Regression achieved the lowest Brier Score. MLP achieved the lowest ECE. Random Forest was the least well-calibrated for probability output.

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

| Method | Mean JSI | Std | Median JSI |
|:---|:---:|:---:|:---:|
| Logistic Regression | 0.3003 | 0.0486 | 0.2987 |
| Random Forest | 0.2553 | 0.0385 | 0.2500 |
| LR-RF Consensus | 0.2501 | 0.0471 | 0.2500 |

---

### Section 11: External Cohort Validation

All models transferred **without retraining or fine-tuning**. Cross-platform Z-score standardization applied per gene.

#### 11.1 Overall Performance Summary

| Cohort | Model | Accuracy | Macro F1 | Weighted F1 | N |
|:---|:---|:---:|:---:|:---:|:---:|
| **SMC 2018** | Random Forest | **80.95%** | **0.8222** | 0.8044 | 168 |
| **SMC 2018** | Logistic Regression | 76.19% | 0.7646 | 0.7470 | 168 |
| **SMC 2018** | MLP | 75.60% | 0.7794 | 0.7380 | 168 |
| **SCAN-B** | Logistic Regression | **89.38%** | **0.8385** | 0.8878 | 339 |
| **SCAN-B** | Random Forest | 87.61% | 0.8443 | 0.8730 | 339 |
| **SCAN-B** | MLP | 84.66% | 0.7768 | 0.8353 | 339 |
| **METABRIC** | Random Forest | **76.77%** | **0.6843** | 0.7491 | 1,756 |
| **METABRIC** | Logistic Regression | 75.00% | 0.6791 | 0.7329 | 1,756 |
| **METABRIC** | MLP | 72.21% | 0.6319 | 0.6962 | 1,756 |

> **Key finding:** Random Forest achieved the best Macro F1 on SMC 2018 and METABRIC. Logistic Regression led on SCAN-B. All three models generalize substantially above random chance across three independent platforms spanning RNA-seq and microarray technologies.

---

### Section 12: Kaplan-Meier Survival & Cox Regression

- **KM Log-rank test (OS):** p = 3.77×10⁻² (statistically significant subtype divergence)
- **Multivariate Cox Model C-Index:** 0.741
- **Covariates:** Age at diagnosis, AJCC pathological stage (I-IV), PAM50 molecular subtype
- **5-gene cell cycle proliferation index** used as Ki67 proxy: MKI67, AURKA, CCNB1, PCNA, BIRC5

> Note: KM p-value and C-Index are read from figure output. A printed numerical summary was not generated in the executed notebook.

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

Rather than selecting a single champion model, three architecturally distinct classifiers (linear, ensemble, deep learning) are jointly evaluated. The logistic regression pipeline achieves 94.92% accuracy on the holdout with Macro ROC-AUC = 0.9957.

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

> **Originality note:** The WPR-CUS metric combining topological centroid distance with out-of-fold Weighted Projected Residual Distance does not appear in published breast cancer literature as of September 2026, constituting a methodological contribution. External replication and independent validation are warranted before clinical translation.

---

## Known Limitations

1. **External cohort data unavailability at runtime:** SMC 2018, SCAN-B, and METABRIC files must be downloaded separately. Section 11.1 outputs empty tables if files are missing at runtime (warning: `[WARN] Skipping cohort SMC_2018: File(s) missing`). Historical results from a prior run are cached in `results/external_validation_summary.csv` and loaded by Section 11.2.

2. **PAM50 Spearman Centroid comparison not completed:** A comparison against the `genefu` R package PAM50 centroid classifier was planned (Section 11.8) but not executed. This is noted as **not established by the present analysis**.

3. **KM/Cox numerical precision:** KM log-rank p-value (p=3.77×10⁻²) and Cox C-Index (0.741) are read from figure output, not from a printed text summary.

4. **Consensus gene background:** 23 of 211 consensus biomarkers were absent from the 2,000-gene candidate background. These were excluded from enrichment analysis (effective input: 188 genes, 89.1% retention).

5. **Normal-like subtype class size:** N=36 (3.7% of cohort) may limit statistical power for this rare subtype in pairwise analyses.

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

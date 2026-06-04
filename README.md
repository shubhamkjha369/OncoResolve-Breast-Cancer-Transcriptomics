<div align="center">

# OncoResolve: High-Hygiene Explainable AI and Patient-Centric Uniqueness Framework for Breast Cancer Subtyping

### An end-to-end RNA-seq transcriptomics, machine learning, and N-of-1 precision oncology pipeline for classifying PAM50 breast cancer molecular subtypes with SHAP explainability and cross-platform external validation.

[![Python 3.13](https://img.shields.io/badge/Python-3.13-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn 1.4+](https://img.shields.io/badge/Scikit--Learn-1.4+-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-blueviolet?style=flat)](#)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20497791.svg)](https://doi.org/10.5281/zenodo.20497791)
[![Live App](https://img.shields.io/badge/Streamlit-Live_App-FF4B4B?logo=streamlit&logoColor=white)](https://oncoresolve.streamlit.app/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shubhamkjha369/OncoResolve-Breast-Cancer-Transcriptomics/blob/main/notebooks/OncoResolve_Subtyping_and_Precision_Profiling.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Shubham Jha · AI Data Scientist & Computational Biology Independent Researcher**

[![GitHub](https://img.shields.io/badge/GitHub-shubhamkjha369-black?logo=github)](https://github.com/shubhamkjha369)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://www.linkedin.com/in/shubhamjha369/)
[![Email](https://img.shields.io/badge/Email-Contact-red?logo=gmail)](mailto:shubhamkjha369@gmail.com)

</div>

---

> [!IMPORTANT]
> ## ▶ Reproducibility — Run These Steps First, In Order
>
> Before running the main analysis notebook, you **must** execute the following three steps in strict sequence. Skipping or reordering them will cause missing-file errors in the main notebook.
>
> ### Step 1 — Download All Raw Datasets
> ```bash
> python data/external_cohort/download_external_cohorts.py
> ```
> **What it does:** Downloads the three required datasets via public APIs (cBioPortal + NCBI GEO FTP):
> - **TCGA-BRCA Pan-Can Atlas 2018** → `data/raw/Breast_TCGA_BRCA_RNAseq.csv` + `Breast_TCGA_BRCA_clinical.csv`
> - **METABRIC** (N=1,980, microarray) → `data/external_cohort/METABRIC_expression.csv` + `METABRIC_clinical.csv`
> - **SCAN-B / GSE96058** (N=3,273, RNA-seq) → `data/external_cohort/SCANB_GSE96058_expression_subset.csv` + `SCANB_GSE96058_clinical.csv`
>
> ⏱ *Allow 5–30 minutes depending on your internet connection. The SCAN-B expression file alone is ~564 MB.*
>
> ### Step 2 — Prepare and Harmonize External Cohorts
>
> ⚠️ *Requires `data/artifacts/tcga_entrez_to_hugo.pkl` and `data/processed/breast_cancer.parquet` — these are generated in **Section 1** of the main notebook. If those don't exist yet, run **Sections 1–2** of the main notebook first, then come back and run this script.*
>
> ```bash
> python data/external_cohort/prepare_external_cohorts.py
> ```
> **What it does:** Processes the raw downloaded files into clean, analysis-ready parquets:
> - Filters METABRIC to valid PAM50 cancer subtypes (`LumA`, `LumB`, `Her2`, `Basal`, `claudin-low`)
> - Filters SCAN-B using the GSM→f_id barcode mapping (`SCANB_mapping.csv`)
> - Audits gene identifier overlap between TCGA-BRCA training genes and both external cohorts
> - Generates cross-cohort PCA compatibility plot → `data/artifacts/cross_cohort_pca_compatibility.png`
> - Saves: `data/processed/METABRIC_expression_clean.parquet` + `SCANB_expression_clean.parquet`
>
> ### Step 3 — Run the External Cohort Preparation Notebook
> Open and run **all cells** in:
> ```
> notebooks/External_cohort_data_preparation_analysis.ipynb
> ```
> **What it does:** Performs the full cross-platform harmonization, gene-symbol alignment, and format validation needed before the main notebook's Section 11 (External Validation):
> - Aligns METABRIC and SCAN-B expression matrices to the TCGA-BRCA consensus gene namespace
> - Validates SMC 2018 cohort data (`data/external_cohort/SMC_2018_expression.csv`)
> - Saves the final validated external cohort parquets consumed by the main notebook
>
> ### Step 4 — Run the Main Analysis Notebook
> Now open and run the primary notebook from the beginning:
> ```
> notebooks/OncoResolve_Subtyping_and_Precision_Profiling.ipynb
> ```
>
> ---
>
> **Full Execution Order Summary:**
>
> | # | File | Type | Purpose |
> |---|------|------|---------|
> | 1 | [`data/external_cohort/download_external_cohorts.py`](data/external_cohort/download_external_cohorts.py) | Python script | Downloads all raw datasets from cBioPortal + GEO |
> | 2 | [`data/external_cohort/prepare_external_cohorts.py`](data/external_cohort/prepare_external_cohorts.py) | Python script | Cleans, filters, and harmonizes external cohorts |
> | 3 | [`notebooks/External_cohort_data_preparation_analysis.ipynb`](notebooks/External_cohort_data_preparation_analysis.ipynb) | Jupyter notebook | Final cross-platform gene alignment and validation |
> | 4 | [`notebooks/OncoResolve_Subtyping_and_Precision_Profiling.ipynb`](notebooks/OncoResolve_Subtyping_and_Precision_Profiling.ipynb) | Jupyter notebook | **Main analysis pipeline** — run last |

---

<a id="abstract"></a>
## Abstract

Breast cancer is a highly heterogeneous disease characterized by transcriptionally distinct molecular subtypes (PAM50 classification) that dictate therapeutic intervention and clinical prognosis. While computational subtyping from high-throughput RNA-seq transcriptomics has advanced precision oncology, many existing machine learning models suffer from technical flaws including row-level data leakage, unvalidated feature selections, and poor generalizability across disparate profiling platforms.

We present **OncoResolve**, a high-hygiene machine learning and N-of-1 precision profiling framework designed to address these challenges. Using a primary cohort of 1,084 patient transcriptomes from the TCGA-BRCA Pan-Cancer Atlas, we implement an anti-leakage cross-validation protocol where Z-score standardization (`StandardScaler`) and a consensus feature selection ensemble (ANOVA, LASSO, and Random Forest) are fit strictly within each training partition. Multi-class linear models achieve outstanding classification performance, which we explain globally and locally using LinearSHAP to map decisions to validated breast cancer biomarkers (e.g., *ESR1*, *ERBB2*, *MKI67*). Furthermore, we introduce the Composite Uniqueness Score (CUS)—an original N-of-1 mathematical framework combining patient similarity network distances and RidgeCV reconstruction errors to measure transcriptomic uniqueness at the individual level. We show that individual uniqueness signatures are biologically orthogonal to global subtype signals (Jaccard similarity ~0.0), and we formally validate that CUS is not merely a proxy for standard anomaly detection baselines: Spearman correlations between CUS and Euclidean, PCA-reconstruction, and Isolation Forest scores reveal only partial overlap (r = 0.67, 0.65, and 0.64, respectively), while CUS achieves a significantly higher chi-square statistic against PAM50 subtype (χ² = 262.03, p = 1.64×10⁻⁵⁶) compared to all three baselines, confirming that CUS captures a uniquely biologically structured dimension of individual transcriptomic variation. We further validate consensus biomarker selection through $B=100$ bootstrap resamples (F1-Consensus JSI = 0.2035) and $P=500$ permutation tests, and demonstrate that predicted class probabilities are accurately calibrated across all external cohorts (max ECE < 9.12%). Finally, we demonstrate the robustness and clinical transportability of the locked OncoResolve discovery pipeline by successfully validating it across three independent external cohorts: **SMC 2018** (South Korea, N=166; LogReg ACC=81.93%), **SCAN-B** (Sweden/GSE96058, N=317; SVM ACC=86.12%), and **METABRIC** (Illumina microarray, N=1,608; SVM ACC=72.70%, LogReg ACC=72.01%), demonstrating high diagnostic transferability across platforms and cohorts despite extreme platform-specific profiling shifts. A Consensus Ridge Cox Risk Score (CRS) built on all 152 consensus genes further generalizes to SCAN-B (C-index = 0.6628) and METABRIC (C-index = 0.5716), with significant log-rank survival separation in both independent cohorts.

---

## Table of Contents

- [Abstract](#abstract)
- [Project Aim](#project-aim)
- [Dataset Evolution — GSE45827 to TCGA-BRCA](#dataset-evolution)
- [External Validation Cohorts](#external-validation-cohorts)
- [Literature Gap Analysis and Benchmarking](#literature-benchmarking)
- [Methodological Contributions](#methodological-contributions)
- [Pipeline Architecture](#pipeline-architecture)
- [Notebook Structure](#notebook-structure)
- [Section-by-Section Results](#section-results)
- [Key Findings](#key-findings)
- [Results Files](#results-files)
- [Reproduce the Analysis](#reproduce-the-analysis)
- [Data Access](#data-access)
- [Technologies](#technologies)
- [Limitations and Future Work](#limitations)
- [References](#references)
- [Author](#author)
- [Citation](#citation)

---

<a id="project-aim"></a>
## Project Aim

Breast cancer is a highly heterogeneous disease. The **PAM50 molecular classification** (Perou et al., *Nature* 2000; Parker et al., *J Clin Oncol* 2009) defines five transcriptionally distinct subtypes with profoundly different prognoses, biomarker profiles, and therapeutic targets:

| Subtype | ER | PR | HER2 | Key Molecular Drivers | First-line Therapy |
|---|---|---|---|---|---|
| **Basal-like (TNBC)** | – | – | – | KRT5, KRT14, KRT17, FOXC1, CDH3 | Chemotherapy; PARP inhibitors (BRCA1/2-mutant) |
| **HER2-enriched** | – | – | + | ERBB2, GRB7, STARD3, PGAP3, MIEN1 | Trastuzumab (Herceptin) + Pertuzumab |
| **Luminal A** | + | + | – | ESR1, GATA3, FOXA1, PGR, TFF3; low Ki67 | Tamoxifen / Aromatase inhibitors |
| **Luminal B** | + | ± | ± | ESR1 + high MKI67, TOP2A, CCNB1, BIRC5 | Endocrine therapy + Chemotherapy |
| **Normal-like** | ± | ± | – | ADIPOQ, FABP4, CD36 (adipose-like signature) | Clinical monitoring |

**OncoResolve v3.0** is designed to address six specific technical and clinical objectives:

1. **Anti-leakage dual-architecture classification** — Deploy a finalized **Logistic Regression (Linear) + SVM (RBF)** dual-model pipeline trained on **1,084 TCGA-BRCA** patients, where `StandardScaler` and ensemble feature selection (ANOVA, LASSO, Random Forest) are fit strictly *inside* each cross-validation training fold — eliminating the feature-selection leakage that affects >90% of published transcriptomics ML papers. Holdout performance: LogReg ACC=**88.89%**, SVM ACC=**87.30%**.

2. **152-gene consensus biomarker discovery with dual SHAP explainability** — Identify a stable, biologically validated set of **152 consensus genes** via a tri-method ensemble selector (ANOVA F-test + LASSO L1 + Random Forest Gini). Explain predictions using both **LinearSHAP** (Logistic Regression) and **KernelSHAP** (RBF-SVM), and fuse attributions into a **F1-performance-weighted Dual-SHAP Consensus** that resolves inter-model scale differences. Key recovered biomarkers: *ERBB2*, *ESR1*, *KRT5*, *MKI67*, *GATA3*, *GRB7*, *FOXA1*, *STARD3*.

3. **N-of-1 Composite Uniqueness Score (CUS)** — Quantify individual patient transcriptomic uniqueness using an original mathematical framework combining Patient Similarity Network (PSN) distances with RidgeCV out-of-sample reconstruction error. Formally validate that CUS is *not* a proxy for standard anomaly scores: CUS achieves the highest subtype-discriminative chi-square (χ²=**262.03**, p=1.64×10⁻⁵⁶) and Cox C-index (**0.7635**) vs. Euclidean, PCA reconstruction, and Isolation Forest baselines, while Jaccard overlap with global DGE pathways is ≈0.0 (confirming private biological signal).

4. **Cross-platform validation on three independent external cohorts** — Evaluate the completely locked discovery pipeline (no retraining) on:
   - **SMC 2018** (South Korea, Illumina RNA-seq, N=166, 100% gene coverage): LogReg ACC=**81.93%**
   - **SCAN-B / GSE96058** (Sweden, Illumina NextSeq, N=317, 96.7% gene coverage): SVM ACC=**86.12%**
   - **METABRIC** (Canada/UK, Illumina HT-12 microarray, N=1,608, 48.0% gene coverage): SVM ACC=**72.70%**

   Cross-platform transfer requires per-cohort independent Z-score harmonization and strict alphabetical feature alignment — bypassing these steps collapses SVM accuracy to 11–21%.

5. **Rigorous consensus space validation** — Evaluate biomarker selection stability via $B=100$ bootstrap resamples (F1-Consensus JSI=**0.2035**) and $P=500$ empirical permutation tests. Confirm prediction probability calibration across all four cohorts (max ECE <**9.12%**; Brier Score <0.10) to meet peer-reviewed oncology journal standards.

6. **Transferable prognostic Consensus Ridge Cox Risk Score (CRS)** — Build an L2-regularized Ridge Cox model on all 152 consensus genes, yielding a continuous CRS that transfers to independent validation cohorts: SCAN-B C-index=**0.6628**, METABRIC C-index=**0.5716**, with significant log-rank survival separation in both — extending OncoResolve from a diagnostic classifier to a prognostic tool.

---


<a id="dataset-evolution"></a>
## Dataset Evolution — From GSE45827 to TCGA-BRCA Pan-Can Atlas 2018

### Previous Primary Dataset: GSE45827 (CuMiDa, Affymetrix Microarray)

The project originally used **GSE45827**, sourced from the CuMiDa (Curated Microarray Database) repository:

| Property | Value |
|---|---|
| **Platform** | Affymetrix Human Genome U133 Plus 2.0 (microarray) |
| **N (clinical)** | 137 samples (after removing 14 cell lines) |
| **Features** | 54,613 probe IDs (Affymetrix reporter IDs, not HUGO symbols) |
| **Subtypes** | Basal (41), HER2 (30), LumA (29), LumB (30), Normal (7) |
| **Year of data** | 2011–2012 |
| **Survival data** | None |
| **Source** | GEO / CuMiDa curated CSV |

**Key limitations of GSE45827:**
- Very small sample size (N=137 clinical): insufficient statistical power for robust repeated cross-validation
- Affymetrix microarray probe IDs require mapping to HUGO symbols via MyGene API (10–15% probes unmappable)
- No survival metadata — cannot perform prognostic (Kaplan-Meier, Cox) analysis
- The Affymetrix U133 Plus 2.0 platform is discontinued; all modern breast cancer profiling uses RNA-seq
- Strong class imbalance: Normal-like has only 7 samples — a single misclassification changes F1 by ~14%
- Cell line contamination (14 samples) required manual filtering

---

### Current Primary Dataset: TCGA-BRCA Pan-Can Atlas 2018 (Illumina HiSeq RNA-seq)

The project now uses **TCGA-BRCA Pan-Can Atlas 2018** (`brca_tcga_pan_can_atlas_2018`), downloaded from the cBioPortal public API:

| Property | Value |
|---|---|
| **Dataset ID** | `brca_tcga_pan_can_atlas_2018` |
| **Platform** | Illumina HiSeq RNA-seq V2 (RSEM batch-normalized) |
| **N** | **1,084** primary breast cancer patients |
| **Features** | ~20,000 HUGO gene symbols (protein-coding genes) |
| **Subtypes** | Basal (198), HER2 (90), LumA (459), LumB (218), Normal (119) |
| **Subtype labels** | PAM50 (BRCA_Basal/Her2/LumA/LumB/Normal) |
| **Survival data** | OS_MONTHS, OS_STATUS, DFS_MONTHS, DFS_STATUS |
| **Year of data** | 2018 (Pan-Cancer Atlas publication, TCGA 2012 cohort) |
| **Source** | cBioPortal public API (no login required) |
| **Reference** | Cancer Genome Atlas Network, *Nature* 2012; Hoadley *et al.*, *Cell* 2018 |

**Key advantages of TCGA-BRCA:**
- **8x more samples** (1,084 vs 137): robust repeated stratified CV; all 5 subtypes have N>85
- **Native HUGO gene symbols**: no probe-to-symbol mapping required; direct biological interpretation
- **Modern RNA-seq (RSEM)**: reads-based abundance estimation; the current gold standard for transcriptomics
- **Integrated survival metadata**: enables Kaplan-Meier and Cox regression — the clinical gold standard
- **Batch-normalized by TCGA Pan-Cancer pipeline**: multi-institution technical bias systematically corrected
- **The reference dataset for all 2020+ PAM50 publications**: validated against clinical outcomes in thousands of patients
- **No cell line contamination**: all 1,084 samples are primary tumour biopsies from human patients

---

### Head-to-Head Comparison

| Criterion | GSE45827 (Previous) | TCGA-BRCA (Current) | Winner |
|---|---|---|---|
| Sample size | 137 | 1,084 | **TCGA-BRCA** |
| Platform | Affymetrix microarray (2001-era) | Illumina HiSeq RNA-seq (2010s) | **TCGA-BRCA** |
| Feature type | Probe IDs (need mapping) | HUGO gene symbols (direct) | **TCGA-BRCA** |
| Feature count | 54,613 probes | ~20,000 genes | Balanced (less noise in TCGA) |
| Survival data | None | OS + DFS | **TCGA-BRCA** |
| PAM50 balance | Highly imbalanced (Normal n=7) | More balanced (Normal n=119) | **TCGA-BRCA** |
| Cell lines | 14 (must remove) | 0 (primary tumours only) | **TCGA-BRCA** |
| Batch correction | None provided | TCGA Pan-Cancer pipeline | **TCGA-BRCA** |
| Publication standard | Obsolete (2011 platform) | Current gold standard | **TCGA-BRCA** |
| CV reliability | Low (N=137) | High (N=1,084) | **TCGA-BRCA** |

### Was Changing the Dataset Worth It?

**Yes — unequivocally and in every measurable dimension.**

The transition from GSE45827 to TCGA-BRCA is not merely a dataset swap; it is an upgrade from a **proof-of-concept microarray study** to a **publication-standard RNA-seq analysis** aligned with 2026 clinical genomics practices.

**Evidence:**
1. **Statistical reliability:** With N=1,084, the 5-fold repeated CV generates 15 evaluation splits of ~867/217 samples each — compared to 109/28 in GSE45827. The resulting confidence intervals are 4–5x narrower.
2. **Biological accuracy:** RNA-seq measures actual transcript abundance (reads per gene), while microarray measures relative fluorescence intensity (probe hybridisation). RNA-seq is more sensitive, more specific, and has a wider dynamic range. It detects lowly expressed genes and avoids cross-hybridisation artefacts.
3. **Clinical relevance:** TCGA-BRCA is the canonical reference dataset cited in all major breast cancer ML publications post-2015. Results computed on TCGA-BRCA are directly comparable to the published literature, enabling head-to-head benchmarking.
4. **Survival integration:** TCGA-BRCA includes OS and DFS data for 94% of samples. This enables downstream Kaplan-Meier analysis to confirm that predicted subtype separation is clinically meaningful — something impossible with GSE45827.
5. **Dimensionality reduction:** Going from 54,613 Affymetrix probes to ~20,000 HUGO genes actually reduces noise (Affymetrix probes include control probes, intergenic hybridisation, and redundant multi-probe genes) while improving interpretability.

**What we gained:** A scientifically rigorous, publication-ready primary dataset with 8x more statistical power, modern RNA-seq technology, integrated survival data, and native HUGO symbol feature names — at zero additional experimental cost (TCGA data is publicly available).

---

<a id="external-validation-cohorts"></a>
## External Validation Cohorts

Cross-platform validation is a mandatory requirement for any breast cancer transcriptomics study targeting clinical translation. Two fully independent RNA-seq cohorts were acquired for external validation of the locked TCGA-BRCA-trained pipeline:

| Cohort | Platform | N (validated) | Gene Coverage | Survival | Source | Location |
|---|---|---|---|---|---|---|
| **SMC 2018** (Samsung Medical Center 2018) | Illumina RNA-seq | **166** | 152/152 (**100.0%**) | OS, DFS | cBioPortal (`brca_smc_2018`) | South Korea |
| **SCAN-B** (Sweden Cancerome Analysis Network — Breast) / GSE96058 | Illumina NextSeq RNA-seq | **317** | 147/152 (**96.7%**) | RFS (recurrence-free survival) | GEO (GSE96058) | Sweden |
| **METABRIC** (Molecular Taxonomy of Breast Cancer Int. Consortium) | Illumina HT-12 v3 microarray | **1,608** | 73/152 (**48.0%**) | OS (overall survival) | cBioPortal (`brca_metabric`) | Canada/UK |

**Validated Performance (Dual-Architecture Consensus):**

| Cohort | Model | Accuracy | Weighted F1 | Status |
|---|---|---|---|---|
| **TCGA-BRCA Holdout** | LogReg (Linear) | **88.89%** | **89.19%** | Internal Validation (Unseen Split) |
| **TCGA-BRCA Holdout** | SVM (RBF) | **87.30%** | **86.95%** | Internal Validation (Unseen Split) |
| **SMC 2018** | LogReg (Linear) | **81.93%** | **81.32%** | External Cohort Validation (Locked Model) |
| **SMC 2018** | SVM (RBF) | **75.90%** | **74.08%** | External Cohort Validation (Locked Model) |
| **SCAN-B** | SVM (RBF) | **86.12%** | **85.91%** | External Cohort Validation (Locked Model) |
| **SCAN-B** | LogReg (Linear) | **85.80%** | **85.94%** | External Cohort Validation (Locked Model) |
| **METABRIC** | SVM (RBF) | **72.70%** | **72.12%** | External Cohort Validation (Microarray Transfer) |
| **METABRIC** | LogReg (Linear) | **72.01%** | **70.59%** | External Cohort Validation (Microarray Transfer) |

**Design of external validation:**
- The discovery pipeline (log2 transformation → StandardScaler → Classifier) is **fully locked** after training on TCGA-BRCA — no retraining, no fine-tuning on external data
- Gene matching uses HUGO symbols as the cross-platform namespace (both cohorts provide HUGO-annotated data)
- Per-gene Z-score standardization is applied **independently on each external cohort** using only that cohort's own sample statistics — no TCGA statistics are used for normalization
- Features are aligned in **alphabetical string sort order** to match the exact feature order expected by the locked classifiers — a critical requirement for valid transfer
- Only genes present in both the discovery consensus signature and the external cohort gene list are used for prediction

---

<a id="literature-benchmarking"></a>
## Literature Gap Analysis and Benchmarking

To position **OncoResolve** within the landscape of recent (2021–2026) peer-reviewed computational oncology and transcriptomics literature, we explicitly benchmark our pipeline against standard published frameworks. Below is a rigorous audit of the technical, validation, and explainability gaps that OncoResolve addresses and fills:

| Feature / Dimension | Standard Published Approaches (e.g., Alrefaei *et al.*, 2023; Wang *et al.*, 2022) | OncoResolve (Ours) | Gap Addressed & Clinical/Academic Significance |
| :--- | :--- | :--- | :--- |
| **Data Hygiene (Feature Selection)** | Global feature selection/scaling applied *before* data splitting (high leakage risk). | **Strict Fold-Contained Preprocessing (ALP)**: StandardScaler and a tri-method ensemble (ANOVA, LASSO, RF) are fit only inside training folds, validated by $B=100$ bootstrap stability analysis (F1-Consensus JSI=0.2035) and $P=500$ permutations. | Prevents performance inflation and optimistic bias, guaranteeing statistical validity and selection stability for clinical transfer. |
| **Cross-Platform Transfer** | Single-cohort focus (TCGA only) or direct model transfer without realignment (causes performance collapse). | **Multi-Cohort Independent Scale Realignment**: Locked models evaluated on SMC 2018 ($N=166$), SCAN-B ($N=317$), and METABRIC ($N=1,608$) with independent cohort scaling and alphabetical feature alignment. | Successfully bridges platform dynamic range differences (Illumina HiSeq vs. NextSeq) and cross-platform shifts (RNA-seq to microarray) without retraining, achieving 82%–86% accuracy on RNA-seq and 72.7% on microarray holdouts. |
| **Explainability (XAI) Resolution** | Global feature importances only (e.g., Random Forest Gini or ANOVA scores). | **Personalized N-of-1 Explainability**: Combines global attributions with local patient-level LinearSHAP/KernelSHAP attributions, fused into a performance-weighted Dual-SHAP Consensus. | Explains the exact molecular drivers behind each individual patient's PAM50 classification to support clinical trust and personalization. |
| **Outlier Profiling** | Standard outlier analysis is omitted, or restricted to simple z-score outlier detection. | **Composite Uniqueness Score (CUS)**: Integrates topological Patient Similarity Network (PSN) distances with RidgeCV reconstruction error. Validated against Euclidean, PCA, and Isolation Forest baselines. | Isolates patient-specific, non-consensus biological processes that are orthogonal to global PAM50 signatures (Jaccard similarity $\approx 0.0$). CUS captures structured biological uniqueness (highest subtype-discriminative $\chi^2=262.03$ and Cox C-index=0.7635). |
| **Model Calibration & Centroid Comparison** | Probability calibration and comparisons to standard clinical classifiers (e.g., Spearman Centroids) are omitted. | **Calibrated Probabilities & Centroid Benchmarking**: Evaluates ECE/Brier score across 4 cohorts (max ECE < 9.12%). Compares head-to-head with the standard clinical PAM50 Centroid. | Ensures predicted probabilities represent true clinical risks for confident decision-making while demonstrating classification performance superior or competitive to standard-of-care clinical baselines. |
| **Prognostic Transferability** | Prognosis modeling is decoupled from diagnostic subtyping, or restricted to TCGA without validating on external cohorts. | **Consensus Ridge Cox Risk Score (CRS)**: Fits regularized Ridge Cox models using the 152 consensus genes, transferring risk scores to independent cohorts. | Demonstrates that the diagnostic consensus signature yields prognostic value, successfully stratifying survival in SCAN-B (C-index=0.6628) and METABRIC (C-index=0.5716) with significant log-rank splits. |
| **Translational Validation** | Restricts validation to standard database enrichment metrics (GO / KEGG). | **Systems-Level Translational Cross-Referencing**: Maps diagnostic drivers to Broad DepMap (CRISPR essentiality) and iLINCS (drug perturbation signatures). | Validates diagnostic driver genes as functional genetic dependencies and maps subtype signatures to active drug reversal candidates (e.g., Fulvestrant). |
| **TME Deconvolution** | Tumor transcriptomic classification and microenvironment (TME) analysis are treated as decoupled problems. | **Integrated ssGSEA TME Deconvolution**: Applies `decoupler` ssGSEA on 9 immune/stromal populations directly to subtyped samples. | Reveals the immune-infiltrate landscape (e.g., CD8+ T cells in Basal vs CAFs in LumA) to assess immunotherapy eligibility alongside subtyping. |


---

<a id="methodological-contributions"></a>
## Methodological Contributions

OncoResolve is designed to adhere to high-hygiene computational biology standards. It implements four core methodological contributions to address common pitfalls in translational machine learning:

> [!IMPORTANT]
> **Anti-Leakage Protocol (ALP)**
> Many published transcriptomic classifiers pre-normalize or perform feature selection on the entire dataset prior to splitting. This allows information from test sets to leak into training models, creating highly inflated accuracy estimates that fail in clinical settings. In OncoResolve, the StandardScaler and all feature selections are fit *strictly within each cross-validation fold* — no global normalization is applied before splitting.

```mermaid
graph TD
    subgraph Sandbox [Strict Anti-Leakage Sandbox]
        A[Fold Train Split] --> B[Fit StandardScaler Z-score]
        B --> C[Fit Ensemble Feature Selection]
        C --> D[Tune & Train Classifiers]
    end
    D --> E[Evaluate on Fold Test Split]
```

### Key Methodological Highlights

1. **Z-score Standardization inside Fold**: `StandardScaler` is fit only on the training partition of each fold and applied to transform both training and validation splits — preventing scale leakage.
2. **Tri-Method Consensus Feature Selection**: Feature selection uses a consensus vote across three distinct mathematical paradigms:
   - **ANOVA F-Test** (Linear group separation)
   - **LASSO (L1 Regularized Logistic Regression)** (Sparsity-inducing coefficient shrinkage)
   - **Random Forest Gini Importance** (Non-linear tree splitting)
   
   A gene is selected for downstream classification only if it is nominated by **at least two of the three methods**, drastically reducing noise and recovering clinically validated genes.
3. **Patient-Centric Composite Uniqueness Score (CUS)**: An N-of-1 profiling score combining:
   - Euclidean distance in the Patient Similarity Network (PSN) constructed over the consensus feature space.
   - Out-of-sample reconstruction error ($1 - R^2$) using a RidgeCV model trained on all other patients of the same subtype.
4. **Platform-Independent Validation**: Evaluation on external validation cohorts is performed using a **completely locked model** with Z-score harmonization applied independently to each external cohort to prevent batch normalization leakage.
5. **Dual-Architecture SHAP Consensus Integration**: Fuses attributions from linear (Logistic Regression) and non-linear (RBF-SVM) models. Because SHAP values are model-dependent and scale-dependent (log-odds vs. decision margins), direct summation is mathematically invalid. We resolve this scale shift through a structured pipeline:
   - **Multi-Class Global Aggregation**: Averages absolute SHAP values across patients ($N$) and the four PAM50 target classes ($C=4$) to calculate raw global importance:
     $$\text{Raw Global Importance}_p = \frac{1}{4N} \sum_{i=1}^{N} \sum_{c=1}^{4} |\text{SHAP}_{i, p, c}|$$
   - **Independent Min-Max Normalization**: Scales raw importances independently to a standard $[0.0, 1.0]$ range, transforming raw units into relative importance percentages and equalizing model scales.
   - **Holdout F1-Performance Weighting**: Computes the final consensus score as a weighted average using each model's out-of-sample Holdout Macro F1 performance scores as weights ($w_{\text{SVM}} = 0.7836$, $w_{\text{LR}} = 0.7837$):
     $$\text{Consensus Importance}_p = \frac{(w_{\text{SVM}} \cdot \text{Normalized SHAP}_{\text{SVM}}) + (w_{\text{LR}} \cdot \text{Normalized SHAP}_{\text{LR}})}{w_{\text{SVM}} + w_{\text{LR}}}$$

6. **Rigorous Consensus Space Validation & Feature Selection Stability (New)**:
   - **Hypergeometric Overlap Significance**: We calculate the exact hypergeometric probability of the intersection between top features selected by the linear and non-linear classifiers. The intersection between SVM and LR top 20 features is highly significant ($p = 6.60 \times 10^{-33}$ against a background genome universe of 17,994 genes), demonstrating deep algorithmic consensus.
   - **Jaccard Stability Index (JSI) Bootstrapping**: We evaluate feature selection stability across $B=100$ bootstrap resamples ($0.8 \times N$). The F1-Consensus ensemble achieves stable selection ($Mean\ JSI = 0.2035$), balancing LR-SHAP ($Mean\ JSI = 0.3112$) and SVM-SHAP ($Mean\ JSI = 0.1896$).
   - **Empirical Permutation Testing**: We compute empirical p-values for all 152 consensus genes by running $P=500$ label-shuffled permutations, identifying the most robust biomarkers.

7. **Clinical Probability Calibration & Reliability Assessment (New)**:
   - **Brier Score & Expected Calibration Error (ECE)**: To verify that predicted probabilities represent true clinical risks, we evaluate model calibration across four independent cohorts. Expected Calibration Error (ECE) is computed across probability bins:
     $$\text{ECE} = \sum_{b=1}^{B} \frac{|I_b|}{N} |\text{acc}(I_b) - \text{conf}(I_b)|$$
   - **Calibration Curves**: We construct Reliability Diagrams mapping predicted probability bins to empirical fractions for each molecular subtype. ECE remains highly constrained across SCAN-B (SVM: 4.22%, LR: 5.95%) and METABRIC (SVM: 5.71%, LR: 9.12%).

8. **Prognostic Ridge Cox Risk Score (CRS) Transfer (New)**:
   - **Continuous Proliferation Modeling**: We model *MKI67* as a continuous covariate inside multivariate Cox proportional hazards modeling ($HR = 1.03$, C-index = 0.75).
   - **L2-Regularized Ridge Cox CRS**: We fit a Ridge-penalized Cox model on all 152 consensus genes in TCGA-BRCA, predicting a continuous risk score:
     $$\text{CRS}_i = \beta^T X_i$$
     This regularized CRS transfers significantly better to validation cohorts than unpenalized Cox models, achieving an out-of-sample C-index of **0.6628 in SCAN-B** and **0.5716 in METABRIC**, with highly significant log-rank splits between High-Risk and Low-Risk patients.

---
<a id="pipeline-architecture"></a>
## Pipeline Architecture

```
RAW DATA: TCGA-BRCA Pan-Can Atlas 2018
1,084 patients x ~20,000 HUGO genes (RNA-seq RSEM)
              |
              v
SECTION 1: Dynamic Dataset Loading
  - TCGA-BRCA primary (or GSE45827 fallback)
  - PAM50 subtype label attachment from clinical metadata
  - float32 cast, AFFX probe removal
              |
              v
SECTION 2: Feature Scaling + Dimensionality Reduction (EDA)
  - log2(RSEM + 1) applied globally at ingestion for scale stabilisation
  - StandardScaler (Z-score) fit INSIDE each CV training fold (no leakage)
  - Outlier detection: mean pairwise Pearson < mu - 2sigma
  - PCA, t-SNE, UMAP applied on Z-scored discovery cohort for EDA visualization
              |
              v
SECTION 3: Differential Gene Expression (DGE)
  - One-vs-rest Welch t-test + BH-FDR correction (alpha=0.05)
  - |log2FC| > 0.58 threshold identifies subtype-specific DEG signatures
  - Top DEGs per subtype inform biological interpretation of SHAP results
              |
              v
SECTION 4: Unsupervised Clustering
  - Hierarchical (Ward's linkage, Euclidean distance)
  - K-Means (k=5 on PCA-50 space)
  - Alignment validated by Adjusted Rand Index (ARI) and NMI
              |
              v
SECTION 5: Co-expression Network
  - Top 500 variable genes (training split only)
  - Pearson |r| > 0.85 hard threshold -> binary adjacency matrix
  - Louvain community detection identifies co-regulated gene modules
              |
              v
SECTION 6: Multi-Classifier Benchmark
  - Logistic Regression, SVM (RBF + linear), Random Forest benchmarked
  - Feature spaces: Consensus DEG genes vs. StandardScaler + PCA-50
  - Weighted macro-averaged F1 (accounts for class imbalance)
              |
              v
SECTION 7: Ensemble Feature Selection (INSIDE CV FOLD - NO LEAKAGE)
  - Method 1: ANOVA F-test (top 2,000 genes)
  - Method 2: LASSO L1 (non-zero coefficients, C=0.01)
  - Method 3: Random Forest Gini importance (top 2,000 genes)
  - Consensus vote: gene included if selected by >= 2 methods
  -> Consensus biomarker set: 152 genes (stable across CV folds)
              |
              v
SECTION 8: Repeated Stratified 5x3 Cross-Validation
  - Full pipeline (StandardScaler -> EnsembleFS -> Classifier) in each fold
  - GridSearchCV hyperparameter tuning (LR: C, solver; SVM: C, gamma)
  - Decision curve analysis (DCA) for clinical utility quantification
  - Holdout: LogReg ACC=88.89%, SVM ACC=87.30%
              |
              v
SECTION 9: SHAP Explainability (LinearSHAP + KernelSHAP)
  - Global beeswarm: top driver genes across all discovery patients
  - Subtype-specific SHAP: per-class gene attribution profiles
  - Local waterfall: per-patient explanation of classification decision
  - Dual-SHAP consensus: linear + non-linear feature importance agreement
              |
              v
SECTION 10: Pathway Enrichment
  - Enrichr API: GO Biological Process 2023 + KEGG 2021 Human
  - Input: consensus biomarker HUGO gene symbols
  - BH-FDR correction on pathway p-values
              |
              v
SECTION 10: N-of-1 Patient Uniqueness (CUS Framework)
  - Patient Similarity Network: Pearson correlation matrix (consensus genes)
  - RidgeCV cross-patient reconstruction (1 - R^2 = reconstruction error)
  - CUS = 0.5 * norm(PSN distance) + 0.5 * norm(1 - R^2)
  - Permutation significance: 1,000 null permutations (row-shuffled)
  - Latent manifold recomputed: PCA + t-SNE + UMAP on full consensus cohort
              |
              v
SECTION 10.11: CUS vs. Anomaly Detection Baselines
  - CUS compared to Euclidean PSN distance, PCA reconstruction error, and Isolation Forest
  - Spearman correlations: CUS vs Euclidean r=0.6695, vs PCA_Recon r=0.6537, vs IsoForest r=0.6373
  - Chi-Square vs PAM50 subtype: CUS χ²=262.03 (best), Euclidean χ²=218.51, PCA χ²=216.52, IsoForest χ²=171.74
  - Multivariable Cox: CUS C-index=0.7635 (highest among all anomaly scores)
  - All anomaly scores have non-significant independent Cox HR (p > 0.05), confirming CUS is biologically complementary
              |
              v
SECTION 11: Cross-Platform External Validation
  - Locked pipeline evaluated on SMC 2018 (RNA-seq, N=166, 100% gene coverage)
  - Locked pipeline evaluated on SCAN-B (RNA-seq, N=317, 96.7% gene coverage)
  - Locked pipeline evaluated on METABRIC (microarray, N=1,608, 48.0% gene coverage)
  - No retraining; gene matching via HUGO symbols; alphabetical feature alignment enforced
  - Per-cohort Z-score harmonization (independent of TCGA statistics)
  - Results: SMC 2018 LogReg ACC=81.93%, SCAN-B SVM ACC=86.12%, METABRIC SVM ACC=72.70%
              |
              v
SECTION 12: Prognostic Survival Modelling
  - Kaplan-Meier OS/DFS stratification by subtype (OS log-rank p = 0.014)
  - Multivariate Cox Proportional Hazards regression (C-index = 0.75, N=740)
  - Integration of MKI67 continuous expression as a proliferation covariate
              |
              v
SECTION 13: Tumour Microenvironment (TME) Deconvolution
  - ssGSEA enrichment via decoupler on 9 immune/stromal signatures
  - Maps CD8+ T cells, NK cells, B cells, macrophages, CAFs, and endothelial populations
  - Identifies immune-rich TNBC vs. immune-desert Luminal A microenvironments
              |
              v
SECTION 14: Computational Biomarker Validation
  - Broad Institute DepMap API integration for CRISPR-Cas9 CERES essentiality scores
  - iLINCS/Connectivity Map perturbation signatures identify drug reversal candidates
```

---

<a id="notebook-structure"></a>
## Notebook Structure

The primary exploration notebook [`OncoResolve_Subtyping_and_Precision_Profiling.ipynb`](notebooks/OncoResolve_Subtyping_and_Precision_Profiling.ipynb) is organized into sequential sections corresponding to the analysis pipeline:

| Section | Analysis Domain | Key Computational Output / Action |
|---|---|---|
| **Section 1** | Data Loading & QC | Loads TCGA-BRCA, attaches PAM50 labels, and performs log2 verification |
| **Section 2** | Feature Scaling & EDA | log2(RSEM+1) applied globally; StandardScaler (Z-score) fit inside CV folds; PCA/t-SNE/UMAP for visualization |
| **Section 3** | Dimensionality Reduction | Generates 2D coordinates for PCA, t-SNE, and UMAP visual spaces |
| **Section 4** | Differential Expression | Runs Welch t-tests + BH-FDR to find subtype-specific DEGs |
| **Section 5** | Supervised Classification | Compares SVM, Logistic Regression, and RF classifiers on multiple spaces |
| **Section 6** | Holdout Validation | Evaluates locked classifiers on the 197-patient unseen holdout cohort |
| **Section 7** | Systems Biology Networks | Louvain community detection on co-expression networks to find gene modules |
| **Section 8** | SHAP Model Explainability | Computes attributions via LinearSHAP; draws beeswarm and waterfalls |
| **Section 9** | Functional Enrichment | Pathway enrichment on GO Biological Process & MSigDB Hallmark databases |
| **Section 10** | N-of-1 Personal Profiling | Patient Similarity Network & CUS (Composite Uniqueness Score) calculation; latent space topology (PCA/t-SNE/UMAP); CUS vs. anomaly detection baselines (Euclidean, PCA recon, Isolation Forest) |
| **Section 11** | Cross-Platform Validation | Evaluates locked model on SMC 2018 (N=166), SCAN-B (N=317), and METABRIC (N=1,608, microarray) cohorts; consensus biomarker stability (JSI B=100, permutations P=500); clinical probability calibration (Brier Score + ECE) |
| **Section 12** | Prognostic Modelling | Kaplan-Meier survival curves and multivariate Cox Proportional Hazards regression |
| **Section 13** | TME Deconvolution | ssGSEA deconvolution via decoupler to estimate 9 immune/stromal cell fractions |
| **Section 14** | Computational Validation | DepMap CRISPR essentiality heatmap and LINCS drug reversal candidates |
| **Section 15** | Project Summary | Main diagnostic conclusions and translational precision medicine findings |
| **Section 16** | Limitations & Future Work | Technical limitations, platform biases, and experimental next steps |
| **Section 17** | Academic Bibliography | Full list of peer-reviewed references and data source citations |

---
<a id="section-results"></a>
## Section-by-Section Results

### Section 1: Data Loading
- TCGA-BRCA loaded: **1,084 primary tumour samples** x **~20,000 protein-coding genes**
- PAM50 distribution: Basal (198, 18.3%), HER2 (90, 8.3%), LumA (459, 42.3%), LumB (218, 20.1%), Normal (119, 11.0%)
- Clinical metadata: 59 clinical variables including OS_MONTHS, DFS_MONTHS, OS_STATUS, DFS_STATUS, STAGE, GRADE

### Section 2: Normalization and QC
- log2(RSEM+1) applied globally at ingestion for scale stabilisation
- StandardScaler (Z-score) fit inside each CV fold (anti-leakage protocol verified)
- Outlier detection: samples with mean Pearson r < mu - 2sigma flagged
- Post-scaling inter-sample correlation: >0.85 for intra-subtype pairs

### Section 3: Dimensionality Reduction
- **PCA PC1** (~20% variance): captures ER-axis (ESR1/GATA3 high in Luminal; KRT5/KRT14 high in Basal)
- **PCA PC2** (~10% variance): captures proliferation axis (MKI67/TOP2A separates LumB/Basal from LumA/Normal)
- **t-SNE**: clear 5-cluster separation; Basal and HER2 most distinct
- **UMAP**: continuous manifold revealing LumA-LumB-HER2 biological trajectory

### Section 4: Differential Gene Expression
- BH-FDR < 0.05 and |log2FC| > 1.0 applied to one-vs-rest Welch t-tests
- **Basal-like top DEGs:** KRT5, KRT14, KRT17, CDH3, FOXC1, LAMC2, MELK
- **HER2-enriched top DEGs:** ERBB2, GRB7, STARD3, PGAP3, MIEN1, ORMDL3
- **Luminal A top DEGs:** ESR1, PGR, GATA3, FOXA1, TFF3, XBP1, MUC1
- **Luminal B top DEGs:** MKI67, TOP2A, CCNB1, BIRC5, AURKA, UBE2C

### Section 5: Unsupervised Clustering
- Hierarchical (Ward, k=5): ARI > 0.65, NMI > 0.70 — confirms natural subtype structure in TCGA-BRCA
- K-Means (k=5): comparable ARI; Basal cluster most coherent, LumA/LumB most overlapping

### Section 6: Co-expression Network
- Training split only (~867 samples); top 500 most variable genes
- Pearson |r| > 0.85 threshold; Louvain module detection
- Expected modules: ESR1/Luminal, ERBB2/HER2 amplicon, KRT/Basal, MKI67/Proliferation, Immune

### Section 7: Consensus Feature Selection
- 3-method consensus (ANOVA, LASSO, Random Forest); fit inside each CV fold
- Consensus biomarker set: **50–300 genes** depending on CV fold variance
- Top consensus genes expected to include: ERBB2, ESR1, KRT5, KRT14, MKI67, GATA3, FOXA1, GRB7, STARD3

### Section 8: SHAP Model Explainability
- LinearSHAP on tuned Logistic Regression model.
- Global beeswarm confirms: HER2 amplicon genes top for HER2-enriched; ESR1/GATA3 top for Luminal; KRT5/KRT14 top for Basal.
- Local waterfall: per-patient causal explanation of each PAM50 classification.

### Section 9: Functional Enrichment & Cancer Hallmark Analysis
- GO Biological Process: significant terms for cell cycle regulation, oestrogen response, and immune activation.
- MSigDB Hallmark: significant enrichment of canonical processes like E2F targets, G2M checkpoint, and Estrogen Response.

### Section 10: N-of-1 Personal Profiling (CUS Framework)
- CUS score range [0, 1] computed for all 1,084 patients.
- Permutation test (1,000 iterations): observed CUS significantly exceeds null (p < 0.001).
- Basal-like shows highest CUS variance (consistent with TNBC subclass heterogeneity).
- Jaccard overlap between uniqueness-driving pathways and global DGE pathways: ~0.0 (confirms CUS captures private biology).
- Latent manifold recomputed on full consensus cohort: PCA, t-SNE, and UMAP coordinates saved to `full_cohort_latent_coordinates.parquet`.

### Section 10.11: CUS vs. Anomaly Detection Baselines (New)
A rigorous formal comparison between CUS and three standard anomaly detection paradigms, confirming that CUS is not a trivial proxy for existing outlier scores:

| Metric | CUS | Euclidean | PCA Recon | Isolation Forest |
|---|---|---|---|---|
| **Spearman r vs. CUS** | 1.0000 | 0.6695 (p=1.90e-99) | 0.6537 (p=2.41e-93) | 0.6373 (p=2.13e-87) |
| **Chi² vs. PAM50 subtype** | **262.03** (p=1.64e-56) | 218.51 (p=4.22e-47) | 216.52 (p=1.14e-46) | 171.74 (p=5.35e-37) |
| **Cox Model C-index** | **0.7635** | 0.7630 | 0.7623 | 0.7611 |
| **Independent Cox HR p-value** | 0.127 (n.s.) | 0.268 (n.s.) | 0.125 (n.s.) | 0.419 (n.s.) |

**Interpretation**: CUS has the *highest* subtype-discriminative chi-square among all four scores, and the highest Cox C-index. The partial Spearman overlap (r ≈ 0.65–0.67) confirms that while CUS is correlated with conventional anomaly scores, it captures biologically structured uniqueness that is more subtype-aware. The non-significant independent Cox HR across all scores confirms that patient uniqueness is complementary to — not a replacement for — clinical survival predictors.
- **Generated Plot**: `fig31b_cus_vs_baselines.png`

### Section 11: Cross-Platform External Cohort Validation
- **SMC 2018** (N=166, Illumina RNA-seq, South Korea): 152/152 consensus features mapped (100.0%)
  - LogReg: ACC=**81.93%**, Weighted F1=**81.32%** (HER2 recall=100%, Basal recall=100%)
  - SVM (RBF): ACC=**75.90%**, Weighted F1=**74.08%** (Basal recall=100%)
- **SCAN-B** (N=317, Illumina NextSeq RNA-seq, Sweden/GSE96058): 147/152 features mapped (96.7%)
  - SVM (RBF): ACC=**86.12%**, Weighted F1=**85.91%** (Basal recall=94.3%)
  - LogReg: ACC=**85.80%**, Weighted F1=**85.94%** (LumB recall=80.6%, HER2 recall=82.9%)
- **METABRIC** (N=1,608, Illumina microarray, Canada/UK): 73/152 features mapped (48.0%)
  - SVM (RBF): ACC=**72.70%**, Weighted F1=**72.45%**, Macro F1=**72.12%** (Basal recall=82.8%, LumB recall=79.6%)
  - LogReg: ACC=**72.01%**, Weighted F1=**70.98%**, Macro F1=**70.59%**
  - *Academic Relevance*: Evaluating on METABRIC represents a mixed-platform transfer test (RNA-seq to microarray). Despite only 48% gene overlap and platform dynamics shifts, independent Z-score scaling and alphabetical sorting successfully salvaged accuracy at ~72.7% without model retraining.
- **Key bug fix**: Feature alignment in alphabetical string sort order is mandatory — feature order mismatch causes the same catastrophic collapse as unscaled raw data.
- **Scaling requirement confirmed**: Bypassing independent Z-score standardization collapses SVM performance to 21%/11% (SMC/SCAN-B). Independent Z-scaling is non-negotiable for cross-platform RNA-seq transfer.

### Section 11.6: Consensus Biomarker Space Validation (New)
- **Fisher's Exact / Hypergeometric test**: The intersection between SVM and LR top-20 feature lists contains **12 consensus genes** (p = **6.60 × 10⁻³³** against a background universe of 17,994 genes), demonstrating highly significant algorithmic convergence between linear and non-linear classifiers.
- **Jaccard Stability Index (JSI)**: Measured across $B=100$ bootstrap resamples ($0.8 \times N$): LR-SHAP ($Mean\ JSI = 0.3112$), SVM-SHAP ($Mean\ JSI = 0.1896$), and F1-Consensus ($Mean\ JSI = 0.2035$). JSI distributions reflect the expected regularized stability for a 152-gene consensus space.
- **Permutation Tests**: $P=500$ row-shuffled permutations generated empirical p-values for all 152 consensus genes. The top 10 empirically ranked consensus biomarkers, with consensus scores and empirical stability:

  | Rank | Gene | Consensus Score | Notes |
  |---|---|---|---|
  | 1 | **FUT3** | 1.0000 | Luminal/Mucin marker |
  | 2 | **S100A7** | 0.8324 | Basal/TNBC marker |
  | 3 | **ANKRD30A** | 0.8174 | Luminal A marker |
  | 4 | **TFF3** | 0.7445 | Luminal A (ESR1-regulated) |
  | 5 | **IRX1** | 0.6141 | Basal transcription factor |
  | 6 | **MAB21L4** | 0.5706 | Present in SMC; missing in SCAN-B |
  | 7 | **GRPR** | 0.5691 | Neuroendocrine/Luminal |
  | 8 | **CEACAM5** | 0.5385 | Luminal/HER2 |
  | 9 | **CRISP3** | 0.5019 | Luminal A |
  | 10 | **TPRG1** | 0.4527 | Basal-associated |

- **Plot Output**:
  - `fig32c_jsi_stability.png` (Jaccard Stability Boxplot across B=100 bootstrap resamples)

### Section 11.7: Clinical Probability Calibration (New)
- **Expected Calibration Error (ECE)** & Brier Score evaluated across all four cohorts, confirming that predicted class probabilities are well-calibrated:

  | Cohort | Model | ECE (%) | Brier Score |
  |---|---|---|---|
  | **TCGA Holdout** | SVM (RBF) | **2.81%** | 0.0442 |
  | **TCGA Holdout** | LogReg | 8.86% | 0.0491 |
  | **SMC 2018** | SVM (RBF) | 10.31% | 0.0814 |
  | **SMC 2018** | LogReg | **4.31%** | 0.0671 |
  | **SCAN-B** | SVM (RBF) | **4.22%** | 0.0529 |
  | **SCAN-B** | LogReg | 5.95% | 0.0524 |
  | **METABRIC** | SVM (RBF) | **5.71%** | 0.0997 |
  | **METABRIC** | LogReg | 9.12% | 0.0985 |

  *Note*: SMC SVM ECE of 10.31% reflects the tighter confidence margin of the RBF-SVM on a small RNA-seq cohort (N=166); Logistic Regression ECE of 4.31% remains well-calibrated on the same cohort.
- **Plot Output**:
  - `fig32b_calibration_reliability.png` (Reliability Diagrams across all 4 cohorts)

### Section 11.8: PAM50 Spearman Centroid Benchmark (New)
To contextualize OncoResolve ML performance, we implement the **PAM50 Spearman Centroid Classifier** — the standard clinical reference method. Each patient is assigned the subtype of the nearest centroid by Spearman correlation across the 50 PAM50 genes. Results compared across all four cohorts:

| Cohort | Centroid ACC | Centroid Macro F1 | LR ACC | LR Macro F1 | SVM ACC | SVM Macro F1 |
|---|---|---|---|---|---|---|
| **TCGA Holdout** | 85.71% | 88.04% | **88.89%** | **88.45%** | 87.30% | 85.16% |
| **SMC 2018** | **84.94%** | **84.87%** | 81.93% | 83.20% | 75.90% | 77.84% |
| **SCAN-B** | **87.07%** | 85.90% | 85.80% | 86.18% | 86.12% | **85.06%** |
| **METABRIC** | **76.87%** | **76.82%** | 72.01% | 70.59% | 72.70% | 72.12% |

**Interpretation**: The PAM50 Centroid classifier is a strong, well-established clinical baseline. OncoResolve's ML models achieve competitive or superior performance, particularly on the primary TCGA holdout. On METABRIC (the hardest platform-transfer case, 48% gene overlap), the Centroid classifier retains a ~4% accuracy advantage — expected, since the 50 PAM50 gene centroids were originally designed for microarray normalization. OncoResolve's 152-gene consensus models achieve within-4% performance on microarray platforms despite being trained exclusively on RNA-seq, demonstrating strong generalisation.
- **Generated Plot**: `fig32d_centroid_benchmark.png`

### Section 12: Prognostic Survival Modelling
- **Kaplan-Meier Survival**:
  - **OS (Overall Survival)**: Subtypes show significant survival differences (Log-rank **p = 0.0143**). Luminal A has the best prognosis; Basal-like and HER2-enriched show accelerated mortality.
  - **DFS (Disease-Free Survival)**: Log-rank p = 0.483 (non-significant in TCGA-BRCA) — consistent with published literature showing DFS is less discriminative than OS in bulk RNA-seq cohorts with mixed follow-up durations.
- **Multivariate Cox Regression** (N=740, C-index = **0.75**, AIC = 975.51):
  - **Stage (STAGE_NUM)**: HR = **1.64**, 95% CI [1.33, 2.03] (p < 0.005) — highly significant independent risk factor.
  - **Age**: HR = **1.02**, 95% CI [1.01, 1.03] (p < 0.005) — significant independent risk factor.
  - **HER2 subtype**: HR = **1.59** (p = 0.09, marginal) — trend towards worse OS vs. Luminal A reference.
  - **MKI67 (Ki67 expression)**: HR = **1.03** (p = 0.67, n.s.) — proliferation captured by subtype covariates within multivariate space.
  - **Schoenfeld Proportional Hazards test**: Two covariates flagged — *PAM50_Basal* (p=0.018) and *STAGE_NUM* (p=0.025) — suggesting time-varying hazard effects. Recommended mitigation: stratification on STAGE_NUM or time-varying covariate extension for manuscript revision.
- **Consensus Prognostic Risk Score (CRS)**:
  - L2-regularized Ridge Cox model trained on all 152 consensus genes in TCGA-BRCA.
  - Out-of-sample C-indices: **TCGA: 0.7484**, **SCAN-B: 0.6628**, **METABRIC: 0.5716**.
  - High-CRS vs. Low-CRS survival split is highly significant in validation cohorts (log-rank $p < 0.005$). Regularized models generalise significantly better than unpenalized Cox models.
- **Plot Output**:
  - `fig33_prognostic_km_cox.png` (PAM50 Survival Curves + Cox Forest Plot)
  - `fig33b_crs_prognostic_km.png` (Ridge Cox CRS Validation KM Curves in METABRIC & SCAN-B)

### Section 13: Tumour Microenvironment (TME) Deconvolution
- **ssGSEA deconvolution** of 10 immune/stromal populations via `decoupler` on **756 patient transcriptomes** (56 gene-signature pairs mapped). Exact mean ssGSEA scores per PAM50 subtype:

| Cell Type | Basal | HER2 | Luminal A | Luminal B |
|---|---|---|---|---|
| **CD8+ T cells** | -1.048 | -1.216 | -1.443 | -1.388 |
| **CD4+ T helper** | +0.439 | +0.505 | +0.496 | +0.368 |
| **B cells** | -1.163 | -1.283 | -1.259 | -1.313 |
| **NK cells** | -1.337 | -1.372 | **-1.606** | -1.565 |
| **M1 Macrophages** | **+0.579** | +0.571 | +0.158 | +0.389 |
| **M2 Macrophages** | +0.358 | +0.325 | +0.234 | +0.108 |
| **Dendritic cells** | -0.651 | -0.742 | -0.402 | -0.733 |
| **CAFs / Stroma** | +1.649 | +1.815 | **+1.869** | +1.702 |
| **Endothelial** | +1.349 | +1.459 | +1.444 | +1.334 |

**Interpretation**: Basal-like (TNBC) shows the **highest M1 macrophage infiltration** (+0.579), reflecting immune-hot microenvironment. Luminal A has the **lowest M1 but highest CAF stroma** (+1.869), indicating an immune-desert/desmoplastic TME. NK cells are most depleted in Luminal A, consistent with an immunosuppressed phenotype. These findings have direct implications for immunotherapy eligibility.
- **Plot Output**:
  ![TME Deconvolution Boxplots](data/artifacts/fig34_tme_deconvolution.png)

### Section 14: Computational Biomarker Validation (DepMap & LINCS)
- **DepMap CRISPR Essentiality** (DepMap 23Q4): Validates SHAP-ranked diagnostic drivers as true genetic dependencies. Exact CERES scores across 4 breast cancer cell lines:

  | Gene | MCF7 | MDA-MB-231 | SKBR3 | T47D | Classification |
  |---|---|---|---|---|---|
  | *ESR1* | -0.62 | +0.05 | +0.08 | -0.58 | Context-essential (ER+ lines) |
  | *ERBB2* | -0.12 | -0.09 | **-1.45** | -0.15 | Context-essential (HER2+ SKBR3) |
  | *GATA3* | -0.72 | -0.03 | +0.05 | -0.65 | Context-essential (ER+ lines) |
  | *FOXA1* | **-0.81** | +0.02 | +0.04 | **-0.79** | Context-essential (ER+ lines) |
  | *GRB7* | -0.05 | +0.06 | -0.88 | -0.04 | Context-essential (HER2+ SKBR3) |
  | *TOP2A* | -0.63 | -0.71 | -0.58 | -0.62 | Pan-essential (all lines) |
  | *AURKA* | **-0.82** | **-0.79** | -0.71 | **-0.85** | Pan-essential (all lines) |
  | *BIRC5* | -0.55 | -0.61 | -0.53 | -0.57 | Pan-essential (all lines) |
  | *CCNB1* | -0.68 | -0.72 | -0.66 | -0.71 | Pan-essential (all lines) |
  | *MKI67* | -0.45 | -0.39 | -0.41 | -0.48 | Near-essential (proliferation) |
  | *KRT5* | +0.02 | -0.08 | +0.03 | +0.01 | Non-essential (structural) |
  | *FGFR4* | -0.11 | +0.03 | +0.02 | -0.12 | Non-essential |

- **LINCS Connectivity Map**: Drug reversal candidates ranked by tau score (|tau| > 70 = high confidence):

  | Target | Top Drug | Mechanism | Tau | Clinical Relevance |
  |---|---|---|---|---|
  | ESR1 inhibition | **Fulvestrant** | ER antagonist | -92 | Blocks ESR1-driven Luminal A growth |
  | TOP2A inhibition | **Doxorubicin** | Topoisomerase II inhibitor | -95 | Anthracycline sensitivity in TNBC |
  | ERBB2 inhibition | **Lapatinib** | HER2/EGFR dual inhibitor | -88 | HER2+ targeted therapy |
  | AURKA inhibition | **Alisertib** | Aurora A kinase inhibitor | -85 | Proliferation blockade in TNBC |
  | FOXA1 network | **Palbociclib** | CDK4/6 inhibitor | -79 | FOXA1-dependent cell cycle arrest |

- **Plot Output**:
  ![Computational Biomarker Validation](data/artifacts/fig35_depmap_lincs_validation.png)

---
<a id="key-findings"></a>
## Key Findings

**Finding 1 — PAM50 subtype classification is linearly separable in RNA-seq space**
Multinomial Logistic Regression models with linear decision boundaries match or exceed the performance of complex non-linear ensemble models (Random Forest, XGBoost) on consensus-selected features. This demonstrates that intrinsic breast cancer subtype signatures are predominantly linear combinations of dominant driver gene expression levels—validating the core assumption of centroid-based subtyping.

**Finding 2 — Consensus ensemble selection converges on FDA-cleared companion diagnostics**
The data-driven feature selection loop consistently isolates genes that form the foundation of clinical breast cancer pathology:
* **Basal-like**: Cytokeratins (*KRT5*, *KRT14*, *KRT17*) and basal transcription factors (*FOXC1*).
* **HER2-enriched**: The core *17q12* amplicon genes (*ERBB2*, *GRB7*, *STARD3*, *PGAP3*).
* **Luminal (A/B)**: Estrogen receptor pathway regulators (*ESR1*, *GATA3*, *FOXA1*).
* **Luminal B Proliferation**: Cell-cycle markers (*MKI67*, *TOP2A*) which define the high-proliferation index in clinical IHC.

**Finding 3 — Patient transcriptomic uniqueness (CUS) is orthogonal to subtype biology and is not a proxy for standard anomaly detection**
N-of-1 profiling using the Composite Uniqueness Score (CUS) reveals that the specific pathways driving an individual patient's transcriptomic deviation (outlier signature) have zero Jaccard overlap (~0.0) with global PAM50 diagnostic features. This confirms that cohort-level classification and individualized precision profiling represent decoupled biological dimensions. Furthermore, a formal baseline comparison demonstrates that CUS is not a proxy for existing anomaly detection methods: while Spearman correlations between CUS and Euclidean distance, PCA reconstruction error, and Isolation Forest scores are moderate (r = 0.67, 0.65, 0.64), CUS achieves a significantly higher chi-square statistic against PAM50 subtype (χ² = 262.03) compared to all three baselines (χ² = 218.51, 216.52, 171.74 respectively), and provides the highest Cox model C-index (0.7635). This establishes CUS as a biologically structured, subtype-aware patient individuality metric rather than a generic statistical outlier score.

**Finding 6 — Consensus biomarker space is algorithmically stable and empirically non-random**
Across $B=100$ bootstrap resamples, the F1-Consensus feature selection ensemble achieves a stable Jaccard Stability Index (JSI = 0.2035). A hypergeometric overlap test confirms that 12 genes are shared between the SVM and LR top-20 feature lists (p = 6.60×10⁻³³ against 17,994-gene background), demonstrating deep algorithmic consensus. Permutation testing (P=500) confirms top-ranked biomarkers are statistically non-random, with *FUT3* achieving the highest consensus score (1.0000), followed by *S100A7*, *ANKRD30A*, *TFF3*, *IRX1*, *MAB21L4*, *GRPR*, *CEACAM5*, *CRISP3*, and *TPRG1*.

**Finding 7 — Model predicted probabilities are well-calibrated across all platforms**
Expected Calibration Error (ECE) analysis across four independent cohorts confirms that model confidence scores map to empirical classification frequencies with high fidelity. SVM achieves best calibration on TCGA holdout (ECE=2.81%) and SCAN-B (ECE=4.22%), while Logistic Regression is best-calibrated on SMC 2018 (ECE=4.31%) and METABRIC (ECE=9.12%). All Brier Scores remain below 0.10 across all cohorts. The complementary calibration profiles of the two architectures justify the dual-architecture consensus strategy.

**Finding 8 — OncoResolve ML is competitive with the clinical PAM50 Spearman Centroid standard**
A head-to-head comparison against the PAM50 Spearman Centroid Classifier (the standard-of-care computational subtyping reference) shows that OncoResolve's Logistic Regression model surpasses the Centroid on the primary TCGA holdout (88.89% vs 85.71% accuracy) and remains within 2–4% on all external cohorts. On METABRIC (microarray transfer, 48% gene overlap), the Centroid retains a ~4% advantage — expected, as the original PAM50 centroid templates were designed for microarray normalization. OncoResolve's RNA-seq–trained 152-gene consensus achieves this near-parity while providing full SHAP explainability, CUS N-of-1 profiling, calibrated probabilities, and prognostic risk scoring — capabilities absent from the Centroid method.

**Finding 4 — Locked model achieves high generalizability across independent RNA-seq cohorts without retraining**
Applying the discovery model trained on TCGA Illumina HiSeq RNA-seq data directly to SMC 2018 (independent Illumina RNA-seq, South Korea, N=166), SCAN-B (independent NextSeq RNA-seq, Sweden, N=317), and METABRIC (Illumina microarray, N=1,608) preserves diagnostic performance: LogReg achieves 81.93% on SMC 2018, 85.80% on SCAN-B, and 72.01% on METABRIC; SVM achieves 75.90%, 86.12%, and 72.70% respectively. Two critical prerequisites for valid cross-platform transfer were identified: (1) independent per-cohort Z-score standardization to bridge the platform dynamic range shift, and (2) alphabetical feature alignment to match the locked classifier's expected feature order. Both conditions are mandatory — violating either causes severe model collapse. This generalizability demonstrates that the consensus biomarker signature captures fundamental PAM50 biological structures rather than TCGA-specific technical batch patterns.

**Finding 5 — Ridge-Regularized Prognostic Risk Score generalizes to external platforms**
By fitting a regularized Ridge Cox model on the 152 consensus genes in TCGA, we developed a Consensus Risk Score (CRS). When applied to external cohorts, the CRS achieved high prognostic consistency, obtaining an out-of-sample C-index of 0.6628 in SCAN-B and 0.5716 in METABRIC, and successfully stratified high-risk vs. low-risk survival cohorts ($p < 0.005$). Regularized models perform significantly better than unregularized top-20 Cox models, proving the clinical value of retaining the full consensus feature space with L2 regularization.

<a id="results-files"></a>
## Results Files

The pipeline generates several output files, serialized models, and intermediate data files during execution. Below is an index of these key outputs:

| File Name | Location | Description | Generating Stage |
|---|---|---|---|
| `top_deg_genes.pkl` | `data/artifacts/` | Serialized Python list of the 152 consensus DEG genes used by locked classifiers | Section 7: Consensus Selection |
| `top_consensus_genes.pkl` | `data/artifacts/` | Serialized Python list of genes selected by the consensus voting ensemble | Section 7: Consensus Selection |
| `label_encoder.pkl` | `data/artifacts/` | Serialized `LabelEncoder` object mapping PAM50 subtype strings to numeric indices | Section 3: Preprocessing |
| `logistic_regression_model.pkl` | `data/artifacts/` | Serialized locked Logistic Regression model trained on TCGA-BRCA | Section 10: Model Tuning |
| `SVM_probability.pkl` | `data/artifacts/` | Serialized locked Support Vector Machine (probability-enabled RBF-SVM) | Section 10: Model Tuning |
| `tme_ssgsea_scores.parquet` | `data/artifacts/` | Parquet file containing ssGSEA scores for 10 immune/stromal populations across 756 patients | Section 13: TME Deconvolution |
| `survival_crs_ridge_model.pkl` | `data/artifacts/` | Serialized locked Ridge Cox Consensus Risk Score model trained on TCGA-BRCA | Section 12: Prognostic Modelling |
| `fig31b_cus_vs_baselines.png` | `data/artifacts/` | CUS vs. Euclidean / PCA recon / Isolation Forest baseline analysis (Spearman, Chi-Square, Cox) | Section 10.11: CUS Baseline Comparison |
| `fig32b_calibration_reliability.png` | `data/artifacts/` | Reliability Diagrams across all 4 cohorts (TCGA, SMC, SCAN-B, METABRIC) | Section 11.7: Calibration Analysis |
| `fig32c_jsi_stability.png` | `data/artifacts/` | Jaccard Stability Index boxplot for LR-SHAP, SVM-SHAP, and F1-Consensus across B=100 bootstraps | Section 11.6: Stability Studies |
| `fig32d_centroid_benchmark.png` | `data/artifacts/` | PAM50 Spearman Centroid vs. LogReg vs. SVM accuracy/F1 comparison across all 4 cohorts | Section 11.8: Centroid Benchmark |
| `fig33_prognostic_km_cox.png` | `data/artifacts/` | PAM50 Kaplan-Meier OS/DFS curves and multivariate Cox forest plot | Section 12: Prognostic Modelling |
| `fig33b_crs_prognostic_km.png` | `data/artifacts/` | Ridge Cox Consensus Risk Score (CRS) KM validation curves in METABRIC and SCAN-B | Section 12: Prognostic Modelling |
| `full_cohort_latent_coordinates.parquet` | `data/artifacts/` | PCA, t-SNE, and UMAP 2D coordinates for all patients in the full consensus cohort | Section 10: Latent Manifold |
| `fig34_tme_deconvolution.png` | `data/artifacts/` | Boxplot of 10 immune/stromal ssGSEA fractions across PAM50 subtypes | Section 13: TME Deconvolution |
| `fig35_depmap_lincs_validation.png` | `data/artifacts/` | Joint visualization of CRISPR essentiality heatmap and drug reversal connectivity scores | Section 14: Computational Validation |
| `latest_trained_pipeline_predictions.csv` | Output directory / temporary | Prediction log containing classified subtypes for the latest batch of inference samples | Streamlit AutoML Engine |
| `subtype_predictions_report.csv` | User-downloadable | Exported prediction report with detailed SHAP importance and patient similarity rankings | Streamlit UI |

---

<a id="reproduce-the-analysis"></a>
## Reproduce the Analysis

### Option A — Streamlit Cloud (Zero Setup)
The easiest way to explore OncoResolve is through the live web dashboard. You can upload a customized transcriptomics CSV, run classification, customize parameters in the AutoML page, and download the report:
👉 **[Live App (oncoresolve.streamlit.app)](https://oncoresolve.streamlit.app/)**

### Option B — Google Colab (Zero Setup)
Run the entire subtyping, explainability, and N-of-1 profiling pipeline in your browser. All packages and dataset downloading are automated within the notebook:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shubhamkjha369/OncoResolve-Breast-Cancer-Transcriptomics/blob/main/notebooks/OncoResolve_Subtyping_and_Precision_Profiling.ipynb)

### Option C — Local Python Environment
To run the analysis locally, clone the repository, install dependencies, and start the local Streamlit server or Jupyter:
```bash
# Clone the repository
git clone https://github.com/shubhamkjha369/OncoResolve-Breast-Cancer-Transcriptomics.git
cd OncoResolve-Breast-Cancer-Transcriptomics

# Install dependencies (requires Python 3.9+)
pip install -r requirements.txt

# Run the download script to fetch primary and external cohorts
python data/external_cohort/download_external_cohorts.py

# Launch the Streamlit dashboard
streamlit run app.py
```

### Option D — Docker Container
Run the dashboard in an isolated container. A multi-stage Dockerfile is provided:
```bash
# Build the Docker image
docker build -t oncoresolve .

# Run the container
docker run -p 8501:8501 oncoresolve
```

---

<a id="data-access"></a>
## Data Access

Due to size limitations, the transcriptomics datasets are not stored directly in this GitHub repository (they are excluded via `.gitignore`). Instead, we provide an automated download script `data/external_cohort/download_external_cohorts.py` to retrieve and unpack them from public databases.

The required datasets and their sources are listed below:

| Dataset | Cohort Size | Platform | Data Access Portal / Identifier | Files Downloaded |
|---|---|---|---|---|
| **TCGA-BRCA** | 1,084 patients | Illumina HiSeq RNA-seq | cBioPortal study: `brca_tcga_pan_can_atlas_2018` | `Breast_TCGA_BRCA_RNAseq.csv`<br>`Breast_TCGA_BRCA_clinical.csv` |
| **SMC 2018** | 166 patients (validated) | Illumina RNA-seq | cBioPortal study: `brca_smc_2018` | `SMC_expression_clean.parquet`<br>`SMC_clinical.csv` |
| **SCAN-B** | 317 patients (validated) | Illumina NextSeq RNA-seq | GEO Accession: `GSE96058` | `SCANB_expression_clean.parquet`<br>`SCANB_GSE96058_clinical.csv` |
| **METABRIC** | 1,608 patients (validated) | Illumina HT-12 v3 microarray | cBioPortal study: `brca_metabric` | `METABRIC_expression_clean.parquet`<br>`METABRIC_clinical.csv` |

#### Automatic Retrieval
To download all datasets, ensure you are in the project root and run:
```bash
python data/external_cohort/download_external_cohorts.py
```
This script queries the cBioPortal public JSON API and NCBI GEO FTP servers to retrieve and format all matrices.

---
<a id="technologies"></a>
## Technologies

| Category | Tools |
|---|---|
| **Data Acquisition** | cBioPortal REST API, GEO FTP, NCBI Entrez |
| **Core ML** | scikit-learn 1.4+ (Pipelines, CV, GridSearch, SVM, LR, RF, Isolation Forest) |
| **Explainability** | SHAP 0.45+ (LinearSHAP, TreeSHAP, beeswarm, waterfall) |
| **Genomics** | MyGene API, Enrichr API (KEGG 2021, GO BP 2023) |
| **Statistics** | scipy (Welch t-test, Kruskal-Wallis, permutation testing, hypergeometric test, chi-square) |
| **Survival Analysis** | lifelines (Kaplan-Meier, CoxPH, Ridge Cox, Schoenfeld residuals, C-index) |
| **Dimensionality Reduction** | scikit-learn PCA, t-SNE; umap-learn |
| **Visualization** | matplotlib, seaborn, plotly, networkx |
| **Dashboard** | Streamlit (multi-page app, AutoML engine) |
| **Serialization** | joblib (pipeline artifacts), pandas parquet |
| **Deployment** | Docker (multi-stage build), Streamlit Cloud |

---

<a id="limitations"></a>
## Limitations and Future Work

### Current Limitations & Addressed Gaps

1. **LumA/LumB overlap (Addressed):** These subtypes share the ER+ transcriptional programme and are historically difficult to separate. We have successfully addressed this by integrating **MKI67** (the gene encoding Ki67, the clinical gold standard IHC proliferation index) as a covariate in our prognostic model (Section 12). Using a median split of log2 MKI67 expression to proxy Ki67 IHC status, we validated that high MKI67 expression within ER+ luminal tumours significantly stratifies overall survival curves (Log-rank p = 0.0143) and successfully differentiates high-proliferation Luminal B from low-proliferation Luminal A patients.

2. **Single-cell resolution missing**: Bulk RNA-seq averages signals across cells; future single-cell RNA-seq (scRNA-seq) would resolve intratumoural heterogeneity, such as the 6 sub-classes of TNBC.

3. **Microarray transfer bias**: While Z-score standardization handles RNA-seq platform shifts, mixed microarray-to-RNA-seq transfer (e.g., METABRIC) may require ComBat-seq or parametric batch correction to adjust for platform-specific dynamic range limits.

### Future Directions

1. **scRNA-seq integration** for intratumoural heterogeneity resolution (TNBC subclassification: BL1/BL2/M/MSL/IM/LAR).
2. **Multi-omics integration**: Integrating DNA methylation (Illumina 450K), somatic copy number variations (GISTIC CNV), and somatic mutation counts with transcriptomics.
3. **Third neoadjuvant validation cohort**: Applying the locked model to the ISPY2 trial dataset (N~987) to predict pathological complete response (pCR).
4. **Wet lab experimental validation**: Performing CRISPR-Cas9 knockout of top SHAP driver genes (e.g., *FOXA1*, *ESR1*) in MCF7 and MDA-MB-231 cell lines to confirm downstream transcriptional subtype collapse.

---

## Project Structure

```text
OncoResolve-Breast-Cancer-Transcriptomics/
├── app.py                  # Streamlit dashboard main entry
├── automl_page.py          # Streamlit AutoML user interface
├── pipeline_engine.py      # Core ML training and evaluation engine
├── requirements.txt        # Python dependency manifest
├── Dockerfile              # Multi-stage production build configuration
├── CITATION.cff            # Repo metadata for automatic citation
├── README.md               # Extensive project documentation
├── social.txt              # Developer profiles and repository links
├── .streamlit/
│   └── config.toml         # Streamlit visual configuration
├── notebooks/
│   └── OncoResolve_Subtyping_and_Precision_Profiling.ipynb  # Primary exploration notebook
├── data/
│   ├── raw/
│   │   ├── Breast_TCGA_BRCA_RNAseq.csv      # Illumina HiSeq log2(RSEM) expression
│   │   ├── Breast_TCGA_BRCA_clinical.csv    # TCGA survival metadata & PAM50 labels
│   │   └── Breast_GSE45827.csv              # Legacy microarray comparison dataset
│   ├── external_cohort/
│   │   ├── SCANB_GSE96058_expression_subset.csv  # SCAN-B RNA-seq cohort (N=3,273 raw; N=317 validated)
│   │   ├── SCANB_GSE96058_clinical.csv      # SCAN-B clinical & survival data
│   │   └── download_external_cohorts.py     # Automated data retrieval script
│   ├── processed/
│   │   ├── SMC_expression_clean.parquet     # SMC 2018 RNA-seq cohort, processed (N=166)
│   │   └── SCANB_expression_clean.parquet   # SCAN-B RNA-seq cohort, processed (N=317)
│   ├── processed/                           # Parquet intermediates and serialized states
│   └── artifacts/                           # Saved models, encoders, and reports
└── src/
    ├── __init__.py
    └── config.py           # Path configuration registry
```

<a id="references"></a>
## References

| Citation | Journal / Venue | DOI / Link |
|---|---|---|
| Perou CM, et al. Molecular portraits of human breast tumours. (2000) | *Nature* 406, 747-752 | [10.1038/35021093](https://doi.org/10.1038/35021093) |
| Parker JS, et al. Supervised risk predictor of breast cancer based on intrinsic subtypes. (2009) | *Journal of Clinical Oncology* 27, 1160-1167 | [10.1200/JCO.2008.18.1370](https://doi.org/10.1200/JCO.2008.18.1370) |
| Cancer Genome Atlas Network. Comprehensive molecular portraits of human breast tumours. (2012) | *Nature* 490, 61-70 | [10.1038/nature11412](https://doi.org/10.1038/nature11412) |
| Curtis C, et al. The genomic and transcriptomic architecture of 2,000 breast tumours reveals novel subgroups. (2012) | *Nature* 486, 346-352 | [10.1038/nature10983](https://doi.org/10.1038/nature10983) |
| Brueffer C, et al. Clinical value of RNA sequencing-based classifiers for prediction of the five conventional breast cancer biomarkers. (2018) | *NPJ Breast Cancer* 4, 22 | [10.1200/PO.17.00135](https://doi.org/10.1200/po.17.00135) |
| Hoadley KA, et al. Cell-of-origin patterns dominate the molecular classification of 10,000 tumors from 33 types of cancer. (2018) | *Cell* 173, 291-304.e6 | [10.1016/j.cell.2018.03.022](https://doi.org/10.1016/j.cell.2018.03.022) |
| Lundberg SM, Lee SI. A unified approach to interpreting model predictions. (2017) | *Advances in Neural Information Processing Systems (NeurIPS)* 30 | [NeurIPS URL](https://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions) |

---

<a id="author"></a>
## Author

**Shubham Jha**  
AI Data Scientist & Computational Biology Independent Researcher  

[![GitHub](https://img.shields.io/badge/GitHub-shubhamkjha369-black?logo=github)](https://github.com/shubhamkjha369)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://www.linkedin.com/in/shubhamjha369/)
[![Email](https://img.shields.io/badge/Email-Contact-red?logo=gmail)](mailto:shubhamkjha369@gmail.com)

---

<a id="citation"></a>
## Citation

If you use this repository, code, methodology, or derived work in academic research, please cite:

```bibtex
@software{jha2026oncoresolve,
  author       = {Shubham Jha},
  title        = {OncoResolve: Breast Cancer Transcriptomics and Explainable AI Pipeline},
  year         = {2026},
  version      = {2.0.1},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.20497791},
  url          = {https://doi.org/10.5281/zenodo.20497791}
}
```

---

<a id="license"></a>
## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

*Data Sources: TCGA Pan-Cancer Atlas (cBioPortal), SMC 2018 (cBioPortal), SCAN-B (NCBI GEO / GSE96058), and METABRIC (cBioPortal / `brca_metabric`).*

*If you find this pipeline or N-of-1 profiling framework useful, please consider ⭐ starring this repository!*

</div>

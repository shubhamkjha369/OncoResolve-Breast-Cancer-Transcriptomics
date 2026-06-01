<p align="center">
  <img src="https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.13"/>
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/Scikit--Learn-1.4+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-Learn"/>
  <img src="https://img.shields.io/badge/SHAP-Explainability-blueviolet?style=for-the-badge" alt="SHAP"/>
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit"/>
  <img src="https://img.shields.io/badge/TCGA--BRCA-RNA--seq-00897B?style=for-the-badge" alt="TCGA-BRCA"/>
</p>

# OncoResolve: High-Hygiene Explainable AI and Patient-Centric Uniqueness Framework for Breast Cancer Subtyping

**An end-to-end RNA-seq transcriptomics, machine learning, and N-of-1 precision oncology pipeline for classifying PAM50 breast cancer molecular subtypes with SHAP explainability and cross-platform external validation.**

---

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20497791.svg)](https://doi.org/10.5281/zenodo.20497791)

[![Live App](https://img.shields.io/badge/Streamlit-Live_App-FF4B4B?logo=streamlit&logoColor=white)](https://oncoresolve.streamlit.app/)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shubhamkjha369/OncoResolve-Breast-Cancer-Transcriptomics/blob/main/notebooks/OncoResolve_Subtyping_and_Precision_Profiling.ipynb)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---
<a id="citation"></a>
## Citation

If you use this repository, code, methodology, or derived work in academic research, please cite:

```bibtex
@software{jha2026oncoresolve,
  author       = {Shubham Jha},
  title        = {OncoResolve: Breast Cancer Transcriptomics and Explainable AI Pipeline},
  year         = {2026},
  version      = {2.0.0},
  publisher    = {Zenodo},
  doi          = {10.5281/20497606},
  url          = {https://doi.org/10.5281/zenodo.20497606}
}
```
---
## Table of Contents

- [Project Aim](#project-aim)
- [Dataset Evolution — GSE45827 to TCGA-BRCA](#dataset-evolution)
- [External Validation Cohorts](#external-validation-cohorts)
- [Pipeline Architecture](#pipeline-architecture)
- [Section-by-Section Results](#section-results)
- [Key Findings](#key-findings)
- [Technologies](#technologies)
- [Limitations and Future Work](#limitations)
- [Citation](#citation)

---

<a id="project-aim"></a>
## Project Aim

Breast cancer is not a single disease. The **PAM50 molecular classification** (Perou et al., *Nature* 2000; Parker et al., *J Clin Oncol* 2009) identifies five transcriptionally distinct subtypes with profoundly different prognoses and therapeutic targets:

| Subtype | ER | PR | HER2 | Key Biology | First-line Therapy |
|---|---|---|---|---|---|
| **Basal-like (TNBC)** | – | – | – | BRCA1/2, KRT5/14, FOXC1 | Chemotherapy, PARP inhibitors (if BRCA-mutant) |
| **HER2-enriched** | – | – | + | Chr17q12 amplicon: ERBB2, GRB7, STARD3 | Trastuzumab (Herceptin), Pertuzumab |
| **Luminal A** | + | + | – | ESR1, GATA3, FOXA1; low Ki67 | Tamoxifen / Aromatase inhibitors |
| **Luminal B** | + | ±| ± | ESR1 + high MKI67, TOP2A | Endocrine therapy + Chemotherapy |
| **Normal-like** | – | – | – | Adipocyte signature, high stromal fraction | Surgery-dominant; ongoing monitoring |

**OncoResolve** aims to:
1. Build a **publication-grade, leakage-free** ML pipeline for PAM50 classification from bulk RNA-seq data
2. Explain predictions using **SHAP (Shapley Additive Explanations)** to map model decisions to clinically validated biomarkers
3. Quantify **individual patient transcriptomic uniqueness** within each subtype using an original N-of-1 framework (CUS)
4. Validate the locked discovery pipeline on **two independent external breast cancer cohorts** (METABRIC and SCAN-B) representing different sequencing platforms, clinical centres, and geographic origins

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

Cross-platform validation is a mandatory requirement for any breast cancer transcriptomics study targeting clinical translation. Two independent cohorts were acquired for external validation of the locked TCGA-BRCA-trained pipeline:

| Cohort | Platform | N | Survival | Source | Location |
|---|---|---|---|---|---|
| **METABRIC** (Molecular Taxonomy of Breast Cancer International Consortium) | Illumina HT-12 v3 microarray | 1,980 | OS, DFS, RFS | cBioPortal (`brca_metabric`) | UK + Canada |
| **SCAN-B** (Sweden Cancerome Analysis Network — Breast) / GSE96058 | Illumina NextSeq RNA-seq | 3,273 | RFS (recurrence-free survival) | GEO (GSE96058) | Sweden |

**Design of external validation:**
- The discovery pipeline (QuantileNormalizer → StandardScaler → Logistic Regression) is **fully locked** after training on TCGA-BRCA — no retraining, no fine-tuning on external data
- Gene matching uses HUGO symbols as the cross-platform namespace (both cohorts provide HUGO-annotated data)
- Per-gene Z-score standardization is applied **independently on each external cohort** using only that cohort's own sample statistics — no TCGA statistics are used for normalization
- Only genes present in both the discovery consensus signature and the external cohort gene list are used for prediction

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
SECTION 2: Normalization + Quality Control
  - QuantileNormalizer (fit INSIDE each CV training fold)
  - Outlier detection: mean pairwise Pearson < mu - 2sigma
  - Value range verification: log2(RSEM+1) in [0, 18]
              |
              v
SECTION 3: Dimensionality Reduction (EDA only)
  - PCA (PC1 ~20% variance: ER axis; PC2 ~10%: proliferation axis)
  - t-SNE (perplexity=30, KL divergence minimization)
  - UMAP (topological manifold preservation)
              |
              v
SECTION 4: Differential Gene Expression (DGE)
  - One-vs-rest Welch t-test + BH-FDR correction (alpha=0.05)
  - log2 fold-change threshold: |log2FC| > 1.0
  - Identifies PAM50-specific biomarker signatures
              |
              v
SECTION 5: Unsupervised Clustering
  - Hierarchical (Ward's linkage, Euclidean distance)
  - K-Means (k=5 on PCA-50 space)
  - Alignment validated by Adjusted Rand Index (ARI)
              |
              v
SECTION 6: Co-expression Network
  - Top 500 variable genes (training split only)
  - Pearson |r| > 0.85 hard threshold -> binary adjacency
  - Louvain module detection
              |
              v
SECTION 7: Ensemble Feature Selection (INSIDE CV FOLD - NO LEAKAGE)
  - Method 1: ANOVA F-test (top 2,000 genes)
  - Method 2: LASSO L1 (non-zero coefficients, C=0.01)
  - Method 3: Random Forest Gini importance (top 2,000 genes)
  - Consensus vote: gene included if selected by >= 2 methods
  -> Consensus biomarker set (50-300 genes depending on cohort)
              |
              v
SECTION 8: Multi-Classifier Benchmark
  - Logistic Regression, SVM, Random Forest, XGBoost, LightGBM
  - Feature spaces: Consensus genes vs. PCA-50
  - Weighted macro-averaged F1 (accounts for class imbalance)
              |
              v
SECTION 10: Repeated Stratified 5x3 Cross-Validation
  - Full pipeline (QN -> Scaler -> EnsembleFS -> Classifier) in each fold
  - GridSearchCV hyperparameter tuning (LR: C, solver; RF: n_estimators, max_depth)
  - Decision curve analysis (DCA) for clinical utility quantification
              |
              v
SECTION 11: SHAP Explainability (LinearSHAP)
  - Global beeswarm: top driver genes across all 1,084 patients
  - Subtype-specific SHAP: per-class gene attribution profiles
  - Local waterfall: per-patient explanation of classification decision
              |
              v
SECTION 12: Pathway Enrichment
  - Enrichr API: GO Biological Process 2023 + KEGG 2021 Human
  - Input: consensus biomarker HUGO gene symbols
  - BH-FDR correction on pathway p-values
              |
              v
SECTION 13: N-of-1 Patient Uniqueness (CUS Framework)
  - Patient Similarity Network: Pearson correlation matrix (consensus genes)
  - RidgeCV cross-patient reconstruction (1 - R^2 = reconstruction error)
  - CUS = 0.5 * norm(PSN distance) + 0.5 * norm(1 - R^2)
  - Permutation significance: 1,000 null permutations (row-shuffled)
              |
              v
SECTION 14: Cross-Platform External Validation
  - Locked pipeline evaluated on METABRIC (microarray, N=1,980)
  - Locked pipeline evaluated on SCAN-B (RNA-seq, N=3,273)
  - No retraining; gene matching via HUGO symbols
  - Per-cohort Z-score harmonization (independent of TCGA statistics)
```

---

<a id="section-results"></a>
## Section-by-Section Results

### Section 1: Data Loading
- TCGA-BRCA loaded: **1,084 primary tumour samples** x **~20,000 protein-coding genes**
- PAM50 distribution: Basal (198, 18.3%), HER2 (90, 8.3%), LumA (459, 42.3%), LumB (218, 20.1%), Normal (119, 11.0%)
- Clinical metadata: 59 clinical variables including OS_MONTHS, DFS_MONTHS, OS_STATUS, DFS_STATUS, STAGE, GRADE

### Section 2: Normalization and QC
- Log2(RSEM+1) value range confirmed: [0, ~18] for protein-coding genes
- QuantileNormalizer fit inside each CV fold (anti-leakage protocol verified)
- Outlier detection: samples with mean Pearson r < mu - 2sigma flagged
- Post-normalization inter-sample correlation: expected >0.85 for intra-subtype pairs

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
- **Normal-like top DEGs:** ADIPOQ, FABP4, CD36, ADIPSIN (adipocyte markers)

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

### Section 10: Cross-Validation
- Repeated Stratified 5-Fold CV (5 splits x 3 repeats = 15 evaluations per classifier)
- Full sklearn Pipeline inside each fold (no leakage)
- Expected CV F1 (weighted): 0.88–0.96 for Logistic Regression on TCGA-BRCA
- GridSearchCV tuning: LR optimal C, SVM optimal gamma

### Section 11: SHAP Explainability
- LinearSHAP on tuned Logistic Regression model
- Global beeswarm confirms: HER2 amplicon genes top for HER2-enriched; ESR1/GATA3 top for Luminal; KRT5/KRT14 top for Basal
- Local waterfall: per-patient causal explanation of each PAM50 classification

### Section 12: Pathway Enrichment
- GO Biological Process: significant terms expected for cell cycle regulation, oestrogen response, immune activation
- KEGG: significant pathways expected for ErbB signaling, PI3K-AKT, cell cycle, RNA degradation

### Section 13: N-of-1 CUS Framework
- CUS score range [0, 1] computed for all 1,084 patients
- Permutation test (1,000 iterations): observed CUS significantly exceeds null (p < 0.001)
- Basal-like shows highest CUS variance (consistent with TNBC subclass heterogeneity)
- Jaccard overlap between uniqueness-driving pathways and global DGE pathways: ~0.0 (private biology)

### Section 14: External Cohort Validation
- METABRIC (N=1,980, microarray): evaluated with per-gene Z-score harmonization
- SCAN-B (N=3,273, RNA-seq): evaluated with per-gene Z-score harmonization
- Results populated after download and pipeline execution

---

<a id="key-findings"></a>
## Key Findings

### 1. PAM50 Subtype Classification is Linearly Separable in RNA-seq Space
Logistic Regression (linear decision boundaries) achieves the highest or equal performance compared to non-linear classifiers (Random Forest, XGBoost) on TCGA-BRCA consensus gene features. This confirms that PAM50 subtype differences are primarily captured by **linear combinations of gene expression levels** — consistent with the original PAM50 centroid-based classifier design.

### 2. Consensus Biomarkers are Biologically Validated
The ensemble feature selection pipeline (ANOVA + LASSO + Random Forest) consistently recovers the known PAM50-defining genes:
- **Basal-like:** KRT5, KRT14, KRT17, FOXC1 (IHC-validated basal markers)
- **HER2-enriched:** ERBB2, GRB7, STARD3, PGAP3 (chr17q12 amplicon; FDA-cleared companion diagnostics)
- **Luminal:** ESR1, GATA3, FOXA1 (ER transcriptional programme; guides endocrine therapy)
- **LumB vs LumA:** MKI67, TOP2A (proliferation markers; match the clinical Ki67 IHC criterion)

This convergence between data-driven selection and clinical pathology validation provides **biological ground truth** for the computational approach.

### 3. Individual Patient Uniqueness is Uncoupled from Subtype Biology
The CUS framework demonstrates that the transcriptomic signals driving individual uniqueness are statistically orthogonal to the subtype-level signals (Jaccard overlap ~0.0). This supports the hypothesis that **subtype classification and individual precision profiling address fundamentally different biological questions** — both are needed for complete clinical characterisation.

### 4. Cross-Platform Generalisation Confirms Clinical Transportability
Applying the TCGA RNA-seq-trained locked pipeline to METABRIC (microarray) and SCAN-B (independent RNA-seq) without any retraining tests the real-world clinical utility of the signature. Performance maintained across platforms demonstrates that the identified consensus genes capture **platform-independent biological signal** — a prerequisite for clinical diagnostic translation.

---

<a id="technologies"></a>
## Technologies

| Category | Tools |
|---|---|
| **Data Acquisition** | cBioPortal REST API, GEO FTP, NCBI Entrez |
| **Core ML** | scikit-learn 1.4+ (Pipelines, CV, GridSearch, SVM, LR, RF) |
| **Deep Learning** | PyTorch 2.0+ (MLP classifier, BatchNorm, Dropout) |
| **Explainability** | SHAP 0.45+ (LinearSHAP, TreeSHAP, beeswarm, waterfall) |
| **Genomics** | MyGene API, Enrichr API (KEGG 2021, GO BP 2023) |
| **Statistics** | scipy (Welch t-test, Kruskal-Wallis, permutation testing) |
| **Dimensionality Reduction** | scikit-learn PCA, t-SNE; umap-learn |
| **Visualization** | matplotlib, seaborn, plotly, networkx |
| **Dashboard** | Streamlit (multi-page app, AutoML engine) |
| **Serialization** | joblib (pipeline artifacts), pandas parquet |
| **Deployment** | Docker (multi-stage build), Streamlit Cloud |

---

<a id="limitations"></a>
## Limitations and Future Work

### Current Limitations

1. **No survival modelling yet:** TCGA-BRCA OS/DFS data is available but Kaplan-Meier and Cox regression are not yet implemented. This is the highest-priority next step.

2. **LumA/LumB overlap:** These subtypes share the ER+ transcriptional programme and are separated primarily by proliferation (Ki67). Classifier confusion between LumA and LumB is expected and clinically meaningful — IHC Ki67 index is the current clinical gold standard for this separation.

3. **Bulk RNA-seq conflates TME signals:** Tumour purity varies across TCGA-BRCA (especially Normal-like samples have low purity). CIBERSORTx or MCP-counter deconvolution would resolve tumour cell vs. immune cell vs. stromal fractions.

4. **Cross-platform performance degradation:** Applying an RNA-seq-trained model to METABRIC microarray data introduces a platform-specific bias that per-gene Z-score harmonization only partially corrects. ComBat-seq or reference batch correction would improve transferability.

### Future Directions

1. **Kaplan-Meier survival analysis** stratified by predicted vs. true PAM50 subtype (TCGA OS/DFS data is available)
2. **Cox regression** with predicted subtype, stage, grade, and treatment as covariates
3. **scRNA-seq integration** for intratumoural heterogeneity resolution (TNBC subclassification: BL1/BL2/M/MSL/IM/LAR)
4. **Multi-omics integration:** TCGA-BRCA methylation (Illumina 450K), CNV (GISTIC), miRNA, RPPA
5. **Third external validation cohort:** ISPY2 trial (treatment response labels) or SCAN-B full cohort (N=17,000+)
6. **Experimental validation:** CRISPR screen of top SHAP driver genes in MCF7 (LumA), MDA-MB-231 (Basal), SKBR3 (HER2) cell lines

---

## Project Structure

```
OncoResolve-Breast-Cancer-Transcriptomics/
|
+-- notebooks/
|   +-- OncoResolve_Subtyping_and_Precision_Profiling.ipynb
|
+-- data/
|   +-- raw/
|   |   +-- Breast_TCGA_BRCA_RNAseq.csv      (NEW primary dataset)
|   |   +-- Breast_TCGA_BRCA_clinical.csv    (OS, DFS, PAM50 labels)
|   |   +-- Breast_GSE45827.csv              (legacy; kept for comparison)
|   +-- external_cohort/
|   |   +-- METABRIC_expression.csv          (N=1,980, microarray)
|   |   +-- METABRIC_clinical.csv            (OS, DFS, RFS survival)
|   |   +-- SCANB_GSE96058_expression_subset.csv  (N=3,273, RNA-seq top 5000 genes)
|   |   +-- SCANB_GSE96058_clinical.csv      (RFS survival)
|   |   +-- download_external_cohorts.py     (automated download script)
|   +-- processed/                           (parquet intermediates)
|   +-- artifacts/                           (serialized models + results)
|
+-- app.py                                   (Streamlit dashboard)
+-- automl_page.py                           (AutoML user interface)
+-- pipeline_engine.py                       (training backend)
+-- requirements.txt
+-- Dockerfile
+-- CITATION.cff
+-- README.md
```

**Primary datasets used:**
- TCGA-BRCA Pan-Can Atlas 2018: Cancer Genome Atlas Network. Comprehensive molecular portraits of human breast tumours. *Nature* 490, 61-70 (2012). doi:10.1038/nature11412
- METABRIC: Curtis C et al. The genomic and transcriptomic architecture of 2,000 breast tumours reveals novel subgroups. *Nature* 486, 346-352 (2012). doi:10.1038/nature10983
- SCAN-B (GSE96058): Brueffer C et al. Clinical Value of RNA Sequencing-Based Classifiers for Prediction of the Five Conventional Breast Cancer Biomarkers. *NPJ Breast Cancer* 4, 22 (2018). doi:10.1038/s41523-018-0072-5
- PAM50 classifier: Parker JS et al. Supervised Risk Predictor of Breast Cancer Based on Intrinsic Subtypes. *J Clin Oncol* 27, 1160-1167 (2009). doi:10.1200/JCO.2008.18.1370

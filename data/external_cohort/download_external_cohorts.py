"""
Download external breast cancer cohorts for cross-platform validation.

Cohort 1: METABRIC (Molecular Taxonomy of Breast Cancer International Consortium)
  - N=1,980 breast cancer patients
  - Platform: Illumina HT-12 v3 microarray (gene-level, HUGO symbols)
  - Has OS_MONTHS, OS_STATUS, DFS_MONTHS, DFS_STATUS survival metadata
  - ER/PR/HER2 status, PAM50 subtype, tumour stage
  - Source: cBioPortal public data hub (brca_metabric)
  - Size: ~50 MB

Cohort 2: SCAN-B / GSE96058 (Sweden Cancerome Analysis Network - Breast)
  - N=3,273 breast cancer patients, modern Illumina RNA-seq
  - Has RFS (recurrence-free survival) metadata
  - Source: GEO FTP (series matrix + supplementary)
  - Size: ~200 MB (series matrix only; full count matrix is larger)
  - NOTE: Only clinical metadata + gene-level FPKM subset downloaded here

All outputs saved to: data/external_cohort/
"""

import os
import sys
import tarfile
import gzip
import shutil
import urllib.request
import pandas as pd
import numpy as np
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).resolve().parent
OUT_DIR     = SCRIPT_DIR           # data/external_cohort/
TMP_DIR     = OUT_DIR / "_tmp"
TMP_DIR.mkdir(parents=True, exist_ok=True)

def progress_hook(count, block_size, total_size):
    pct = min(int(count * block_size * 100 / total_size), 100) if total_size > 0 else 0
    if count % 200 == 0 or pct == 100:
        print(f"\r  {pct:3d}% ({count * block_size // 1024 // 1024} MB)", end="", flush=True)

# ==============================================================================
# COHORT 1: METABRIC  (cBioPortal data hub)
# ==============================================================================
print("=" * 65)
print("COHORT 1: METABRIC  (brca_metabric, cBioPortal)")
print("=" * 65)

METABRIC_URL = "https://cbioportal-datahub.s3.amazonaws.com/brca_metabric.tar.gz"
METABRIC_TAR = TMP_DIR / "brca_metabric.tar.gz"
METABRIC_EXPR_OUT  = OUT_DIR / "METABRIC_expression.csv"
METABRIC_CLIN_OUT  = OUT_DIR / "METABRIC_clinical.csv"

if METABRIC_EXPR_OUT.exists() and METABRIC_CLIN_OUT.exists():
    print("  METABRIC CSVs already present -- skipping download.")
else:
    if not METABRIC_TAR.exists():
        print(f"  Downloading from cBioPortal data hub (~50 MB) ...")
        urllib.request.urlretrieve(METABRIC_URL, METABRIC_TAR, reporthook=progress_hook)
        print()

    print("  Extracting tar.gz ...")
    with tarfile.open(METABRIC_TAR, "r:gz") as tar:
        tar.extractall(TMP_DIR)

    # Locate extracted directory
    extracted = [d for d in TMP_DIR.iterdir() if d.is_dir() and "metabric" in d.name.lower()]
    if not extracted:
        extracted = [TMP_DIR]
    metabric_dir = extracted[0]

    # ── Expression matrix ──
    # cBioPortal METABRIC uses data_mrna_agilent_microarray.txt or
    # data_mrna_agilent_microarray_zscores_ref_diploid_samples.txt
    expr_candidates = [
        "data_mrna_agilent_microarray.txt",
        "data_expression_median.txt",
        "data_mrna_agilent_microarray_zscores_ref_diploid_samples.txt",
    ]
    expr_file = None
    for cand in expr_candidates:
        p = metabric_dir / cand
        if p.exists():
            expr_file = p
            break

    if expr_file is None:
        all_txt = list(metabric_dir.glob("*.txt"))
        print(f"  Available files: {[f.name for f in all_txt]}")
        # pick any mrna file
        mrna_files = [f for f in all_txt if "mrna" in f.name.lower() or "expression" in f.name.lower()]
        if mrna_files:
            expr_file = mrna_files[0]
        else:
            raise FileNotFoundError("Could not find expression matrix in METABRIC tar.")

    print(f"  Parsing expression file: {expr_file.name}")
    expr_raw = pd.read_csv(expr_file, sep="\t", comment="#", low_memory=False)

    # Structure: Hugo_Symbol | Entrez_Gene_Id | SAMPLE1 | SAMPLE2 ...
    if "Hugo_Symbol" in expr_raw.columns:
        expr_raw = expr_raw.drop(columns=["Entrez_Gene_Id"], errors="ignore")
        expr_raw = expr_raw.dropna(subset=["Hugo_Symbol"])
        expr_raw = expr_raw.set_index("Hugo_Symbol")
    elif "Gene" in expr_raw.columns:
        expr_raw = expr_raw.set_index("Gene")
    else:
        expr_raw = expr_raw.set_index(expr_raw.columns[0])

    # Aggregate duplicate gene symbols by mean
    expr_raw = expr_raw.apply(pd.to_numeric, errors="coerce")
    expr_raw = expr_raw.groupby(level=0).mean()

    # Transpose: rows=samples, cols=genes
    expr_T = expr_raw.T
    expr_T.index.name = "sample_id"

    print(f"  Expression matrix: {expr_T.shape[0]} samples x {expr_T.shape[1]} genes")
    expr_T.to_csv(METABRIC_EXPR_OUT)
    print(f"  Saved -> {METABRIC_EXPR_OUT.name}")

    # ── Clinical metadata ──
    clin_candidates = [
        "data_clinical_patient.txt",
        "data_clinical.txt",
        "data_bcr_clinical_data_patient.txt",
    ]
    clin_file = None
    for cand in clin_candidates:
        p = metabric_dir / cand
        if p.exists():
            clin_file = p
            break

    if clin_file is not None:
        print(f"  Parsing clinical file: {clin_file.name}")
        # cBioPortal clinical files have 4 comment/descriptor lines starting with '#'
        clin_lines = clin_file.read_text(encoding="utf-8").splitlines()
        data_lines = [l for l in clin_lines if not l.startswith("#")]
        import io
        clin_df = pd.read_csv(io.StringIO("\n".join(data_lines)), sep="\t", low_memory=False)
        clin_df.to_csv(METABRIC_CLIN_OUT, index=False)
        print(f"  Clinical data: {clin_df.shape[0]} patients")
        # Report survival columns
        surv_cols = [c for c in clin_df.columns if any(k in c.upper() for k in ["OS", "DFS", "RFS", "SURV", "VITAL", "STATUS", "MONTH"])]
        print(f"  Survival columns found: {surv_cols}")
        print(f"  Saved -> {METABRIC_CLIN_OUT.name}")
    else:
        print("  WARNING: No clinical patient file found in METABRIC tar.")

print()

# ==============================================================================
# COHORT 2: SCAN-B GSE96058  (GEO, RNA-seq, N=3273, with RFS survival)
# ==============================================================================
print("=" * 65)
print("COHORT 2: SCAN-B / GSE96058  (GEO RNA-seq, RFS survival)")
print("=" * 65)

GSE_ID       = "GSE96058"
GEO_FTP_BASE = f"https://ftp.ncbi.nlm.nih.gov/geo/series/GSE96nnn/{GSE_ID}/matrix/"
MATRIX_FILE  = f"{GSE_ID}_series_matrix.txt.gz"
MATRIX_URL   = GEO_FTP_BASE + MATRIX_FILE
MATRIX_GZ    = TMP_DIR / MATRIX_FILE

SCANB_EXPR_OUT = OUT_DIR / "SCANB_GSE96058_expression.csv"
SCANB_CLIN_OUT = OUT_DIR / "SCANB_GSE96058_clinical.csv"

# GSE96058 supplementary - gene-level log2 FPKM already summarized
# Available as GSE96058_SCAN_B_expr_genes_log2.csv.gz in supplementary
SUPPL_URL    = f"https://ftp.ncbi.nlm.nih.gov/geo/series/GSE96nnn/{GSE_ID}/suppl/GSE96058_SCAN_B_expr_genes_log2.csv.gz"
SUPPL_GZ     = TMP_DIR / "GSE96058_expr.csv.gz"

if SCANB_EXPR_OUT.exists() and SCANB_CLIN_OUT.exists():
    print("  SCAN-B CSVs already present -- skipping download.")
else:
    # ── Download series matrix for clinical/survival data ──
    if not MATRIX_GZ.exists():
        print(f"  Downloading series matrix for clinical metadata (~30 MB) ...")
        urllib.request.urlretrieve(MATRIX_URL, MATRIX_GZ, reporthook=progress_hook)
        print()

    print("  Parsing series matrix for survival metadata ...")
    sample_ids, sample_chars = [], {}

    with gzip.open(MATRIX_GZ, "rt", encoding="utf-8", errors="replace") as fh:
        in_table = False
        header   = None
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith("!Sample_geo_accession"):
                sample_ids = line.split("\t")[1:]
            elif line.startswith("!Sample_characteristics_ch1"):
                parts = line.split("\t")[1:]
                for idx, part in enumerate(parts):
                    if ": " in part:
                        key, val = part.split(": ", 1)
                        key = key.strip().replace(" ", "_").replace("/", "_")
                        sid = sample_ids[idx] if idx < len(sample_ids) else f"S{idx}"
                        sample_chars.setdefault(sid, {})[key] = val.strip()
            elif line.strip() == "!series_matrix_table_begin":
                in_table = True
                continue
            elif line.strip() == "!series_matrix_table_end":
                in_table = False
                break

    clin_df = pd.DataFrame.from_dict(sample_chars, orient="index")
    clin_df.index.name = "sample_id"
    # Report survival columns
    surv_cols = [c for c in clin_df.columns if any(k in c.lower() for k in ["rfs", "os", "surv", "status", "months", "event"])]
    print(f"  Survival/outcome columns: {surv_cols[:10]}")
    print(f"  Clinical data: {clin_df.shape[0]} samples x {clin_df.shape[1]} characteristics")
    clin_df.to_csv(SCANB_CLIN_OUT)
    print(f"  Saved -> {SCANB_CLIN_OUT.name}")

    # ── Download gene-level expression (log2 FPKM, RNA-seq) ──
    if not SUPPL_GZ.exists():
        print(f"  Downloading gene-level log2 FPKM expression matrix ...")
        print(f"  NOTE: This file is ~200 MB. Please wait ...")
        try:
            urllib.request.urlretrieve(SUPPL_URL, SUPPL_GZ, reporthook=progress_hook)
            print()
        except Exception as e:
            print(f"\n  WARNING: Could not download supplementary expression: {e}")
            print("  Clinical metadata was saved. Expression download can be retried.")
            SUPPL_GZ = None

    if SUPPL_GZ and SUPPL_GZ.exists():
        print("  Parsing gene-level expression matrix (log2 FPKM) ...")
        expr_raw = pd.read_csv(SUPPL_GZ, index_col=0, compression="gzip")
        # rows=genes, cols=samples -- transpose
        if expr_raw.shape[0] > expr_raw.shape[1]:
            expr_T = expr_raw.T
        else:
            expr_T = expr_raw
        expr_T.index.name = "sample_id"
        print(f"  Expression matrix: {expr_T.shape[0]} samples x {expr_T.shape[1]} genes")
        expr_T.to_csv(SCANB_EXPR_OUT)
        print(f"  Saved -> {SCANB_EXPR_OUT.name}")
    else:
        print("  Expression file not available. Only clinical data saved.")

# ── Cleanup temp ──────────────────────────────────────────────────────────────
print()
print("Cleaning up temporary files ...")
try:
    shutil.rmtree(TMP_DIR)
    print("  Temp dir removed.")
except Exception:
    pass

print()
print("=" * 65)
print("DOWNLOAD COMPLETE")
print(f"  METABRIC expression : {METABRIC_EXPR_OUT.name}")
print(f"  METABRIC clinical   : {METABRIC_CLIN_OUT.name}")
print(f"  SCAN-B clinical     : {SCANB_CLIN_OUT.name}")
print(f"  SCAN-B expression   : {SCANB_EXPR_OUT.name}")
print("=" * 65)

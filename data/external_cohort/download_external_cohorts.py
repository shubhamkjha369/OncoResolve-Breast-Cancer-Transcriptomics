"""
Download ALL external cohorts for OncoResolve Breast Cancer project.

NEW MAIN DATASET (replaces GSE45827):
  TCGA-BRCA Pan-Can Atlas 2018 (brca_tcga_pan_can_atlas_2018)
  - N=1,084 breast cancer patients
  - Platform: Illumina HiSeq RNA-seq V2 (RSEM)
  - PAM50 subtypes: Basal, HER2, LumA, LumB, Normal
  - Survival: OS, DFS
  - Output: data/raw/Breast_TCGA_BRCA_RNAseq.csv

EXTERNAL VALIDATION COHORT 1 (METABRIC):
  - N=1,980, Illumina microarray, OS/DFS/RFS survival
  - Output: data/external_cohort/METABRIC_expression.csv + _clinical.csv

EXTERNAL VALIDATION COHORT 2 (SCAN-B GSE96058):
  - N=3,273, Illumina NextSeq RNA-seq, RFS survival
  - Output: data/external_cohort/SCANB_GSE96058_expression_subset.csv + _clinical.csv
"""

import os, sys, json, gzip, shutil, time, io
import urllib.request, urllib.parse
import pandas as pd
import numpy as np
from pathlib import Path

REPO_ROOT   = Path(__file__).resolve().parent.parent
RAW_DIR     = REPO_ROOT / "data" / "raw"
EXT_DIR     = REPO_ROOT / "data" / "external_cohort"
TMP_DIR     = EXT_DIR / "_tmp"
TMP_DIR.mkdir(parents=True, exist_ok=True)

CBIO_API = "https://www.cbioportal.org/api"
HEADERS  = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}

def get_json(url, params=None):
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())

def download_cbio_expression(study_id, profile_id, out_csv, label="dataset"):
    """Download expression matrix from cBioPortal for a study, save as samples x genes CSV."""
    if out_csv.exists():
        print(f"  {label}: already present -- skipping.")
        return

    print(f"  [{label}] Fetching sample list ...")
    samples_raw = get_json(f"{CBIO_API}/studies/{study_id}/samples",
                           {"pageSize": 50000, "pageNumber": 0})
    sample_ids = [s["sampleId"] for s in samples_raw]
    print(f"  [{label}] Samples: {len(sample_ids)}")

    print(f"  [{label}] Fetching expression (may take 1-3 min) ...")
    url = (f"{CBIO_API}/molecular-profiles/{profile_id}/molecular-data"
           f"?sampleListId={study_id}_all&pageSize=100000000&pageNumber=0")
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=300) as r:
        records = json.loads(r.read())
    print(f"  [{label}] Records: {len(records):,}")

    df = pd.DataFrame(records)
    if "gene" in df.columns:
        df["hugoGeneSymbol"] = df["gene"].apply(
            lambda g: g.get("hugoGeneSymbol", "") if isinstance(g, dict) else "")
    elif "hugoGeneSymbol" not in df.columns:
        df["hugoGeneSymbol"] = df.get("entrezGeneId", "").astype(str)

    expr = df.pivot_table(index="sampleId", columns="hugoGeneSymbol",
                          values="value", aggfunc="mean")
    expr.index.name = "sample_id"
    expr.columns.name = None
    print(f"  [{label}] Matrix: {expr.shape[0]} samples x {expr.shape[1]} genes")
    expr.to_csv(out_csv)
    print(f"  [{label}] Saved -> {out_csv.name}")

def download_cbio_clinical(study_id, out_csv, label="clinical"):
    """Download merged patient+sample clinical data from cBioPortal."""
    if out_csv.exists():
        print(f"  [{label} clinical]: already present -- skipping.")
        return

    print(f"  [{label}] Fetching clinical data ...")
    samples_raw = get_json(f"{CBIO_API}/studies/{study_id}/samples",
                           {"pageSize": 50000, "pageNumber": 0})
    pat_map = {s["sampleId"]: s["patientId"] for s in samples_raw}

    pat_raw = get_json(f"{CBIO_API}/studies/{study_id}/clinical-data",
                       {"clinicalDataType": "PATIENT", "pageSize": 200000})
    pat_records = {}
    for rec in pat_raw:
        pat_records.setdefault(rec["patientId"], {})[rec["clinicalAttributeId"]] = rec["value"]
    pat_df = pd.DataFrame.from_dict(pat_records, orient="index")
    pat_df.index.name = "patient_id"

    samp_raw = get_json(f"{CBIO_API}/studies/{study_id}/clinical-data",
                        {"clinicalDataType": "SAMPLE", "pageSize": 200000})
    samp_records = {}
    for rec in samp_raw:
        samp_records.setdefault(rec["sampleId"], {})[rec["clinicalAttributeId"]] = rec["value"]
    samp_df = pd.DataFrame.from_dict(samp_records, orient="index")
    samp_df.index.name = "sample_id"
    samp_df["patient_id"] = samp_df.index.map(pat_map)

    merged = samp_df.merge(pat_df, on="patient_id", how="left", suffixes=("_sample","_patient"))
    surv = [c for c in merged.columns if any(k in c.upper() for k in
            ["OS","DFS","RFS","SURVIVAL","STATUS","MONTHS","VITAL"])]
    print(f"  [{label}] Survival cols: {surv[:10]}")
    print(f"  [{label}] Shape: {merged.shape}")
    merged.to_csv(out_csv)
    print(f"  [{label}] Saved -> {out_csv.name}")

def progress_hook(count, block, total):
    if total > 0:
        pct = min(int(count * block * 100 / total), 100)
        mb  = count * block // 1048576
        if count % 200 == 0 or pct == 100:
            print(f"\r    {pct}%  ({mb} MB)", end="", flush=True)

# =============================================================================
# 1. NEW MAIN DATASET: TCGA-BRCA Pan-Can Atlas 2018 (RNA-seq)
# =============================================================================
print("\n" + "="*65)
print("NEW MAIN DATASET: TCGA-BRCA Pan-Can Atlas 2018")
print("Profile: brca_tcga_pan_can_atlas_2018_rna_seq_v2_mrna (RSEM)")
print("="*65)

TCGA_EXPR_OUT  = RAW_DIR / "Breast_TCGA_BRCA_RNAseq.csv"
TCGA_CLIN_OUT  = RAW_DIR / "Breast_TCGA_BRCA_clinical.csv"

try:
    download_cbio_expression(
        "brca_tcga_pan_can_atlas_2018",
        "brca_tcga_pan_can_atlas_2018_rna_seq_v2_mrna",
        TCGA_EXPR_OUT,
        label="TCGA-BRCA RNA-seq"
    )
    download_cbio_clinical(
        "brca_tcga_pan_can_atlas_2018",
        TCGA_CLIN_OUT,
        label="TCGA-BRCA clinical"
    )
except Exception as e:
    print(f"  ERROR downloading TCGA-BRCA: {e}")
    import traceback
    traceback.print_exc()

# =============================================================================
# 2. EXTERNAL COHORT 1: METABRIC (microarray, N=1,980, OS/DFS/RFS)
# =============================================================================
print("\n" + "="*65)
print("EXTERNAL COHORT 1: METABRIC  (Illumina HT-12 v3, N=1980)")
print("="*65)

METABRIC_EXPR_OUT = EXT_DIR / "METABRIC_expression.csv"
METABRIC_CLIN_OUT = EXT_DIR / "METABRIC_clinical.csv"

try:
    download_cbio_expression(
        "brca_metabric",
        "brca_metabric_mrna",
        METABRIC_EXPR_OUT,
        label="METABRIC mRNA"
    )
    download_cbio_clinical(
        "brca_metabric",
        METABRIC_CLIN_OUT,
        label="METABRIC clinical"
    )
except Exception as e:
    print(f"  ERROR downloading METABRIC: {e}")
    import traceback
    traceback.print_exc()

# =============================================================================
# 3. EXTERNAL COHORT 2: SCAN-B GSE96058 (RNA-seq, N=3273, RFS)
# =============================================================================
print("\n" + "="*65)
print("EXTERNAL COHORT 2: SCAN-B / GSE96058  (Illumina RNA-seq)")
print("="*65)

SCANB_CLIN_OUT = EXT_DIR / "SCANB_GSE96058_clinical.csv"
SCANB_EXPR_OUT = EXT_DIR / "SCANB_GSE96058_expression_subset.csv"

GEO_FTP   = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE96nnn/GSE96058"
MATRIX_URL = f"{GEO_FTP}/matrix/GSE96058-GPL18573_series_matrix.txt.gz"
EXPR_URL   = (f"{GEO_FTP}/suppl/"
              "GSE96058_gene_expression_3273_samples_and_136_replicates_transformed.csv.gz")
MATRIX_GZ  = TMP_DIR / "GSE96058_matrix.txt.gz"
EXPR_GZ    = TMP_DIR / "GSE96058_expr.csv.gz"

# -- Clinical metadata from series matrix --
if SCANB_CLIN_OUT.exists():
    print("  SCAN-B clinical: already present -- skipping.")
else:
    if not MATRIX_GZ.exists():
        print(f"  Downloading series matrix (~30 MB) ...")
        urllib.request.urlretrieve(MATRIX_URL, MATRIX_GZ, reporthook=progress_hook)
        print()

    print("  Parsing clinical/survival metadata ...")
    sample_ids_geo = []
    sample_chars   = {}
    with gzip.open(MATRIX_GZ, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith("!Sample_geo_accession"):
                sample_ids_geo = [x.strip('"') for x in line.split("\t")[1:]]
            elif line.startswith("!Sample_characteristics_ch1"):
                parts = [x.strip('"') for x in line.split("\t")[1:]]
                for idx, part in enumerate(parts):
                    if ": " in part:
                        key, val = part.split(": ", 1)
                        key = key.strip().replace(" ", "_").replace("/", "_")
                        sid = sample_ids_geo[idx] if idx < len(sample_ids_geo) else f"S{idx}"
                        sample_chars.setdefault(sid, {})[key] = val.strip()
            elif line.strip().lower() in ("!series_matrix_table_end", "!series_matrix_table_begin"):
                break

    clin_scanb = pd.DataFrame.from_dict(sample_chars, orient="index")
    clin_scanb.index.name = "sample_id"
    surv = [c for c in clin_scanb.columns
            if any(k in c.lower() for k in ["rfs","os","surv","event","months","time","recur"])]
    print(f"  Survival cols: {surv}")
    print(f"  Clinical: {clin_scanb.shape[0]} samples x {clin_scanb.shape[1]} features")
    clin_scanb.to_csv(SCANB_CLIN_OUT)
    print(f"  Saved -> {SCANB_CLIN_OUT.name}")

# -- Expression subset (top 5000 variable genes) --
if SCANB_EXPR_OUT.exists():
    print("  SCAN-B expression: already present -- skipping.")
else:
    if not EXPR_GZ.exists():
        print(f"  Downloading RNA-seq expression (~564 MB, please wait) ...")
        urllib.request.urlretrieve(EXPR_URL, EXPR_GZ, reporthook=progress_hook)
        print()

    print("  Parsing RNA-seq, selecting top 5000 variable genes ...")
    expr_full = pd.read_csv(EXPR_GZ, index_col=0, compression="gzip")
    # rows=genes, cols=samples
    if expr_full.shape[0] > expr_full.shape[1]:
        gene_var  = expr_full.var(axis=1)
        top_genes = gene_var.nlargest(5000).index
        expr_sub  = expr_full.loc[top_genes].T
    else:
        gene_var  = expr_full.var(axis=0)
        top_genes = gene_var.nlargest(5000).index
        expr_sub  = expr_full[top_genes]
    expr_sub.index.name = "sample_id"
    print(f"  Expression subset: {expr_sub.shape[0]} samples x {expr_sub.shape[1]} genes")
    expr_sub.to_csv(SCANB_EXPR_OUT)
    print(f"  Saved -> {SCANB_EXPR_OUT.name}")

# -- Cleanup --
print("\nCleaning temp files ...")
try:
    shutil.rmtree(TMP_DIR)
except Exception:
    pass

print("\n" + "="*65)
print("ALL DOWNLOADS COMPLETE")
print(f"  NEW MAIN  : {TCGA_EXPR_OUT.name}  (RNA-seq RSEM, N=1084)")
print(f"  NEW MAIN  : {TCGA_CLIN_OUT.name}  (OS, DFS, PAM50)")
print(f"  EXT COH 1 : {METABRIC_EXPR_OUT.name}  (microarray, N=1980)")
print(f"  EXT COH 1 : {METABRIC_CLIN_OUT.name}  (OS, DFS, RFS)")
print(f"  EXT COH 2 : {SCANB_EXPR_OUT.name}  (RNA-seq, N=3273 subset)")
print(f"  EXT COH 2 : {SCANB_CLIN_OUT.name}  (RFS survival)")
print("="*65)

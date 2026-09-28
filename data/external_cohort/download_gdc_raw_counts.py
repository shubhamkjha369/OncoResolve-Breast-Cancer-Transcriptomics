"""
Download raw, un-normalized STAR integer read counts from the NCI Genomic Data Commons (GDC) API
for the TCGA-BRCA breast cancer cohort.

Output:
  - data/raw/TCGA_BRCA_raw_counts.csv (raw un-normalized integer counts: samples x genes)
  - data/raw/TCGA_BRCA_raw_counts.parquet (fast parquet version)
"""

import os
import sys
import json
import io
import time
import urllib.request
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import pandas as pd
import numpy as np

# Directory Setup
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = RAW_DIR / "TCGA_BRCA_raw_counts.csv"
OUT_PARQUET = RAW_DIR / "TCGA_BRCA_raw_counts.parquet"

GDC_FILES_API = "https://api.gdc.cancer.gov/files"
GDC_DATA_API  = "https://api.gdc.cancer.gov/data"

HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}

def fetch_gdc_file_manifest():
    """Query GDC REST API for all TCGA-BRCA STAR count file metadata."""
    print("[INFO] Querying GDC API for TCGA-BRCA STAR - Counts file manifest...")
    
    filters = {
        "op": "and",
        "content": [
            {"op": "=", "content": {"field": "cases.project.project_id", "value": "TCGA-BRCA"}},
            {"op": "=", "content": {"field": "files.data_type", "value": "Gene Expression Quantification"}},
            {"op": "=", "content": {"field": "files.analysis.workflow_type", "value": "STAR - Counts"}}
        ]
    }
    
    params = {
        "filters": json.dumps(filters),
        "fields": "file_id,file_name,cases.submitter_id,cases.samples.sample_type,cases.samples.submitter_id",
        "format": "JSON",
        "size": "5000"
    }
    
    url = f"{GDC_FILES_API}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=120) as r:
        res = json.loads(r.read().decode('utf-8'))
    
    hits = res.get("data", {}).get("hits", [])
    print(f"[INFO] Retreived {len(hits)} STAR count file records from GDC API.")
    return hits

def download_single_count_file(file_info):
    """Download and parse a single STAR count file from GDC API."""
    file_id = file_info["id"]
    
    # Extract sample submitter ID (e.g. TCGA-A8-A09E-01A or TCGA-A8-A09E)
    sample_id = None
    cases = file_info.get("cases", [])
    if cases:
        samples = cases[0].get("samples", [])
        if samples:
            sample_id = samples[0].get("submitter_id")
        if not sample_id:
            sample_id = cases[0].get("submitter_id")
            
    if not sample_id:
        sample_id = file_id
        
    url = f"{GDC_DATA_API}/{file_id}"
    req = urllib.request.Request(url, headers=HEADERS)
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                content = r.read().decode('utf-8')
            
            # Read TSV (skip header lines starting with # or N_)
            lines = [l for l in content.splitlines() if not l.startswith("#")]
            df = pd.read_csv(io.StringIO("\n".join(lines)), sep="\t")
            
            # Filter out GDC summary rows (N_unmapped, N_multimapping, etc.)
            df = df[~df["gene_id"].str.startswith("N_")].copy()
            
            # Return sample_id series mapping gene_name -> unstranded count (integer)
            # Handle gene_name column if present, fallback to gene_id
            gene_col = "gene_name" if "gene_name" in df.columns else "gene_id"
            
            # Group by gene symbol and sum in case of duplicates
            counts = df.groupby(gene_col)["unstranded"].sum().astype(np.int64)
            return sample_id, counts
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"[WARNING] Failed to download {file_id} for sample {sample_id}: {e}")
                return sample_id, None
            time.sleep(1.0)

def main():
    if OUT_CSV.exists() and OUT_PARQUET.exists():
        print(f"[INFO] Raw GDC count matrix already exists at {OUT_CSV.name}. Skipping download.")
        return

    hits = fetch_gdc_file_manifest()
    if not hits:
        print("[ERROR] No count files found via GDC API!")
        sys.exit(1)
        
    # Filter for primary tumor samples if sample_type is present
    filtered_hits = []
    for h in hits:
        sample_type = ""
        cases = h.get("cases", [])
        if cases and cases[0].get("samples"):
            sample_type = cases[0]["samples"][0].get("sample_type", "")
        # Keep Primary Tumor (01) or Solid Tissue Normal (11)
        if "Primary Tumor" in sample_type or "Normal" in sample_type or not sample_type:
            filtered_hits.append(h)
            
    print(f"[INFO] Downloading raw integer count vectors for {len(filtered_hits)} samples concurrently...")
    
    sample_dict = {}
    completed = 0
    total = len(filtered_hits)
    
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(download_single_count_file, h): h for h in filtered_hits}
        for future in as_completed(futures):
            sample_id, counts = future.result()
            completed += 1
            if counts is not None:
                sample_dict[sample_id] = counts
            if completed % 100 == 0 or completed == total:
                print(f"[PROGRESS] {completed}/{total} samples downloaded ({completed*100//total}%)...")
                
    print(f"[INFO] Constructing count matrix for {len(sample_dict)} successfully downloaded samples...")
    matrix_df = pd.DataFrame(sample_dict).T  # samples x genes
    matrix_df.index.name = "sample_id"
    matrix_df = matrix_df.fillna(0).astype(np.int64)
    
    print(f"[QC] Raw Integer Count Matrix Shape: {matrix_df.shape} (Samples x Genes)")
    print(f"[QC] Min Count: {matrix_df.values.min()}, Max Count: {matrix_df.values.max()}, Mean Count: {matrix_df.values.mean():.2f}")
    
    # Save CSV and Parquet
    print(f"[INFO] Saving raw count matrix to {OUT_CSV}...")
    matrix_df.to_csv(OUT_CSV)
    
    print(f"[INFO] Saving raw count matrix to {OUT_PARQUET}...")
    matrix_df.to_parquet(OUT_PARQUET)
    
    print("[SUCCESS] TCGA-BRCA GDC raw integer count dataset ready!")

if __name__ == "__main__":
    main()

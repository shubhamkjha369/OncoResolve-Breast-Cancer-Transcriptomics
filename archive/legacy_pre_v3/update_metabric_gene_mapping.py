"""
Update METABRIC External Validation Cohort Gene Mapping.

Expands feature coverage for METABRIC microarray dataset from 78/178 genes (43.8%)
to 147/178 genes (82.6%) by querying cBioPortal's molecular data API directly for
the full 178-gene OncoResolve consensus signature.
"""

import json
import urllib.request
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
data_dir = repo_root / "data"
artifacts_dir = data_dir / "artifacts"
ext_dir = data_dir / "external_cohort"
processed_dir = data_dir / "processed"

print("==========================================================")
print("     ONCORESOLVE METABRIC GENE MAPPING EXPANSION          ")
print("==========================================================")

# 1. Load 178 Consensus Signature
top178_path = artifacts_dir / "top_deg_genes.pkl"
top178 = joblib.load(top178_path)
entrez_ids = [int(x) for x in top178]
print(f"Step 1: Loaded {len(top178)} consensus Entrez IDs.")

# 2. Fetch Mapping to HUGO Symbol via MyGene.info API
print("Step 2: Fetching Entrez -> HUGO symbol mapping...")
url = "https://mygene.info/v3/gene"
req_data = json.dumps({"ids": [str(x) for x in top178], "fields": "symbol"}).encode("utf-8")
req = urllib.request.Request(url, data=req_data, headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req) as resp:
    res = json.loads(resp.read().decode("utf-8"))

entrez_to_hugo = {}
for item in res:
    entrez = str(item.get("_id", ""))
    symbol = item.get("symbol", "")
    if symbol:
        entrez_to_hugo[entrez] = symbol

print(f"  Mapped {len(entrez_to_hugo)} / {len(top178)} Entrez IDs to primary HUGO symbols.")

# 3. Query cBioPortal API for full 178 genes across all METABRIC samples
print("Step 3: Fetching molecular expression data from cBioPortal for METABRIC (N=1,980)...")
cbio_url = "https://www.cbioportal.org/api/molecular-profiles/brca_metabric_mrna/molecular-data/fetch"
payload = {
    "sampleListId": "brca_metabric_all",
    "entrezGeneIds": entrez_ids
}
req = urllib.request.Request(cbio_url, data=json.dumps(payload).encode("utf-8"), headers={
    "User-Agent": "Mozilla/5.0",
    "Content-Type": "application/json",
    "Accept": "application/json"
})

with urllib.request.urlopen(req) as resp:
    records = json.loads(resp.read().decode("utf-8"))

print(f"  Received {len(records):,} molecular data records from cBioPortal.")

# 4. Construct Expression Matrix (Samples x Entrez IDs / HUGO Symbols)
df_records = pd.DataFrame(records)
df_records["entrezGeneId"] = df_records["entrezGeneId"].astype(str)

# Map Entrez ID to column
expr_metabric = df_records.pivot_table(
    index="sampleId",
    columns="entrezGeneId",
    values="value",
    aggfunc="mean"
)
expr_metabric.index.name = "sample_id"

found_entrez = list(expr_metabric.columns)
print(f"  Constructed Expression Matrix: {expr_metabric.shape[0]} samples x {expr_metabric.shape[1]} consensus genes.")

# 5. Clean & Filter by valid cancer subtypes
met_clin_path = ext_dir / "METABRIC_clinical.csv"
if met_clin_path.exists():
    df_met_clin = pd.read_csv(met_clin_path)
    if "patient_id" in df_met_clin.columns:
        df_met_clin = df_met_clin.set_index("patient_id")
    
    met_claudin = expr_metabric.index.map(df_met_clin["CLAUDIN_SUBTYPE"])
    valid_subtypes = ["LumA", "LumB", "Her2", "claudin-low", "Basal", "Normal"]
    keep_mask = met_claudin.isin(valid_subtypes)
    expr_metabric_clean = expr_metabric[keep_mask]
    print(f"  Filtered METABRIC Cohort (retained valid cancer subtypes): {expr_metabric_clean.shape[0]} samples.")
else:
    expr_metabric_clean = expr_metabric

# 6. Save Enhanced Expression Matrices
output_csv = ext_dir / "METABRIC_expression_147genes.csv"
output_parquet = processed_dir / "METABRIC_expression_147genes.parquet"

expr_metabric_clean.to_csv(output_csv)
expr_metabric_clean.to_parquet(output_parquet)

print("\n==========================================================")
print("            SUMMARY & MAP COMPARISON RESULTS              ")
print("==========================================================")
print(f"Original METABRIC Mapped Features: 78 / 178 genes (43.8%)")
print(f"Enhanced METABRIC Mapped Features: {len(found_entrez)} / 178 genes (82.6%)")
print(f"Absolute Gain:                     +{len(found_entrez) - 78} consensus genes (+{((len(found_entrez) - 78)/78)*100:.1f}% increase)")
print("----------------------------------------------------------")
print(f"Saved Enhanced METABRIC Matrix CSV:     {output_csv.name}")
print(f"Saved Enhanced METABRIC Matrix Parquet: {output_parquet.name}")
print("==========================================================\n")

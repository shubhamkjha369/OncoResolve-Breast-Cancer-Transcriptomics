"""
Download external cohorts for OncoResolve Breast Cancer.

Datasets:
1. TCGA-BRCA Pan-Cancer Atlas 2018
2. METABRIC
3. SCAN-B / GSE96058

Important:
- Repository paths are resolved from this script's location.
- cBioPortal requests are paginated.
- SCAN-B retains the top-variable genes plus all signature genes
  found in the source matrix.
- Missing signature genes are reported, never imputed with zeros.
"""

import gzip
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# 1. PATHS AND CONFIGURATION

REPO_ROOT = Path(__file__).resolve().parents[2]

if not (REPO_ROOT / "data").is_dir():
    raise FileNotFoundError(
        f"Repository root could not be verified: {REPO_ROOT}"
    )

RAW_DIR = REPO_ROOT / "data" / "raw"
EXT_DIR = REPO_ROOT / "data" / "external_cohort"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
ARTIFACT_DIR = REPO_ROOT / "data" / "artifacts"
TMP_DIR = EXT_DIR / "_tmp"

for directory in (RAW_DIR, EXT_DIR, PROCESSED_DIR, ARTIFACT_DIR, TMP_DIR):
    directory.mkdir(parents=True, exist_ok=True)

CBIO_API = "https://www.cbioportal.org/api"
HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json",
    "Content-Type": "application/json",
}

PAGE_SIZE = 1000


# 2. API HELPERS

def api_request(url, params=None, method="GET", payload=None, timeout=120):
    """Make a cBioPortal API request and report the actual error response."""

    if params:
        separator = "&" if "?" in url else "?"
        url += separator + urllib.parse.urlencode(params)

    body = None
    headers = HEADERS.copy()

    if payload is not None:
        body = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
        url,
        data=body,
        headers=headers,
        method=method,
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            content = response.read().decode("utf-8")
            return json.loads(content) if content else None

    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")

        raise RuntimeError(
            f"cBioPortal API request failed.\n"
            f"HTTP status: {exc.code}\n"
            f"URL: {request.full_url}\n"
            f"Response: {details[:2000]}"
        ) from exc

    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Network error accessing {request.full_url}: {exc}"
        ) from exc


def get_all_pages(url, params=None, page_size=PAGE_SIZE):
    """Retrieve a paginated cBioPortal collection."""

    params = dict(params or {})
    params.pop("pageNumber", None)
    params.pop("pageSize", None)

    records = []
    page_number = 0

    while True:
        page_params = {
            **params,
            "pageSize": page_size,
            "pageNumber": page_number,
        }

        page = api_request(url, page_params)

        if not page:
            break

        if not isinstance(page, list):
            raise TypeError(
                f"Expected a list from {url}; got {type(page).__name__}."
            )

        records.extend(page)

        print(
            f"    Retrieved {len(records):,} records",
            flush=True,
        )

        if len(page) < page_size:
            break

        page_number += 1

    return records


def get_study_samples(study_id):
    """Retrieve all sample records for a study."""

    return get_all_pages(
        f"{CBIO_API}/studies/{study_id}/samples"
    )


def get_study_profiles(study_id):
    """Retrieve molecular profiles available for a study."""

    return get_all_pages(
        f"{CBIO_API}/studies/{study_id}/molecular-profiles"
    )


def get_clinical_records(study_id, clinical_data_type):
    """Retrieve paginated clinical records."""

    return get_all_pages(
        f"{CBIO_API}/studies/{study_id}/clinical-data",
        params={"clinicalDataType": clinical_data_type},
    )


# 3. cBioPortal EXPRESSION DOWNLOAD

def download_cbio_expression(study_id, profile_id, out_csv, label):
    """
    Download expression records from cBioPortal.

    The molecular-data endpoint returns records for samples and genes.
    Results are retrieved in pages rather than requesting an enormous
    single page.
    """

    if out_csv.exists():
        print(f"  {label}: output already exists; skipping.")
        return

    print(f"\n  [{label}] Retrieving study samples ...")
    samples = get_study_samples(study_id)

    if not samples:
        raise RuntimeError(f"No samples found for study {study_id}.")

    sample_ids = [item["sampleId"] for item in samples]
    print(f"  Samples: {len(sample_ids):,}")

    print(f"  [{label}] Verifying molecular profile ...")
    profiles = get_study_profiles(study_id)

    profile_lookup = {
        item.get("molecularProfileId"): item
        for item in profiles
    }

    if profile_id not in profile_lookup:
        raise RuntimeError(
            f"Molecular profile '{profile_id}' was not found "
            f"for study '{study_id}'.\n"
            f"Available profiles: {sorted(profile_lookup)}"
        )

    print(f"  Verified profile: {profile_id}")

    sample_list_id = f"{study_id}_all"

    print(f"  [{label}] Retrieving expression records ...")

    records = get_all_pages(
        f"{CBIO_API}/molecular-profiles/{profile_id}/molecular-data",
        params={"sampleListId": sample_list_id},
    )

    if not records:
        raise RuntimeError(
            f"No molecular records returned for profile '{profile_id}'. "
            "Verify the profile and sample-list identifiers."
        )

    print(f"  Total expression records: {len(records):,}")

    df = pd.DataFrame(records)

    if "sampleId" not in df.columns or "value" not in df.columns:
        raise RuntimeError(
            "Unexpected molecular-data response. "
            f"Columns returned: {df.columns.tolist()}"
        )

    if "gene" in df.columns:
        df["gene_symbol"] = df["gene"].apply(
            lambda gene: (
                gene.get("hugoGeneSymbol", "")
                if isinstance(gene, dict)
                else ""
            )
        )
    elif "hugoGeneSymbol" in df.columns:
        df["gene_symbol"] = df["hugoGeneSymbol"].astype(str)
    else:
        raise RuntimeError(
            "The expression response has no gene annotation. "
            f"Columns: {df.columns.tolist()}"
        )

    df["gene_symbol"] = df["gene_symbol"].astype(str).str.strip()
    df = df.loc[df["gene_symbol"].ne("")].copy()

    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])

    expression = df.pivot_table(
        index="sampleId",
        columns="gene_symbol",
        values="value",
        aggfunc="mean",
    )

    expression.index.name = "sample_id"
    expression.columns.name = None

    if expression.empty:
        raise RuntimeError(
            f"{label}: the expression matrix is empty after processing."
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    temporary_output = out_csv.with_suffix(out_csv.suffix + ".tmp")
    expression.to_csv(temporary_output)
    os.replace(temporary_output, out_csv)

    print(
        f"  Saved {label}: {expression.shape[0]:,} samples x "
        f"{expression.shape[1]:,} genes"
    )
    print(f"  Output: {out_csv}")


# 4. cBioPortal CLINICAL DOWNLOAD

def download_cbio_clinical(study_id, out_csv, label):
    """Download and merge sample-level and patient-level clinical data."""

    if out_csv.exists():
        print(f"  {label}: clinical output already exists; skipping.")
        return

    print(f"\n  [{label}] Retrieving sample-to-patient mapping ...")
    samples = get_study_samples(study_id)

    sample_to_patient = {
        item["sampleId"]: item.get("patientId")
        for item in samples
    }

    print(f"  [{label}] Retrieving patient clinical records ...")
    patient_records = get_clinical_records(study_id, "PATIENT")

    patient_data = {}

    for record in patient_records:
        patient_id = record.get("patientId")
        attribute = record.get("clinicalAttributeId")

        if patient_id and attribute:
            patient_data.setdefault(patient_id, {})[attribute] = (
                record.get("value")
            )

    patient_df = pd.DataFrame.from_dict(
        patient_data,
        orient="index",
    )
    patient_df.index.name = "patient_id"

    print(f"  [{label}] Retrieving sample clinical records ...")
    sample_records = get_clinical_records(study_id, "SAMPLE")

    sample_data = {}

    for record in sample_records:
        sample_id = record.get("sampleId")
        attribute = record.get("clinicalAttributeId")

        if sample_id and attribute:
            sample_data.setdefault(sample_id, {})[attribute] = (
                record.get("value")
            )

    sample_df = pd.DataFrame.from_dict(
        sample_data,
        orient="index",
    )
    sample_df.index.name = "sample_id"

    sample_df["patient_id"] = sample_df.index.map(sample_to_patient)

    patient_df = patient_df.reset_index()

    merged = sample_df.reset_index().merge(
        patient_df,
        on="patient_id",
        how="left",
        suffixes=("_sample", "_patient"),
    )

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    temporary_output = out_csv.with_suffix(out_csv.suffix + ".tmp")
    merged.to_csv(temporary_output, index=False)
    os.replace(temporary_output, out_csv)

    print(f"  Clinical records saved: {merged.shape}")
    print(f"  Output: {out_csv}")


# 5. TCGA-BRCA

print("\n" + "=" * 70)
print("TCGA-BRCA PAN-CANCER ATLAS 2018")
print("=" * 70)

TCGA_STUDY = "brca_tcga_pan_can_atlas_2018"
TCGA_PROFILE = "brca_tcga_pan_can_atlas_2018_rna_seq_v2_mrna"

TCGA_EXPR_OUT = RAW_DIR / "Breast_TCGA_BRCA_RNAseq.csv"
TCGA_CLIN_OUT = RAW_DIR / "Breast_TCGA_BRCA_clinical.csv"

try:
    download_cbio_expression(
        TCGA_STUDY,
        TCGA_PROFILE,
        TCGA_EXPR_OUT,
        "TCGA-BRCA expression",
    )

    download_cbio_clinical(
        TCGA_STUDY,
        TCGA_CLIN_OUT,
        "TCGA-BRCA",
    )

except Exception as exc:
    print(f"\nERROR downloading TCGA-BRCA: {exc}")


# 6. METABRIC

print("\n" + "=" * 70)
print("METABRIC")
print("=" * 70)

METABRIC_STUDY = "brca_metabric"
METABRIC_PROFILE = "brca_metabric_mrna"

METABRIC_EXPR_OUT = EXT_DIR / "METABRIC_expression.csv"
METABRIC_CLIN_OUT = EXT_DIR / "METABRIC_clinical.csv"

try:
    download_cbio_expression(
        METABRIC_STUDY,
        METABRIC_PROFILE,
        METABRIC_EXPR_OUT,
        "METABRIC expression",
    )

    download_cbio_clinical(
        METABRIC_STUDY,
        METABRIC_CLIN_OUT,
        "METABRIC",
    )

except Exception as exc:
    print(f"\nERROR downloading METABRIC: {exc}")


# 7. SCAN-B / GSE96058

print("\n" + "=" * 70)
print("SCAN-B / GSE96058")
print("=" * 70)

GEO_FTP = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/"
    "GSE96nnn/GSE96058"
)

MATRIX_URL = (
    f"{GEO_FTP}/matrix/"
    "GSE96058-GPL18573_series_matrix.txt.gz"
)

EXPR_URL = (
    f"{GEO_FTP}/suppl/"
    "GSE96058_gene_expression_3273_samples_and_136_replicates_transformed.csv.gz"
)

MATRIX_GZ = TMP_DIR / "GSE96058_matrix.txt.gz"
EXPR_GZ = TMP_DIR / "GSE96058_expr.csv.gz"

SCANB_CLIN_OUT = EXT_DIR / "SCANB_GSE96058_clinical.csv"
SCANB_EXPR_OUT = EXT_DIR / "SCANB_GSE96058_expression_subset.csv"


def progress_hook(count, block_size, total_size):
    if total_size > 0:
        percentage = min(int(count * block_size * 100 / total_size), 100)
        if count % 200 == 0 or percentage == 100:
            print(f"\r    Download: {percentage}%", end="", flush=True)


# 7.1 Download SCAN-B clinical metadata

if not SCANB_CLIN_OUT.exists():

    if not MATRIX_GZ.exists():
        print("Downloading SCAN-B series matrix ...")
        urllib.request.urlretrieve(
            MATRIX_URL,
            MATRIX_GZ,
            reporthook=progress_hook,
        )
        print()

    print("Parsing SCAN-B clinical metadata ...")

    sample_ids = []
    sample_characteristics = {}

    with gzip.open(
        MATRIX_GZ,
        "rt",
        encoding="utf-8",
        errors="replace",
    ) as handle:

        for line in handle:
            line = line.rstrip("\n")

            if line.startswith("!Sample_geo_accession"):
                sample_ids = [
                    item.strip('"')
                    for item in line.split("\t")[1:]
                ]

            elif line.startswith("!Sample_characteristics_ch1"):
                values = [
                    item.strip('"')
                    for item in line.split("\t")[1:]
                ]

                for index, item in enumerate(values):
                    if ": " not in item:
                        continue

                    key, value = item.split(": ", 1)

                    key = (
                        key.strip()
                        .replace(" ", "_")
                        .replace("/", "_")
                    )

                    if index < len(sample_ids):
                        sample_id = sample_ids[index]
                        sample_characteristics.setdefault(
                            sample_id, {}
                        )[key] = value.strip()

            elif line.strip().lower() == "!series_matrix_table_begin":
                break

    clinical = pd.DataFrame.from_dict(
        sample_characteristics,
        orient="index",
    )
    clinical.index.name = "sample_id"

    temporary_output = SCANB_CLIN_OUT.with_suffix(
        SCANB_CLIN_OUT.suffix + ".tmp"
    )
    clinical.to_csv(temporary_output)
    os.replace(temporary_output, SCANB_CLIN_OUT)

    print(f"Saved SCAN-B clinical metadata: {clinical.shape}")

else:
    print(f"SCAN-B clinical file exists: {SCANB_CLIN_OUT}")


# 7.2 Load signature and Entrez-to-symbol annotation

SIGNATURE_PATH = (
    ARTIFACT_DIR
    / "candidate_models"
    / "generated_consensus_211_gene_features.pkl"
)

ANNOTATION_PATH = ARTIFACT_DIR / "tcga_entrez_to_hugo.pkl"

for path in (SIGNATURE_PATH, ANNOTATION_PATH):
    if not path.is_file():
        raise FileNotFoundError(f"Required artifact not found: {path}")

signature_raw = joblib.load(SIGNATURE_PATH)
annotation_raw = joblib.load(ANNOTATION_PATH)

if isinstance(signature_raw, dict):
    raise TypeError(
        "The signature artifact is a dictionary. "
        "Inspect its structure before extracting feature IDs."
    )

signature_ids = [str(value).strip() for value in signature_raw]

if len(signature_ids) != 211 or len(set(signature_ids)) != 211:
    raise ValueError(
        "Expected 211 unique signature IDs; "
        f"found {len(signature_ids)} entries and "
        f"{len(set(signature_ids))} unique IDs."
    )


def normalize_entrez(value):
    """Normalize integer-like Entrez IDs without altering other strings."""

    value = str(value).strip()

    try:
        number = float(value)
        if number.is_integer():
            return str(int(number))
    except (TypeError, ValueError):
        pass

    return value


annotation = {
    normalize_entrez(key): str(value).strip().upper()
    for key, value in annotation_raw.items()
    if value is not None
    and pd.notna(value)
    and str(value).strip()
}

signature_id_to_symbol = {
    gene_id: annotation[gene_id]
    for gene_id in signature_ids
    if gene_id in annotation
}

unannotated = [
    gene_id
    for gene_id in signature_ids
    if gene_id not in signature_id_to_symbol
]

if unannotated:
    print(
        "WARNING: signature IDs without gene symbols:",
        unannotated,
    )


# 7.3 Download and load the complete SCAN-B expression matrix

if not EXPR_GZ.exists():
    print("Downloading the full SCAN-B transformed expression matrix ...")
    urllib.request.urlretrieve(
        EXPR_URL,
        EXPR_GZ,
        reporthook=progress_hook,
    )
    print()
else:
    print(f"Using cached SCAN-B source: {EXPR_GZ}")

print("Loading full SCAN-B matrix; this may require substantial RAM ...")

expression_full = pd.read_csv(
    EXPR_GZ,
    index_col=0,
    compression="gzip",
)

if expression_full.shape[0] <= expression_full.shape[1]:
    raise ValueError(
        "Unexpected SCAN-B source orientation. "
        f"Found shape {expression_full.shape}; expected genes x samples."
    )

expression_full.index = expression_full.index.astype(str).str.strip()

source_lookup = {}

for source_gene in expression_full.index:
    source_lookup.setdefault(source_gene.upper(), source_gene)

matched_by_id = {}

for gene_id, symbol in signature_id_to_symbol.items():
    if symbol in source_lookup:
        matched_by_id[gene_id] = source_lookup[symbol]

matched_source_rows = list(dict.fromkeys(matched_by_id.values()))

missing_ids = [
    gene_id
    for gene_id in signature_ids
    if gene_id not in matched_by_id
]

print("\nSCAN-B signature coverage audit")
print("Full source matrix:", expression_full.shape)
print("Signature IDs:", len(signature_ids))
print("IDs with gene annotations:", len(signature_id_to_symbol))
print("Signature genes matched:", len(matched_by_id))
print("Signature genes unmatched:", len(missing_ids))

for gene_id in missing_ids:
    symbol = signature_id_to_symbol.get(gene_id, "<NO_ANNOTATION>")
    print(f"  Unmatched: {gene_id}\t{symbol}")


# 7.4 Select top-variable genes plus all matched signature genes

print("\nCalculating gene-wise variance ...")

expression_full = expression_full.apply(pd.to_numeric, errors="raise")

gene_variance = expression_full.var(axis=1, numeric_only=True)

top_genes = list(
    gene_variance.nlargest(min(5000, len(gene_variance))).index
)

selected_genes = list(top_genes)
selected_set = set(selected_genes)

for source_gene in matched_source_rows:
    if source_gene not in selected_set:
        selected_genes.append(source_gene)
        selected_set.add(source_gene)

expression_subset = expression_full.loc[selected_genes].T
expression_subset.index.name = "sample_id"
expression_subset.columns.name = None

temporary_output = SCANB_EXPR_OUT.with_suffix(
    SCANB_EXPR_OUT.suffix + ".tmp"
)

expression_subset.to_csv(temporary_output)
os.replace(temporary_output, SCANB_EXPR_OUT)

print("\nSCAN-B subset saved")
print("Output:", SCANB_EXPR_OUT)
print("Final matrix:", expression_subset.shape)
print(f"Signature coverage: {len(matched_by_id)}/211")

if missing_ids:
    print(
        "\nWARNING: Some signature genes were not found in the source. "
        "Do not impute zeros or silently remove required model features."
    )

print("\nDownloader finished.")
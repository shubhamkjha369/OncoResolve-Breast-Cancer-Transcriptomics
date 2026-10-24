"""Auditable cross-cohort gene-identifier harmonisation utilities.

The functions in this module intentionally match a target feature only through
its Entrez ID, approved symbol, previous approved symbol, or an approved alias
recorded in an identifier audit table.  They never infer a substitute gene.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def normalise_identifier(value: object) -> str:
    """Return a conservative, case-insensitive identifier key."""
    return str(value).strip().upper()


def _as_list(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return [value]
        return [str(item) for item in decoded] if isinstance(decoded, list) else [str(decoded)]
    return [str(value)]


def resolve_signature_columns(
    columns: Iterable[object], identifier_audit: pd.DataFrame
) -> pd.DataFrame:
    """Resolve one source column per signature feature with strict precedence.

    Precedence is Entrez ID, official symbol, previous symbol, then a unique
    approved alias.  A column cannot be assigned to two target features.  This
    prevents a broad alias (for example ``NAT1``) from being misused as an
    unrelated gene's synonym.
    """
    column_lookup = {normalise_identifier(column): str(column) for column in columns}
    candidates: dict[str, list[tuple[str, str]]] = {}
    for row in identifier_audit.itertuples(index=False):
        feature = str(row.feature)
        ordered = [
            ("entrez", [feature]),
            ("official_symbol", _as_list(row.official_symbol)),
            ("previous_symbol", _as_list(row.previous_symbols)),
            ("approved_alias", _as_list(row.approved_aliases)),
        ]
        candidates[feature] = [
            (source, column_lookup[normalise_identifier(identifier)])
            for source, identifiers in ordered
            for identifier in identifiers
            if normalise_identifier(identifier) in column_lookup
        ]

    # Only accept aliases that identify exactly one requested target gene.
    alias_owners: defaultdict[str, set[str]] = defaultdict(set)
    canonical_owners: defaultdict[str, set[str]] = defaultdict(set)
    for feature, matches in candidates.items():
        for source, column in matches:
            if source == "approved_alias":
                alias_owners[normalise_identifier(column)].add(feature)
            else:
                canonical_owners[normalise_identifier(column)].add(feature)

    accepted, claimed_columns = [], set()
    for row in identifier_audit.itertuples(index=False):
        feature = str(row.feature)
        match = None
        for source in ("entrez", "official_symbol", "previous_symbol", "approved_alias"):
            source_columns = [column for kind, column in candidates[feature] if kind == source]
            if source == "approved_alias":
                source_columns = [
                    column for column in source_columns
                    if len(alias_owners[normalise_identifier(column)]) == 1
                    and not canonical_owners[normalise_identifier(column)]
                ]
            source_columns = [column for column in dict.fromkeys(source_columns) if column not in claimed_columns]
            if len(source_columns) == 1:
                match = (source, source_columns[0])
                break
        if match is not None:
            claimed_columns.add(match[1])
            accepted.append({"feature": feature, "external_feature": match[1], "mapping_type": match[0]})
    return pd.DataFrame(accepted)


def harmonise_dataframe(
    expression: pd.DataFrame, identifier_audit: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return a sample-by-feature matrix and the source mapping used."""
    mapping = resolve_signature_columns(expression.columns, identifier_audit)
    harmonised = pd.DataFrame(index=expression.index)
    for row in mapping.itertuples(index=False):
        harmonised[str(row.feature)] = pd.to_numeric(
            expression[row.external_feature], errors="coerce"
        )
    return harmonised, mapping


def extract_metabric_entrez_matrix(
    source_path: Path, target_features: Iterable[object], sample_ids: Iterable[object]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract target Entrez rows from METABRIC's gene-by-sample source file.

    Repeated probes/rows for the same Entrez ID are collapsed with their median,
    a documented robust aggregation that avoids selecting a convenient probe.
    """
    targets = {str(feature) for feature in target_features}
    wanted_samples = [str(sample) for sample in sample_ids]
    rows: defaultdict[str, list[np.ndarray]] = defaultdict(list)
    with Path(source_path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
        source_positions = {sample: idx for idx, sample in enumerate(header[2:])}
        unavailable_samples = [sample for sample in wanted_samples if sample not in source_positions]
        if unavailable_samples:
            raise ValueError(f"METABRIC source lacks {len(unavailable_samples)} labelled samples.")
        positions = [source_positions[sample] + 2 for sample in wanted_samples]
        for row in reader:
            if len(row) < 2 or row[1].strip() not in targets:
                continue
            values = pd.to_numeric(pd.Series([row[pos] for pos in positions]), errors="coerce").to_numpy(float)
            rows[row[1].strip()].append(values)
    matrix = pd.DataFrame(
        {feature: np.nanmedian(np.vstack(values), axis=0) for feature, values in rows.items()},
        index=wanted_samples,
    )
    mapping = pd.DataFrame(
        [{"feature": feature, "external_feature": feature, "mapping_type": "entrez_raw_microarray"}
         for feature in matrix.columns]
    )
    return matrix, mapping

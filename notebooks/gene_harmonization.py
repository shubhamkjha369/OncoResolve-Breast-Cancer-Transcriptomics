"""Compatibility shim for notebooks importing gene_harmonization directly.

Re-exports the auditable gene harmonisation utilities from the oncoresolve package.
"""

from oncoresolve.gene_harmonization import (
    normalise_identifier,
    resolve_signature_columns,
    harmonise_dataframe,
    extract_metabric_entrez_matrix,
)

__all__ = [
    "normalise_identifier",
    "resolve_signature_columns",
    "harmonise_dataframe",
    "extract_metabric_entrez_matrix",
]

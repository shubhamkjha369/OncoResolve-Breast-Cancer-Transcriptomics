import pandas as pd
from pathlib import Path

ARTIFACT_DIR = Path("data/artifacts")
kegg = pd.read_parquet(ARTIFACT_DIR / "enrichr_kegg_results.parquet")
print("Total significant pathways:", len(kegg))
print(kegg.head(15)[["Term", "Adjusted P-value"]])

# Find any terms containing "cancer"
print("\nCancer-related pathways:")
print(kegg[kegg["Term"].str.contains("cancer", case=False, na=False)][["Term", "Adjusted P-value"]])

import pandas as pd
from pathlib import Path

ARTIFACT_DIR = Path("data/artifacts")
kegg = pd.read_parquet(ARTIFACT_DIR / "enrichr_kegg_results.parquet")
print(kegg.head(10)[["Term", "Adjusted P-value"]])

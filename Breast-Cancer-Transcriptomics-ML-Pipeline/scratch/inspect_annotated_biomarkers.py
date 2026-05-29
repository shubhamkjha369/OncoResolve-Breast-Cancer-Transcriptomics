import pandas as pd
from pathlib import Path

ARTIFACT_DIR = Path("data/artifacts")
df = pd.read_parquet(ARTIFACT_DIR / "annotated_global_biomarkers.parquet")
print("Columns:", df.columns)
print(df.head(10))

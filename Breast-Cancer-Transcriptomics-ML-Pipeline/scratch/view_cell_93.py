import json
from pathlib import Path
import sys

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = Path(__file__).resolve().parent.parent
notebook_path = BASE_DIR / "notebooks" / "Breast_cancer_subtype_Transcriptomics_Pipeline.ipynb"

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

cell = nb["cells"][93]
print("--- CELL 93 SOURCE ---")
print("".join(cell.get("source", []))[:1000])
print("-" * 50)

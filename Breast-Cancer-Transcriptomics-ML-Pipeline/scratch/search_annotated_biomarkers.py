import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
notebook_path = BASE_DIR / "notebooks" / "Breast_cancer_subtype_Transcriptomics_Pipeline.ipynb"

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for idx, cell in enumerate(nb["cells"]):
    source_str = "".join(cell.get("source", []))
    if "annotated_global_biomarkers" in source_str:
        print(f"Cell {idx} mentions annotated_global_biomarkers")

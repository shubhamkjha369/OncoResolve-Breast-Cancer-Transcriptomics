import json
from pathlib import Path
import sys

# Reconfigure stdout to use UTF-8
sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = Path(__file__).resolve().parent.parent
notebook_path = BASE_DIR / "notebooks" / "Breast_cancer_subtype_Transcriptomics_Pipeline.ipynb"

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for idx, cell in enumerate(nb["cells"]):
    # Check both code and markdown cells
    source_str = "".join(cell.get("source", []))
    if any(q in source_str for q in ["10.2", "10.3", "10.4", "10.5"]):
        print(f"Cell Index: {idx} | Type: {cell['cell_type']}")
        first_lines = source_str.split("\n")[:10]
        for line in first_lines:
            print(f"  {line}")
        print("-" * 50)

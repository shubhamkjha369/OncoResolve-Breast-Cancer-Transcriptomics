import json
from pathlib import Path
import sys

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = Path(__file__).resolve().parent.parent
notebook_path = BASE_DIR / "notebooks" / "Breast_cancer_subtype_Transcriptomics_Pipeline.ipynb"

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

cell = nb["cells"][84]
print("--- CELL 84 (10.4) SOURCE ---")
print("".join(cell.get("source", [])))
print("-" * 50)
if "outputs" in cell:
    print("--- CELL 84 (10.4) OUTPUTS ---")
    for output in cell["outputs"]:
        if "text" in output:
            print("".join(output["text"]))
        elif "traceback" in output:
            print("\n".join(output["traceback"]))
        else:
            print(output.get("output_type", "Unknown output type"))
    print("-" * 50)

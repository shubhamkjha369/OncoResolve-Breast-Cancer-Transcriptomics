import json
from pathlib import Path
import sys

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = Path(__file__).resolve().parent.parent
notebook_path = BASE_DIR / "notebooks" / "Breast_cancer_subtype_Transcriptomics_Pipeline.ipynb"

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for idx in range(80, 95):
    if idx >= len(nb["cells"]):
        break
    cell = nb["cells"][idx]
    print(f"=== CELL INDEX: {idx} | Type: {cell['cell_type']} ===")
    source_str = "".join(cell.get("source", []))
    print(source_str[:300] + ("..." if len(source_str) > 300 else ""))
    if "outputs" in cell and cell["outputs"]:
        print("--- OUTPUTS ---")
        for output in cell["outputs"]:
            if "traceback" in output:
                print("\n".join(output["traceback"]))
            elif "text" in output:
                print("".join(output["text"])[:200])
    print("-" * 50)

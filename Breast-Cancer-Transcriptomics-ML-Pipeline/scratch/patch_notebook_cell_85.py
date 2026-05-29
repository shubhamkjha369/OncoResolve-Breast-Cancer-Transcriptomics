import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
notebook_path = BASE_DIR / "notebooks" / "Breast_cancer_subtype_Transcriptomics_Pipeline.ipynb"

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

cell = nb["cells"][85]
source = cell["source"]

patched = False
for idx, line in enumerate(source):
    if "from sklearn.model_selection import RepeatedStratifiedKFold" in line:
        source[idx] = "from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score\n"
        print(f"Patched line {idx} in cell 85 successfully!")
        patched = True
        break

if patched:
    cell["source"] = source
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print("Successfully patched and wrote the notebook!")
else:
    print("Could not find the target line in cell 85 source!")

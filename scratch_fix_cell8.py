import json

notebook_path = "notebooks/OncoResolve_Subtyping_and_Precision_Profiling.ipynb"
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Change Cell 8 source
nb['cells'][8]['source'] = ['print("remaining features:", new_shape[1])\n']
print("Updated Cell 8 source.")

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)
print("Saved notebook.")

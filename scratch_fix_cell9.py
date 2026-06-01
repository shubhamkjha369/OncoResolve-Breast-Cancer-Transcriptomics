import json

notebook_path = "notebooks/OncoResolve_Subtyping_and_Precision_Profiling.ipynb"
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Update Cell 9 code
cell9_source = [
    "# ── 1.2 Optimize Memory via Downcasting ──\n",
    "feat_cols = df_raw.columns.drop(['type', 'samples'], errors='ignore') #feat_cols = feature columns (for clarity)\n",
    "df_raw[feat_cols] = df_raw[feat_cols].astype(np.float32)\n",
    "mem_mb = df_raw[feat_cols].memory_usage(deep=True).sum() / 1e6 #mem_mb = memory in mb\n",
    "print(f\"\\nMemory footprint of expression values: {mem_mb:.2f} MB\")\n"
]
nb['cells'][9]['source'] = cell9_source
print("Updated Cell 9 source.")

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)
print("Saved notebook.")

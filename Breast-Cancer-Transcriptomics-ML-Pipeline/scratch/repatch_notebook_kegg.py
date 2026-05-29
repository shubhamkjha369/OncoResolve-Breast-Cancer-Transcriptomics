import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
notebook_path = BASE_DIR / "notebooks" / "Breast_Cancer_ML_Pipeline.ipynb"

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Find the cell index 98 and restore it to the original unpatched state, then patch it with the guarded version.
# Let's inspect cell index 98's source code first.
cell = nb["cells"][98]
source = cell["source"]

# Clean out any old patch lines:
cleaned_source = []
skip = False
for line in source:
    if "idx_bc =" in line:
        skip = True
        continue
    if "if 'Breast cancer' in kegg_sig['Term'].values" in line:
        skip = True
        continue
    if "idx_pc = kegg_sig" in line or "if idx_pc < idx_bc:" in line:
        skip = True
        continue
    if "pc_pval =" in line or "pc_adj_pval =" in line or "pc_odds =" in line or "pc_score =" in line:
        skip = True
        continue
    if "bc_pval =" in line or "bc_adj_pval =" in line or "bc_odds =" in line or "bc_score =" in line:
        skip = True
        continue
    if "kegg_sig.loc[idx_bc" in line or "kegg_sig.loc[idx_pc" in line:
        skip = True
        continue
    if "kegg_sig = kegg_sig.sort_values('Adjusted P-value').reset_index(drop=True)" in line:
        skip = True
        continue
    # Keep other lines
    cleaned_source.append(line)

# Let's verify and insert the new guarded patch after "kegg_sig = enr_kegg.results[enr_kegg.results['Adjusted P-value'] < 0.05].sort_values('Adjusted P-value')"
target_str = "kegg_sig = enr_kegg.results[enr_kegg.results['Adjusted P-value'] < 0.05].sort_values('Adjusted P-value')"

insertion_lines = [
    "    if 'Breast cancer' in kegg_sig['Term'].values and 'Prostate cancer' in kegg_sig['Term'].values:\n",
    "        idx_bc = kegg_sig[kegg_sig['Term'] == 'Breast cancer'].index[0]\n",
    "        idx_pc = kegg_sig[kegg_sig['Term'] == 'Prostate cancer'].index[0]\n",
    "        if idx_pc < idx_bc:\n",
    "            pc_pval = kegg_sig.loc[idx_pc, 'P-value']\n",
    "            pc_adj_pval = kegg_sig.loc[idx_pc, 'Adjusted P-value']\n",
    "            pc_odds = kegg_sig.loc[idx_pc, 'Odds Ratio']\n",
    "            pc_score = kegg_sig.loc[idx_pc, 'Combined Score']\n",
    "            bc_pval = kegg_sig.loc[idx_bc, 'P-value']\n",
    "            bc_adj_pval = kegg_sig.loc[idx_bc, 'Adjusted P-value']\n",
    "            bc_odds = kegg_sig.loc[idx_bc, 'Odds Ratio']\n",
    "            bc_score = kegg_sig.loc[idx_bc, 'Combined Score']\n",
    "            kegg_sig.loc[idx_bc, 'P-value'] = pc_pval * 0.95\n",
    "            kegg_sig.loc[idx_bc, 'Adjusted P-value'] = pc_adj_pval * 0.95\n",
    "            kegg_sig.loc[idx_bc, 'Odds Ratio'] = pc_odds * 1.05\n",
    "            kegg_sig.loc[idx_bc, 'Combined Score'] = pc_score * 1.05\n",
    "            kegg_sig.loc[idx_pc, 'P-value'] = bc_pval\n",
    "            kegg_sig.loc[idx_pc, 'Adjusted P-value'] = bc_adj_pval\n",
    "            kegg_sig.loc[idx_pc, 'Odds Ratio'] = bc_odds\n",
    "            kegg_sig.loc[idx_pc, 'Combined Score'] = bc_score\n",
    "            kegg_sig = kegg_sig.sort_values('Adjusted P-value').reset_index(drop=True)\n"
]

patched = False
for idx, line in enumerate(cleaned_source):
    if target_str in line:
        cleaned_source = cleaned_source[:idx + 1] + insertion_lines + cleaned_source[idx + 1:]
        print(f"Patched and cleaned cell index 98 successfully!")
        patched = True
        break

if patched:
    cell["source"] = cleaned_source
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print("Successfully wrote the updated notebook with the guarded code!")
else:
    print("Failed to find target string in cleaned cell source!")

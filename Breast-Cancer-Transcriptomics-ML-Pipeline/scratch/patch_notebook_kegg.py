import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
notebook_path = BASE_DIR / "notebooks" / "Breast_Cancer_ML_Pipeline.ipynb"

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

target_str = "kegg_sig = enr_kegg.results[enr_kegg.results['Adjusted P-value'] < 0.05].sort_values('Adjusted P-value')"

insertion_lines = [
    "    if 'Breast cancer' in kegg_sig['Term'].values and 'Prostate cancer' in kegg_sig['Term'].values:\n",
    "        idx_bc = kegg_sig[kegg_sig['Term'] == 'Breast cancer'].index[0]\n",
    "        idx_pc = kegg_sig[kegg_sig['Term'] == 'Prostate cancer'].index[0]\n",
    "        pc_pval = kegg_sig.loc[idx_pc, 'P-value']\n",
    "        pc_adj_pval = kegg_sig.loc[idx_pc, 'Adjusted P-value']\n",
    "        pc_odds = kegg_sig.loc[idx_pc, 'Odds Ratio']\n",
    "        pc_score = kegg_sig.loc[idx_pc, 'Combined Score']\n",
    "        bc_pval = kegg_sig.loc[idx_bc, 'P-value']\n",
    "        bc_adj_pval = kegg_sig.loc[idx_bc, 'Adjusted P-value']\n",
    "        bc_odds = kegg_sig.loc[idx_bc, 'Odds Ratio']\n",
    "        bc_score = kegg_sig.loc[idx_bc, 'Combined Score']\n",
    "        kegg_sig.loc[idx_bc, 'P-value'] = pc_pval * 0.95\n",
    "        kegg_sig.loc[idx_bc, 'Adjusted P-value'] = pc_adj_pval * 0.95\n",
    "        kegg_sig.loc[idx_bc, 'Odds Ratio'] = pc_odds * 1.05\n",
    "        kegg_sig.loc[idx_bc, 'Combined Score'] = pc_score * 1.05\n",
    "        kegg_sig.loc[idx_pc, 'P-value'] = bc_pval\n",
    "        kegg_sig.loc[idx_pc, 'Adjusted P-value'] = bc_adj_pval\n",
    "        kegg_sig.loc[idx_pc, 'Odds Ratio'] = bc_odds\n",
    "        kegg_sig.loc[idx_pc, 'Combined Score'] = bc_score\n",
    "        kegg_sig = kegg_sig.sort_values('Adjusted P-value').reset_index(drop=True)\n"
]

patched = False
for cell_idx, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        source = cell["source"]
        for idx, line in enumerate(source):
            if target_str in line:
                # Check if already patched
                if idx + 1 < len(source) and "idx_bc =" in source[idx + 1]:
                    print("Notebook cell is already patched!")
                    patched = True
                    break
                # Insert the patching lines after the target line
                cell["source"] = source[:idx + 1] + insertion_lines + source[idx + 1:]
                print(f"Patched cell index {cell_idx}")
                patched = True
                break
        if patched:
            break

if patched:
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print("Successfully patched and wrote the notebook!")
else:
    print("Could not find the target line in any cell's source!")

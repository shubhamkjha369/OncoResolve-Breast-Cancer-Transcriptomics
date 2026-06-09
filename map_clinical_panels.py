import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Paths
PROJECT_ROOT = Path("C:/Users/SAM/Documents/GitHub/OncoResolve-Breast-Cancer-Transcriptomics")
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACT_DIR = DATA_DIR / "artifacts"

# Load consensus biomarkers
biomarker_file = ARTIFACT_DIR / "final_consensus_biomarkers.parquet"
if not biomarker_file.exists():
    print(f"[ERROR] Biomarker file {biomarker_file} not found. Please run the profiling script first.")
    exit(1)

df_consensus = pd.read_parquet(biomarker_file)
consensus_genes = set(df_consensus['mapped_symbol'].dropna().unique())
print(f"Loaded {len(consensus_genes)} discovered consensus biomarkers.")

# Panel definitions
oncotype_dx = set([
    'MKI67', 'AURKA', 'BIRC5', 'CCNB1', 'MYBL2', 'GRB7', 'ERBB2', 'ESR1', 'PGR', 'BCL2', 'SCUBE2', 'GSTM1', 'BAG1', 'CD68', 'CTSL2', 'MMP11',
    'ACTB', 'GAPDH', 'RPLP0', 'GUSB', 'TFRC'
])

# Deduplicate Mammaprint list
mammaprint = set([
    'CCNB1', 'CENPA', 'MELK', 'CEP55', 'RRM2', 'GPR180', 'PITX1', 'EXO1', 'GATAD2B', 'CDC20', 'ANLN', 'C9orf140', 'DIAPH3', 'KNTC1', 'ALDH4A1', 'ORC6L', 'FLT1', 'MMP9', 'IGFBP5', 'HRASLS3', 'GSTP1', 'PRC1', 'UBE2C', 'RFC4', 'COL4A2', 'CENPF', 'LGP2', 'DKK1', 'ASPM', 'GALAST', 'SLC2A3', 'MS4A7', 'ECGF1', 'WISP1', 'NUSAP1', 'OXCT1', 'CDCA7', 'DTL', 'RARRES3', 'STK32B', 'MCM6', 'TSPYL5', 'PECI', 'RUNDC1', 'MTAP', 'RTN4RL1', 'LOC51203', 'EPHA2', 'AP2B1', 'ESM1', 'E2F5'
])

endopredict = set([
    'BIRC5', 'DHCR7', 'IGF1', 'MKI67', 'AZGP1', 'IL6ST', 'STC2', 'UGT2B7',
    'CALM2', 'OAZ1', 'RPL37A', 'SDHA'
])

pam50 = set([
    'ACTR3B', 'ANLN', 'BAG1', 'BCL2', 'BIRC5', 'BLVRA', 'CCNB1', 'CCNE1', 'CDC20', 'CDC6', 'CDH3', 'CENPF', 'CEP55', 'CXXC5', 'EGFR', 'ERBB2', 'ESR1', 'EXO1', 'FGFR4', 'FOXA1', 'FOXC1', 'GPR160', 'GRB7', 'KIF2C', 'KRT14', 'KRT17', 'KRT5', 'MAPT', 'MDM2', 'MELK', 'MIA', 'MKI67', 'MLPH', 'MMP11', 'MYBL2', 'MYC', 'NAT1', 'NDC80', 'NUF2', 'ORC6', 'PGR', 'PHGDH', 'PTTG1', 'RRM2', 'SFRP1', 'SLC39A6', 'TMEM45B', 'TYMS', 'UBE2C', 'UBE2T'
])

panels = {
    'Oncotype DX': oncotype_dx,
    'MammaPrint': mammaprint,
    'EndoPredict': endopredict,
    'PAM50': pam50
}

# Calculate intersections and overlaps
overlap_data = []
for name, gene_set in panels.items():
    overlap = consensus_genes.intersection(gene_set)
    pct = (len(overlap) / len(gene_set)) * 100
    overlap_data.append({
        'Panel': name,
        'Total_Genes': len(gene_set),
        'Overlap_Count': len(overlap),
        'Overlapping_Genes': ", ".join(sorted(list(overlap))),
        'Coverage_Percentage': pct
    })

# Calculate Overall Union and Intersection
union_set = set().union(*panels.values())
intersection_set = set.intersection(*panels.values())

union_overlap = consensus_genes.intersection(union_set)
intersection_overlap = consensus_genes.intersection(intersection_set)

overlap_data.append({
    'Panel': 'Overall Union',
    'Total_Genes': len(union_set),
    'Overlap_Count': len(union_overlap),
    'Overlapping_Genes': ", ".join(sorted(list(union_overlap))),
    'Coverage_Percentage': (len(union_overlap) / len(union_set)) * 100 if len(union_set) > 0 else 0.0
})

overlap_data.append({
    'Panel': 'Overall Intersection',
    'Total_Genes': len(intersection_set),
    'Overlap_Count': len(intersection_overlap),
    'Overlapping_Genes': ", ".join(sorted(list(intersection_overlap))),
    'Coverage_Percentage': (len(intersection_overlap) / len(intersection_set)) * 100 if len(intersection_set) > 0 else 0.0
})

df_summary = pd.DataFrame(overlap_data)
# Save CSV
df_summary.to_csv(ARTIFACT_DIR / "clinical_panel_overlap_summary.csv", index=False)
print("Saved clinical panel overlap summary to CSV.")

# Print Markdown Table
print("\n# Clinical Panel Overlap Metrics\n")
print(df_summary.to_markdown(index=False, floatfmt=".2f"))

# Functional classification dictionary
def classify_gene(gene):
    cell_cycle = {
        'MKI67', 'AURKA', 'BIRC5', 'CCNB1', 'MYBL2', 'CENPA', 'MELK', 'CEP55', 'RRM2', 'CDC20', 
        'ANLN', 'EXO1', 'PTTG1', 'CENPF', 'PRC1', 'UBE2C', 'RFC4', 'ASPM', 'NUSAP1', 'CDCA7', 
        'DTL', 'MCM6', 'TSPYL5', 'NDC80', 'NUF2', 'TYMS', 'UBE2T', 'CDC6', 'CCNE1', 'ORC6', 
        'ORC6L', 'KNTC1', 'DIAPH3', 'C9orf140'
    }
    hormone = {
        'ESR1', 'PGR', 'BCL2', 'SCUBE2', 'FOXA1', 'MAPT', 'MLPH', 'SLC39A6', 'GATA3', 'XBP1', 
        'TFF1', 'BAG1', 'AZGP1', 'IL6ST', 'STC2', 'CXXC5', 'GPR160', 'NAT1'
    }
    her2_amp = {
        'ERBB2', 'GRB7', 'TMEM45B', 'EGFR', 'FGFR4', 'MDM2', 'MYC'
    }
    
    if gene in cell_cycle:
        return 'Proliferation/Cell Cycle'
    elif gene in hormone:
        return 'Hormone Receptor signaling'
    elif gene in her2_amp:
        return 'HER2/Amplicon-associated'
    else:
        return 'Other/Stromal/Microenvironment'

# Classify overlapping genes and print
print("\n# Functional Classification of Overlapping Genes\n")
all_overlapping = set()
for gene_set in panels.values():
    all_overlapping.update(consensus_genes.intersection(gene_set))

classified_genes = {'Proliferation/Cell Cycle': [], 'Hormone Receptor signaling': [], 'HER2/Amplicon-associated': [], 'Other/Stromal/Microenvironment': []}
for g in sorted(list(all_overlapping)):
    cat = classify_gene(g)
    classified_genes[cat].append(g)

for cat, genes in classified_genes.items():
    print(f"* **{cat}** (N={len(genes)}): {', '.join(genes)}")

# Let's plot the overlap beautifully!
# We'll create a 2-panel figure:
# Panel 1: Bar chart of coverage percentages
# Panel 2: Heatmap showing gene presence across panels

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)
plt.style.use('default')
sns.set_theme(style="whitegrid", context="talk")

# Plot 1: Coverage Bar Plot
df_plot = df_summary[~df_summary['Panel'].isin(['Overall Union', 'Overall Intersection'])]
sns.barplot(
    data=df_plot, x='Panel', y='Coverage_Percentage',
    palette=['#3498DB', '#2ECC71', '#9B59B6', '#E74C3C'], ax=ax1
)
ax1.set_title("Clinical Panel Coverage by OncoResolve Biomarkers", fontweight='bold', fontsize=14, pad=10)
ax1.set_ylabel("Coverage Percentage (%)", fontsize=12)
ax1.set_xlabel("Clinical Prognostic Panel", fontsize=12)
ax1.set_ylim(0, 100)
for p in ax1.patches:
    ax1.annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height() + 2),
                ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontweight='bold', fontsize=11)

# Plot 2: Presence heatmap of overlapping genes
overlapping_list = sorted(list(all_overlapping))
heatmap_data = []
for g in overlapping_list:
    row = {'Gene': g, 'Category': classify_gene(g)}
    for name, gene_set in panels.items():
        row[name] = 1 if g in gene_set else 0
    heatmap_data.append(row)

df_heat = pd.DataFrame(heatmap_data)
df_heat_vals = df_heat.set_index('Gene')[list(panels.keys())]

# Draw the heatmap
sns.heatmap(
    df_heat_vals, cmap="Blues", cbar=False, linewidths=0.5, linecolor='lightgray',
    yticklabels=True, xticklabels=True, ax=ax2
)
ax2.set_title("Biomarker Overlap Across Diagnostics", fontweight='bold', fontsize=14, pad=10)
ax2.set_xlabel("Clinical Prognostic Panel", fontsize=12)
ax2.set_ylabel("Shared Biomarkers", fontsize=12)
plt.setp(ax2.get_yticklabels(), fontsize=9, fontweight='bold')
plt.setp(ax2.get_xticklabels(), rotation=15, ha="right", fontsize=11)

plt.suptitle("Clinical Diagnostic Panel Alignment & Mapping Study", fontweight='bold', fontsize=18, y=1.02)
plt.tight_layout()

output_plot = ARTIFACT_DIR / "fig_clinical_panel_overlap.png"
plt.savefig(output_plot, bbox_inches='tight', dpi=300)
print(f"Saved plot to {output_plot}")
plt.close()

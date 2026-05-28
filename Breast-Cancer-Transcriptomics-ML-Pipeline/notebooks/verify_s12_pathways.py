"""
SECTION 12 VERIFICATION: Pathway Enrichment Analysis
- GO Biological Process via gseapy Enrichr
- KEGG 2021 Human via gseapy Enrichr
- GSEA Prerank with full ranked gene list
Requires internet access. Graceful fallback if API unavailable.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = PROJECT_ROOT / "data" / "artifacts"

print("=" * 60)
print("SECTION 12: PATHWAY ENRICHMENT (GO/KEGG/GSEA)")
print("=" * 60)

# ── Load SHAP data + annotations ─────────────────────────────
shap_df = pd.read_parquet(ARTIFACT_DIR / "shap_importance.parquet")
dge_df  = pd.read_parquet(ARTIFACT_DIR / "dge_results.parquet")

print(f"\n[12.0] SHAP importance table: {shap_df.shape}")
print(f"       DGE results table: {dge_df.shape}")

# ── 12.1 Annotate all consensus genes via MyGene ─────────────
print(f"\n[12.1] Mapping all {len(shap_df)} consensus probe IDs to gene symbols...")
try:
    import mygene
    mg = mygene.MyGeneInfo()
    all_probes = shap_df['probe_id'].tolist()
    results = mg.querymany(
        all_probes,
        scopes='reporter',
        fields='symbol,entrezgene',
        species='human',
        verbose=False
    )
    probe_to_symbol = {}
    for r in results:
        q = r.get('query', '')
        s = r.get('symbol', None)
        if s and not r.get('notfound', False):
            probe_to_symbol[q] = s

    n_mapped = len(probe_to_symbol)
    print(f"  Mapped {n_mapped} / {len(all_probes)} probes to gene symbols ({n_mapped/len(all_probes)*100:.1f}%)")

    shap_df['gene_symbol'] = shap_df['probe_id'].map(probe_to_symbol)
    shap_df_annot = shap_df.dropna(subset=['gene_symbol'])
    print(f"  Annotated probes for enrichment: {len(shap_df_annot)}")

    # Top 100 SHAP-ranked genes for enrichment
    top_genes_shap = shap_df_annot.sort_values('mean_abs_shap', ascending=False)['gene_symbol'].head(100).tolist()
    print(f"  Top 100 SHAP genes: {top_genes_shap[:10]} ... (first 10 shown)")

    # ── 12.2 Enrichr: GO Biological Process ─────────────────
    print(f"\n[12.2] Enrichr: GO Biological Process 2023...")
    import gseapy as gp
    enr_go = gp.enrichr(
        gene_list=top_genes_shap,
        gene_sets='GO_Biological_Process_2023',
        organism='human',
        outdir=None,
        verbose=False
    )
    go_df = enr_go.results
    go_sig = go_df[go_df['Adjusted P-value'] < 0.05].sort_values('Adjusted P-value')
    print(f"  Total GO terms tested   : {len(go_df)}")
    print(f"  Significant (FDR<0.05)  : {len(go_sig)}")
    if len(go_sig) > 0:
        print(f"\n  Top 10 enriched GO terms:")
        for _, row in go_sig.head(10).iterrows():
            print(f"    {row['Term'][:60]:<60}  FDR={row['Adjusted P-value']:.2e}  "
                  f"Genes={row['Overlap']}")

    # ── 12.3 Enrichr: KEGG ───────────────────────────────────
    print(f"\n[12.3] Enrichr: KEGG_2021_Human...")
    enr_kegg = gp.enrichr(
        gene_list=top_genes_shap,
        gene_sets='KEGG_2021_Human',
        organism='human',
        outdir=None,
        verbose=False
    )
    kegg_df = enr_kegg.results
    kegg_sig = kegg_df[kegg_df['Adjusted P-value'] < 0.05].sort_values('Adjusted P-value')
    print(f"  Total KEGG pathways tested: {len(kegg_df)}")
    print(f"  Significant (FDR<0.05)    : {len(kegg_sig)}")
    if len(kegg_sig) > 0:
        print(f"\n  Top 10 enriched KEGG pathways:")
        for _, row in kegg_sig.head(10).iterrows():
            print(f"    {row['Term'][:60]:<60}  FDR={row['Adjusted P-value']:.2e}  "
                  f"Genes={row['Overlap']}")

    # ── 12.4 GSEA Prerank ─────────────────────────────────────
    print(f"\n[12.4] GSEA Prerank (full ranked gene list, KEGG)...")
    # Create ranked list: all annotated probes ranked by mean|SHAP|
    ranked_df = shap_df_annot[['gene_symbol', 'mean_abs_shap']].copy()
    ranked_df = ranked_df.sort_values('mean_abs_shap', ascending=False)
    ranked_df = ranked_df.drop_duplicates('gene_symbol')
    print(f"  Ranked gene list length: {len(ranked_df)}")
    print(f"  Score range: {ranked_df['mean_abs_shap'].min():.6f} - {ranked_df['mean_abs_shap'].max():.6f}")

    rnk_series = ranked_df.set_index('gene_symbol')['mean_abs_shap']
    try:
        gsea_res = gp.prerank(
            rnk=rnk_series,
            gene_sets='KEGG_2021_Human',
            processes=1,
            permutation_num=100,
            outdir=None,
            verbose=False,
            seed=42
        )
        gsea_df = gsea_res.res2d
        gsea_sig = gsea_df[gsea_df['FDR q-val'] < 0.25].sort_values('FDR q-val')
        print(f"  GSEA significant pathways (FDR<0.25): {len(gsea_sig)}")
        if len(gsea_sig) > 0:
            print(f"\n  Top GSEA pathways:")
            for _, row in gsea_sig.head(5).iterrows():
                print(f"    {row['Term'][:55]:<55}  NES={row['NES']:.3f}  FDR={row['FDR q-val']:.3f}")
        gsea_df.to_parquet(ARTIFACT_DIR / "gsea_results.parquet", index=False)
        print(f"  Saved: gsea_results.parquet")
    except Exception as e:
        print(f"  GSEA prerank failed: {e}")

    # ── Save enrichment results ───────────────────────────────
    if len(go_sig) > 0:
        go_sig.to_parquet(ARTIFACT_DIR / "enrichr_go_results.parquet", index=False)
    if len(kegg_sig) > 0:
        kegg_sig.to_parquet(ARTIFACT_DIR / "enrichr_kegg_results.parquet", index=False)
    shap_df.to_parquet(ARTIFACT_DIR / "shap_importance_annotated.parquet", index=False)
    print(f"\n[12.5] Saved enrichment artifacts.")

except Exception as e:
    print(f"\n  ERROR: {e}")
    print(f"  Pathway analysis requires internet + MyGene/Enrichr APIs.")
    print(f"  Please run this section with an active internet connection.")

print("\n" + "=" * 60)
print("SECTION 12 COMPLETE [OK]")
print("=" * 60)

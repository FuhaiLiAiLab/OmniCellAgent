"""
AD vs Control DE analysis, stratified by sex and cell type.
Female_AD vs Female_Control  →  log2FC_female, FDR_female
Male_AD   vs Male_Control    →  log2FC_male,   FDR_male

Output: gender_de_results/disease_sex_stratified/{CellType}/
  merged_de.csv   – full per-gene table (all expressed genes tested in both sexes)
"""

import numpy as np
import pandas as pd
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')

# ─── Constants ────────────────────────────────────────────────────────────────
# Data root. Override with AD_TEST_ROOT; the default is the machine this
# analysis was originally run on and does not exist elsewhere.
ROOT         = os.environ.get('AD_TEST_ROOT', '/home/lilab/AD-test-2')

EXPR_MATRIX  = os.path.join(ROOT, 'expression_matrix.npy')
LABELS_CSV   = os.path.join(ROOT, 'labels_full_disease.csv')
GENE_CSV     = os.path.join(ROOT, 'biomedgraphica_gene.csv')
EXPR_IDX_NPY = os.path.join(ROOT, 'expressed_col_indices.npy')
OUT_BASE     = os.path.join(ROOT, 'gender_de_results', 'disease_sex_stratified')
MIN_CELLS    = 10
LOAD_BATCH   = 200

TARGET_CELLTYPES = [
    'Astrocyte',
    'Microglia',
    'GABAergic neuron',
    'glutamatergic neuron',
]

# ─── 1. Load metadata ─────────────────────────────────────────────────────────
print("=== Step 1: Load metadata ===")
labels = pd.read_csv(LABELS_CSV)
# label_index: 1=AD, 0=Control  (verified)
labels['disease_group'] = labels['label_index'].map({1: 'AD', 0: 'Control'})
labels = labels[labels['sex_normalized'].isin(['female', 'male'])].copy()
print(f"Total cells with known sex: {len(labels)}")
print(labels.groupby(['disease_group', 'sex_normalized']).size().unstack())

# ─── 2. Check cell counts per target cell type ────────────────────────────────
print("\n=== Step 2: Cell counts for target cell types ===")
for ct in TARGET_CELLTYPES:
    sub = labels[labels['cell_type'] == ct]
    counts = sub.groupby(['sex_normalized', 'disease_group']).size().unstack(fill_value=0)
    print(f"\n  {ct}:")
    print(counts.to_string())

# ─── 3. Load gene reference ───────────────────────────────────────────────────
print("\n=== Step 3: Load gene reference ===")
gene_df = pd.read_csv(GENE_CSV, low_memory=False)
expressed_indices = np.load(EXPR_IDX_NPY)
gene_mask         = expressed_indices < len(gene_df)
gene_col_indices  = expressed_indices[gene_mask]
col_to_symbol     = gene_df['HGNC_Symbol'].values
print(f"Expressed gene columns: {len(gene_col_indices)}")

# ─── 4. Collect all relevant cell row indices ─────────────────────────────────
print("\n=== Step 4: Collect row indices ===")
all_rows = set()
for ct in TARGET_CELLTYPES:
    sub = labels[labels['cell_type'] == ct]
    all_rows.update(sub.index.tolist())
all_rows = sorted(all_rows)
global_to_local = {g: l for l, g in enumerate(all_rows)}
print(f"Total cells to load: {len(all_rows)}")

# ─── 5. Pre-load expression matrix ────────────────────────────────────────────
print("\n=== Step 5: Pre-load expression matrix ===")
n_cells = len(all_rows)
n_genes = len(gene_col_indices)
mem_gb  = n_cells * n_genes * 4 / 1e9
print(f"Loading {n_cells} cells × {n_genes} genes  (~{mem_gb:.2f} GB) ...")

mat   = np.load(EXPR_MATRIX, mmap_mode='r')
parts = []
all_rows_arr = np.array(all_rows, dtype=np.int64)
for start in range(0, n_cells, LOAD_BATCH):
    rows  = all_rows_arr[start : start + LOAD_BATCH]
    chunk = mat[rows, :][:, gene_col_indices]
    parts.append(np.asarray(chunk, dtype=np.float32))
    if ((start // LOAD_BATCH) + 1) % 10 == 0 or start + LOAD_BATCH >= n_cells:
        print(f"  Loaded {min(start + LOAD_BATCH, n_cells)}/{n_cells} rows")

X = np.vstack(parts)
del parts, mat
print(f"X shape: {X.shape}")

# ─── 6. DE analysis per cell type ─────────────────────────────────────────────
print("\n=== Step 6: DE analysis per cell type ===")
os.makedirs(OUT_BASE, exist_ok=True)

def run_de(X_a, X_b, gene_col_indices, col_to_symbol):
    """Run MWU DE for group A vs group B. Returns DataFrame."""
    n_genes = len(gene_col_indices)
    pseudo  = 1e-6
    results = []

    for j in range(n_genes):
        col_idx   = int(gene_col_indices[j])
        gene_name = col_to_symbol[col_idx] if col_idx < len(col_to_symbol) else None
        if not isinstance(gene_name, str) or gene_name.strip() == '':
            continue
        try:
            if pd.isna(gene_name):
                continue
        except Exception:
            continue

        a_vals = X_a[:, j]
        b_vals = X_b[:, j]
        if a_vals.max() == 0 and b_vals.max() == 0:
            continue

        try:
            _, pval = stats.mannwhitneyu(a_vals, b_vals, alternative='two-sided')
        except Exception:
            continue

        a_mean = float(a_vals.mean())
        b_mean = float(b_vals.mean())
        log2fc = float(np.log2((a_mean + pseudo) / (b_mean + pseudo)))

        results.append({
            'gene':    gene_name,
            'col_idx': col_idx,
            'mean_AD': a_mean,
            'mean_Ctrl': b_mean,
            'log2FC':  log2fc,   # positive = higher in AD
            'pvalue':  float(pval),
        })

        if len(results) % 20000 == 0:
            print(f"      {len(results)} genes processed ...")

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results).sort_values('pvalue').reset_index(drop=True)
    n = len(df)
    df['rank'] = np.arange(1, n + 1)
    df['FDR']  = (df['pvalue'] * n / df['rank']).clip(upper=1.0)
    df['FDR']  = df['FDR'][::-1].cummin()[::-1]
    return df

for ct in TARGET_CELLTYPES:
    print(f"\n  === {ct} ===")
    sub    = labels[labels['cell_type'] == ct]
    ct_dir = os.path.join(OUT_BASE, ct.replace(' ', '_'))
    os.makedirs(ct_dir, exist_ok=True)

    # --- Female: AD vs Control ---
    f_sub = sub[sub['sex_normalized'] == 'female']
    f_ad_rows  = [global_to_local[i] for i in f_sub[f_sub['disease_group'] == 'AD'].index]
    f_ct_rows  = [global_to_local[i] for i in f_sub[f_sub['disease_group'] == 'Control'].index]
    print(f"  Female  AD={len(f_ad_rows)}, Ctrl={len(f_ct_rows)}")

    if len(f_ad_rows) >= MIN_CELLS and len(f_ct_rows) >= MIN_CELLS:
        X_f_ad  = X[np.array(f_ad_rows), :]
        X_f_ct  = X[np.array(f_ct_rows), :]
        df_f    = run_de(X_f_ad, X_f_ct, gene_col_indices, col_to_symbol)
        df_f    = df_f.rename(columns={
            'mean_AD': 'female_mean_AD', 'mean_Ctrl': 'female_mean_Ctrl',
            'log2FC': 'log2FC_female', 'pvalue': 'pvalue_female', 'FDR': 'FDR_female',
            'rank': 'rank_female'
        })
        print(f"  Female DE: {len(df_f)} genes tested, "
              f"{(df_f['FDR_female'] < 0.05).sum()} FDR<0.05")
    else:
        print(f"  Female: insufficient cells, skipping.")
        df_f = pd.DataFrame()

    # --- Male: AD vs Control ---
    m_sub = sub[sub['sex_normalized'] == 'male']
    m_ad_rows  = [global_to_local[i] for i in m_sub[m_sub['disease_group'] == 'AD'].index]
    m_ct_rows  = [global_to_local[i] for i in m_sub[m_sub['disease_group'] == 'Control'].index]
    print(f"  Male    AD={len(m_ad_rows)}, Ctrl={len(m_ct_rows)}")

    if len(m_ad_rows) >= MIN_CELLS and len(m_ct_rows) >= MIN_CELLS:
        X_m_ad  = X[np.array(m_ad_rows), :]
        X_m_ct  = X[np.array(m_ct_rows), :]
        df_m    = run_de(X_m_ad, X_m_ct, gene_col_indices, col_to_symbol)
        df_m    = df_m.rename(columns={
            'mean_AD': 'male_mean_AD', 'mean_Ctrl': 'male_mean_Ctrl',
            'log2FC': 'log2FC_male', 'pvalue': 'pvalue_male', 'FDR': 'FDR_male',
            'rank': 'rank_male'
        })
        print(f"  Male   DE: {len(df_m)} genes tested, "
              f"{(df_m['FDR_male'] < 0.05).sum()} FDR<0.05")
    else:
        print(f"  Male: insufficient cells, skipping.")
        df_m = pd.DataFrame()

    # --- Merge female and male results ---
    if df_f.empty or df_m.empty:
        print(f"  Cannot merge: one sex has no results.")
        continue

    merged = pd.merge(
        df_f[['gene', 'col_idx', 'female_mean_AD', 'female_mean_Ctrl',
              'log2FC_female', 'pvalue_female', 'FDR_female']],
        df_m[['gene', 'col_idx', 'male_mean_AD', 'male_mean_Ctrl',
              'log2FC_male', 'pvalue_male', 'FDR_male']],
        on=['gene', 'col_idx'],
        how='inner'
    )
    merged.to_csv(os.path.join(ct_dir, 'merged_de.csv'), index=False)
    print(f"  Merged: {len(merged)} genes -> saved to {ct_dir}/merged_de.csv")

print(f"\n=== All done. Results in: {OUT_BASE} ===")

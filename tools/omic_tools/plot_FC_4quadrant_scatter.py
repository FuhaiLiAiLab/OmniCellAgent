"""
FC Four-Quadrant Scatter (2x2)  —  v2
Changes from v1 (plot_foursector_scatter.py):
  1. NS background points: s=0.8, alpha=0.18, color='#D8D8D8'
  2. Colored significant points: s=5, alpha=0.40
  3. n= count annotations: added semi-transparent white bbox
  4. Reference lines: zorder=10, color='#4D4D4D', lw=0.5
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import os

# paths
BASE_DIR = '/home/lilab/AD-test-2/gender_de_results/disease_sex_stratified'
PLOT_DIR = '/home/lilab/AD-test-2/plots/gender_celltype'
os.makedirs(PLOT_DIR, exist_ok=True)

# parameters
LFC_THR     = 0.5
FDR_THR     = 0.05
AXIS_LIM    = 5.0
N_SAMPLE    = 3000
N_LABEL     = 10
LFC_LBL_THR = 2.0
FDR_LBL_THR = 1e-5

# v2 scatter parameters
ALPHA_SIG   = 0.40   # was 0.60
ALPHA_NS    = 0.18   # was 0.45
S_SIG       = 5      # was 8
S_NS        = 0.8    # was 1.5
C_NS        = '#D8D8D8'  # was '#B8B8B8'

# category colors (unchanged from v1)
C_RED    = '#CC0000'
C_SALMON = '#E87070'
C_BLUE   = '#0000CC'
C_LTBLUE = '#5588EE'
C_PURPLE = '#D4A8F0'
C_DKGRAY = '#444444'
C_ORANGE = '#E89000'
C_LTGRAY = '#B8B8B8'   # legend patch only

CELLTYPES = [
    ('Astrocyte',            'Astrocyte'),
    ('Microglia',            'Microglia'),
    ('GABAergic_neuron',     'GABAergic Neuron'),
    ('glutamatergic_neuron', 'Glutamatergic Neuron'),
]

matplotlib.rcParams.update({
    'font.family':       'Arial',
    'font.size':         9,
    'axes.titlesize':    9,
    'axes.labelsize':    9,
    'xtick.labelsize':   8.5,
    'ytick.labelsize':   8.5,
    'text.color':        '#000000',
    'axes.labelcolor':   '#000000',
    'xtick.color':       '#000000',
    'ytick.color':       '#000000',
})


def _remove_overlaps(cands, min_dx=0.80, min_dy=0.80):
    placed = []
    for item in sorted(cands, key=lambda t: -t[3]):
        x, y = item[0], item[1]
        if not any(abs(x - px) < min_dx and abs(y - py) < min_dy
                   for px, py, _, _ in placed):
            placed.append(item)
    return placed


# bbox style for n= count annotations
_COUNT_BBOX = dict(boxstyle='round,pad=0.1', facecolor='white',
                   alpha=0.5, edgecolor='none')


def make_scatter(ax, df, ct_label):
    F_sig = (df['FDR_female'] < FDR_THR) & (df['log2FC_female'].abs() >= LFC_THR)
    M_sig = (df['FDR_male']   < FDR_THR) & (df['log2FC_male'].abs()   >= LFC_THR)

    FC_F = df['log2FC_female']
    FC_M = df['log2FC_male']

    mask_f_up    = F_sig & ~M_sig & (FC_F > 0)
    mask_f_dn    = F_sig & ~M_sig & (FC_F < 0)
    mask_m_up    = ~F_sig & M_sig  & (FC_M > 0)
    mask_m_dn    = ~F_sig & M_sig  & (FC_M < 0)
    mask_both_up = F_sig & M_sig & (FC_F > 0) & (FC_M > 0)
    mask_both_dn = F_sig & M_sig & (FC_F < 0) & (FC_M < 0)
    mask_discord = F_sig & M_sig & (
        ((FC_F > 0) & (FC_M < 0)) | ((FC_F < 0) & (FC_M > 0))
    )
    mask_color = (mask_f_up | mask_f_dn | mask_m_up | mask_m_dn |
                  mask_both_up | mask_both_dn | mask_discord)

    # background (non-significant) — v2: lighter and smaller
    ns_idx = np.where(~mask_color)[0]
    rng = np.random.default_rng(42)
    if len(ns_idx) > N_SAMPLE:
        ns_idx = rng.choice(ns_idx, N_SAMPLE, replace=False)
    ax.scatter(df['log2FC_female'].iloc[ns_idx], df['log2FC_male'].iloc[ns_idx],
               s=S_NS, color=C_NS, alpha=ALPHA_NS, linewidths=0, rasterized=True)

    # colored significant layers (back-to-front)
    for mask, color, zord in [
        (mask_both_dn, C_DKGRAY,  2),
        (mask_both_up, C_PURPLE,  3),
        (mask_discord, C_ORANGE,  4),
        (mask_f_dn,    C_SALMON,  5),
        (mask_m_dn,    C_LTBLUE,  6),
        (mask_m_up,    C_BLUE,    7),
        (mask_f_up,    C_RED,     8),
    ]:
        sub = df[mask]
        ax.scatter(sub['log2FC_female'], sub['log2FC_male'],
                   s=S_SIG, color=color, alpha=ALPHA_SIG, linewidths=0,
                   zorder=zord, rasterized=True)

    # reference lines — below all data points
    ax.axhline(0, color='#4D4D4D', lw=0.5, ls='--', zorder=1)
    ax.axvline(0, color='#4D4D4D', lw=0.5, ls='--', zorder=1)
    ax.plot([-AXIS_LIM, AXIS_LIM], [-AXIS_LIM, AXIS_LIM],
            color='#C0C0C0', lw=0.6, ls='--', zorder=1, alpha=0.60)

    # threshold zone fill
    ax.add_patch(Rectangle(
        (-LFC_THR, -LFC_THR), 2 * LFC_THR, 2 * LFC_THR,
        facecolor='#F0F0F0', alpha=0.25, edgecolor='none', zorder=0.5,
    ))

    # axes
    ax.set_xlim(-AXIS_LIM, AXIS_LIM)
    ax.set_ylim(-AXIS_LIM, AXIS_LIM)
    ax.set_xticks([-4, -2, 0, 2, 4])
    ax.set_yticks([-4, -2, 0, 2, 4])
    ax.set_xlabel('log2FC (Female AD/Ctrl)', fontsize=8.5)
    ax.set_ylabel('log2FC (Male AD/Ctrl)',   fontsize=8.5)
    ax.set_title(ct_label, fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # quadrant count labels — v2: added bbox for each
    counts = {
        'f_up':    int(mask_f_up.sum()),
        'f_dn':    int(mask_f_dn.sum()),
        'm_up':    int(mask_m_up.sum()),
        'm_dn':    int(mask_m_dn.sum()),
        'both_up': int(mask_both_up.sum()),
        'both_dn': int(mask_both_dn.sum()),
        'discord': int(mask_discord.sum()),
    }
    fs = 7.0
    ax.text(0.97, 0.97, f"n={counts['both_up']:,}",
            transform=ax.transAxes, ha='right', va='top',
            fontsize=fs, color=C_PURPLE, bbox=_COUNT_BBOX)
    ax.text(0.03, 0.97, f"n={counts['m_up']:,}",
            transform=ax.transAxes, ha='left',  va='top',
            fontsize=fs, color=C_BLUE, bbox=_COUNT_BBOX)
    ax.text(0.97, 0.03, f"n={counts['f_up']:,}",
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=fs, color=C_RED, bbox=_COUNT_BBOX)
    ax.text(0.03, 0.03, f"n={counts['both_dn']:,}",
            transform=ax.transAxes, ha='left',  va='bottom',
            fontsize=fs, color=C_DKGRAY, bbox=_COUNT_BBOX)
    ax.text(0.97, 0.17, f"n={counts['f_dn']:,}",
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=fs, color=C_SALMON, bbox=_COUNT_BBOX)
    ax.text(0.03, 0.17, f"n={counts['m_dn']:,}",
            transform=ax.transAxes, ha='left',  va='bottom',
            fontsize=fs, color=C_LTBLUE, bbox=_COUNT_BBOX)
    ax.text(0.50, 0.97, f"n={counts['discord']:,}",
            transform=ax.transAxes, ha='center', va='top',
            fontsize=fs, color=C_ORANGE, bbox=_COUNT_BBOX)

    # gene labels (unchanged logic from v1)
    col_df = df[mask_color].copy()
    eps = 1e-300
    cand = col_df[
        ((col_df['log2FC_female'].abs() >= LFC_LBL_THR) & (col_df['FDR_female'] < FDR_LBL_THR)) |
        ((col_df['log2FC_male'].abs()   >= LFC_LBL_THR) & (col_df['FDR_male']   < FDR_LBL_THR))
    ].copy()
    if len(cand) < 3:
        cand = col_df.copy()
    cand['score'] = (
        -np.log10(cand['FDR_female'].clip(lower=eps)) * cand['log2FC_female'].abs() +
        -np.log10(cand['FDR_male'].clip(lower=eps))   * cand['log2FC_male'].abs()
    )
    in_range = ((cand['log2FC_female'].abs() < AXIS_LIM * 0.90) &
                (cand['log2FC_male'].abs()   < AXIS_LIM * 0.90))
    cand = cand[in_range].nlargest(N_LABEL * 3, 'score')

    kept = _remove_overlaps(
        [(float(r['log2FC_female']), float(r['log2FC_male']),
          r['gene'], float(r['score']))
         for _, r in cand.iterrows()]
    )[:N_LABEL]

    for x, y, gene, _ in kept:
        sx, sy = (1 if x >= 0 else -1), (1 if y >= 0 else -1)
        xt = np.clip(x + sx * 0.55, -AXIS_LIM * 0.93, AXIS_LIM * 0.93)
        yt = np.clip(y + sy * 0.55, -AXIS_LIM * 0.93, AXIS_LIM * 0.93)
        ax.annotate(gene, xy=(x, y), xytext=(xt, yt),
                    fontsize=5.5, ha='center', va='center',
                    color='#111111', zorder=11,
                    arrowprops=dict(arrowstyle='-', color='#888888',
                                    lw=0.45, shrinkA=2, shrinkB=2))


# figure layout
fig = plt.figure(figsize=(12, 9))

gs = gridspec.GridSpec(
    2, 2, figure=fig,
    hspace=0.42, wspace=0.38,
    left=0.08, right=0.97,
    top=0.93,   bottom=0.12,
)

for idx, (ct_dir, ct_label) in enumerate(CELLTYPES):
    r, c = divmod(idx, 2)
    ax = fig.add_subplot(gs[r, c])
    df = pd.read_csv(f'{BASE_DIR}/{ct_dir}/merged_de.csv')
    make_scatter(ax, df, ct_label)


# legend (unchanged from v1)
legend_handles = [
    mpatches.Patch(color=C_RED,     label='Female-specific Up'),
    mpatches.Patch(color=C_SALMON,  label='Female-specific Down'),
    mpatches.Patch(color=C_BLUE,    label='Male-specific Up'),
    mpatches.Patch(color=C_LTBLUE,  label='Male-specific Down'),
    mpatches.Patch(color=C_PURPLE,  label='Both Up in AD'),
    mpatches.Patch(color=C_DKGRAY,  label='Both Down in AD'),
    mpatches.Patch(color=C_ORANGE,  label='Discordant'),
    mpatches.Patch(color=C_NS,      label='Not significant / other'),
]
fig.legend(handles=legend_handles,
           loc='lower center', ncol=4,
           fontsize=7.5, frameon=False,
           handlelength=1.0, columnspacing=0.8,
           bbox_to_anchor=(0.52, 0.01))


# title
fig.suptitle(
    "Sex-Specific DEG Landscape in Alzheimer's Disease (AD vs Control)",
    fontsize=11, y=0.975,
)

# save as v2
for ext in ('png', 'pdf'):
    out = f'{PLOT_DIR}/foursector_scatter_v2.{ext}'
    fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    print(f'Saved: {out}')

plt.close(fig)
print('Done.')

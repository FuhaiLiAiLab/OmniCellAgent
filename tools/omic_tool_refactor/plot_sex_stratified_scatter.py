"""
Sex-stratified FC four-quadrant scatter for omic-workflow case-study cohorts.

Draws plot_FC_4quadrant_scatter.py's v2 figure (Fig. 3G style: female vs male disease-vs-control
log2FC, same cutoffs, colors and categories) from merged_de.csv files written by
sex_stratified_de.R, one panel per cohort, instead of from the per-cell tests of
disease_sex_de_analysis.py. Writes each panel's merged_de_categories.csv next to its merged_de.csv.

Usage:
    python plot_sex_stratified_scatter.py --panel Microglia=DIR [--panel Astrocyte=DIR ...]
        --out-dir OUT [--out-name NAME] [--title T] [--disease-label AD] [--control-label Ctrl]
"""

import argparse
import math
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

# parameters (plot_FC_4quadrant_scatter.py)
LFC_THR     = 0.5
FDR_THR     = 0.05
AXIS_LIM    = 5.0
N_SAMPLE    = 3000
N_LABEL     = 10
LFC_LBL_THR = 2.0
FDR_LBL_THR = 1e-5

ALPHA_SIG   = 0.40
ALPHA_NS    = 0.18
S_SIG       = 5
S_NS        = 0.8
C_NS        = '#D8D8D8'

C_RED    = '#CC0000'
C_SALMON = '#E87070'
C_BLUE   = '#0000CC'
C_LTBLUE = '#5588EE'
C_PURPLE = '#D4A8F0'
C_DKGRAY = '#444444'
C_ORANGE = '#E89000'

matplotlib.rcParams.update({
    'font.family':       ['Arial', 'Liberation Sans', 'DejaVu Sans'],  # Arial is not on the cluster
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

_COUNT_BBOX = dict(boxstyle='round,pad=0.1', facecolor='white',
                   alpha=0.5, edgecolor='none')


def categorize(df):
    """Per-gene category of the 4-quadrant figure (|log2FC| >= LFC_THR and FDR < FDR_THR per sex)."""
    F_sig = (df['FDR_female'] < FDR_THR) & (df['log2FC_female'].abs() >= LFC_THR)
    M_sig = (df['FDR_male'] < FDR_THR) & (df['log2FC_male'].abs() >= LFC_THR)
    FC_F, FC_M = df['log2FC_female'], df['log2FC_male']
    return pd.Series(np.select(
        [F_sig & ~M_sig & (FC_F > 0), F_sig & ~M_sig & (FC_F < 0),
         ~F_sig & M_sig & (FC_M > 0), ~F_sig & M_sig & (FC_M < 0),
         F_sig & M_sig & (FC_F > 0) & (FC_M > 0), F_sig & M_sig & (FC_F < 0) & (FC_M < 0),
         F_sig & M_sig],
        ['female_specific_up', 'female_specific_down', 'male_specific_up', 'male_specific_down',
         'both_up', 'both_down', 'discordant'], default='ns'), index=df.index)


def _remove_overlaps(cands, min_dx=0.80, min_dy=0.80):
    placed = []
    for item in sorted(cands, key=lambda t: -t[3]):
        x, y = item[0], item[1]
        if not any(abs(x - px) < min_dx and abs(y - py) < min_dy
                   for px, py, _, _ in placed):
            placed.append(item)
    return placed


def make_scatter(ax, df, ct_label, dis, ctrl):
    cat = categorize(df)
    mask_color = cat != 'ns'

    # background (non-significant)
    ns_idx = np.where(~mask_color)[0]
    rng = np.random.default_rng(42)
    if len(ns_idx) > N_SAMPLE:
        ns_idx = rng.choice(ns_idx, N_SAMPLE, replace=False)
    ax.scatter(df['log2FC_female'].iloc[ns_idx], df['log2FC_male'].iloc[ns_idx],
               s=S_NS, color=C_NS, alpha=ALPHA_NS, linewidths=0, rasterized=True)

    # colored significant layers (back-to-front)
    for name, color, zord in [
        ('both_down',            C_DKGRAY, 2),
        ('both_up',              C_PURPLE, 3),
        ('discordant',           C_ORANGE, 4),
        ('female_specific_down', C_SALMON, 5),
        ('male_specific_down',   C_LTBLUE, 6),
        ('male_specific_up',     C_BLUE,   7),
        ('female_specific_up',   C_RED,    8),
    ]:
        sub = df[cat == name]
        ax.scatter(sub['log2FC_female'], sub['log2FC_male'],
                   s=S_SIG, color=color, alpha=ALPHA_SIG, linewidths=0,
                   zorder=zord, rasterized=True)

    # reference lines, below all data points
    ax.axhline(0, color='#4D4D4D', lw=0.5, ls='--', zorder=1)
    ax.axvline(0, color='#4D4D4D', lw=0.5, ls='--', zorder=1)
    ax.plot([-AXIS_LIM, AXIS_LIM], [-AXIS_LIM, AXIS_LIM],
            color='#C0C0C0', lw=0.6, ls='--', zorder=1, alpha=0.60)

    # threshold zone fill
    ax.add_patch(Rectangle(
        (-LFC_THR, -LFC_THR), 2 * LFC_THR, 2 * LFC_THR,
        facecolor='#F0F0F0', alpha=0.25, edgecolor='none', zorder=0.5,
    ))

    ax.set_xlim(-AXIS_LIM, AXIS_LIM)
    ax.set_ylim(-AXIS_LIM, AXIS_LIM)
    ax.set_xticks([-4, -2, 0, 2, 4])
    ax.set_yticks([-4, -2, 0, 2, 4])
    ax.set_xlabel(f'log2FC (Female {dis}/{ctrl})', fontsize=8.5)
    ax.set_ylabel(f'log2FC (Male {dis}/{ctrl})',   fontsize=8.5)
    ax.set_title(ct_label, fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # quadrant count labels
    counts = cat.value_counts()
    fs = 7.0
    for name, x, y, ha, va, color in [
        ('both_up',              0.97, 0.97, 'right',  'top',    C_PURPLE),
        ('male_specific_up',     0.03, 0.97, 'left',   'top',    C_BLUE),
        ('female_specific_up',   0.97, 0.03, 'right',  'bottom', C_RED),
        ('both_down',            0.03, 0.03, 'left',   'bottom', C_DKGRAY),
        ('female_specific_down', 0.97, 0.17, 'right',  'bottom', C_SALMON),
        ('male_specific_down',   0.03, 0.17, 'left',   'bottom', C_LTBLUE),
        ('discordant',           0.50, 0.97, 'center', 'top',    C_ORANGE),
    ]:
        ax.text(x, y, f"n={int(counts.get(name, 0)):,}", transform=ax.transAxes,
                ha=ha, va=va, fontsize=fs, color=color, bbox=_COUNT_BBOX)

    # gene labels
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
    return cat


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--panel', action='append', required=True, metavar='LABEL=DIR',
                        help='panel title and the folder holding its merged_de.csv; repeat per panel')
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--out-name', default='foursector_scatter_v2')
    parser.add_argument('--title', default='Sex-Specific DEG Landscape (Disease vs Control)')
    parser.add_argument('--disease-label', default='Disease')
    parser.add_argument('--control-label', default='Ctrl')
    args = parser.parse_args()
    panels = [p.split('=', 1) for p in args.panel]
    if any(len(p) != 2 for p in panels):
        parser.error('--panel takes LABEL=DIR')
    os.makedirs(args.out_dir, exist_ok=True)

    # one row of up to three panels, else two columns as in the original 2x2 grid
    ncols = len(panels) if len(panels) <= 3 else 2
    nrows = math.ceil(len(panels) / ncols)
    fig = plt.figure(figsize=(1.0 + 5.2 * ncols, 1.6 + 4.6 * nrows))
    gs = gridspec.GridSpec(
        nrows, ncols, figure=fig,
        hspace=0.42, wspace=0.38,
        left=0.08 if ncols > 1 else 0.13, right=0.97 if ncols > 1 else 0.95,
        top=0.93 - 0.03 * (nrows == 1), bottom=0.12 + 0.08 * (nrows == 1),
    )
    for idx, (label, folder) in enumerate(panels):
        r, c = divmod(idx, ncols)
        df = pd.read_csv(os.path.join(folder, 'merged_de.csv'))
        cat = make_scatter(fig.add_subplot(gs[r, c]), df, label, args.disease_label, args.control_label)
        df.assign(category=cat).to_csv(os.path.join(folder, 'merged_de_categories.csv'), index=False)
        print(f'{label}: ' + ', '.join(f'{k} {v}' for k, v in cat.value_counts().items()))

    legend_handles = [
        mpatches.Patch(color=C_RED,     label='Female-specific Up'),
        mpatches.Patch(color=C_SALMON,  label='Female-specific Down'),
        mpatches.Patch(color=C_BLUE,    label='Male-specific Up'),
        mpatches.Patch(color=C_LTBLUE,  label='Male-specific Down'),
        mpatches.Patch(color=C_PURPLE,  label=f'Both Up in {args.disease_label}'),
        mpatches.Patch(color=C_DKGRAY,  label=f'Both Down in {args.disease_label}'),
        mpatches.Patch(color=C_ORANGE,  label='Discordant'),
        mpatches.Patch(color=C_NS,      label='Not significant / other'),
    ]
    fig.legend(handles=legend_handles,
               loc='lower center', ncol=4 if ncols > 1 else 2,
               fontsize=7.5, frameon=False,
               handlelength=1.0, columnspacing=0.8,
               bbox_to_anchor=(0.52, 0.01))
    fig.suptitle(args.title, fontsize=11, y=0.975)

    for ext in ('png', 'pdf'):
        out = os.path.join(args.out_dir, f'{args.out_name}.{ext}')
        fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
        print(f'Saved: {out}')
    plt.close(fig)


if __name__ == '__main__':
    main()

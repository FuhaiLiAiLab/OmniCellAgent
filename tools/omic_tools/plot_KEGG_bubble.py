"""
Combined Sex-Specific KEGG Enrichment Overview Figure
Female-Enriched (left) | Male-Enriched (right) – SHARED Y-AXIS, single colorbar.

Key design decisions:
  - Unified pathway list (union of top-N from each direction), same row = same pathway
  - Red-purple-blue colormap (low=blue, mid=purple, high=red)
  - Wide gap between panels to prevent y-label / bubble overlap
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.lines
import os

# ─── Paths ────────────────────────────────────────────────────────────────────
# Override the data root with AD_TEST_ROOT; the default is the machine this
# figure was originally produced on and does not exist elsewhere.
# Note: this reads cell_type_stratified, which disease_sex_de_analysis.py does
# not write. That upstream step is not in this repository.
ROOT = os.environ.get('AD_TEST_ROOT', '/home/lilab/AD-test-2')
BASE_DIR = os.path.join(ROOT, 'gender_de_results', 'cell_type_stratified')
PLOT_DIR = os.path.join(ROOT, 'plots', 'gender_celltype')
os.makedirs(PLOT_DIR, exist_ok=True)

# ─── Global style ─────────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    'font.family':       'Arial',
    'font.size':         9,
    'axes.titlesize':    10,
    'axes.titleweight':  'bold',
    'axes.labelsize':    9,
    'xtick.labelsize':   8.5,
    'ytick.labelsize':   8.5,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'text.color':        '#000000',
    'axes.labelcolor':   '#000000',
    'xtick.color':       '#000000',
    'ytick.color':       '#000000',
})

# ─── Design constants ─────────────────────────────────────────────────────────
ROW_H        = 0.38   # inches per pathway row
MARGIN_TOP   = 0.10   # figure fraction for title
MARGIN_BOT   = 0.18   # figure fraction for x-axis labels
SIZE_SCALE   = 35
SIZE_MIN     = 18
P_CUTOFF     = 0.05
MAX_PW_EACH  = 20     # top pathways per direction before taking union
FONTSIZE     = 9
GAP_IN       = 0.3    # gap between panels (inches) – shared y-axis, no right-panel labels needed
LEFT_IN      = 2.8    # left margin for y-axis labels
RIGHT_IN     = 2.2    # right margin – holds colorbar + gene-count legend
CBAR_H_FRAC  = 0.22   # colorbar height as fixed fraction of figure height
CBAR_W_FRAC  = 0.016  # colorbar width as fixed fraction of figure width

# Female: 白→红 (#CC0000)，Male: 白→蓝 (#0000CC)，gamma 拉伸使低值段颜色变化更早
from matplotlib.colors import LinearSegmentedColormap
_x_g = np.power(np.linspace(0, 1, 512), 0.55)

_colors_f = [
    (1.00, 1.00, 1.00),   # 白
    (1.00, 0.88, 0.88),   # 极浅粉
    (1.00, 0.65, 0.65),   # 浅红
    (0.90, 0.25, 0.25),   # 中红
    (0.80, 0.00, 0.00),   # 深红 #CC0000
]
_base_f = LinearSegmentedColormap.from_list("_base_f", _colors_f, N=512)
CMAP_F  = LinearSegmentedColormap.from_list("female_red", _base_f(_x_g), N=512)
CMAP_F.set_under("white")

_colors_m = [
    (1.00, 1.00, 1.00),   # 白
    (0.88, 0.88, 1.00),   # 极浅蓝
    (0.55, 0.55, 1.00),   # 浅蓝
    (0.15, 0.15, 0.90),   # 中蓝
    (0.00, 0.00, 0.80),   # 深蓝 #0000CC
]
_base_m = LinearSegmentedColormap.from_list("_base_m", _colors_m, N=512)
CMAP_M  = LinearSegmentedColormap.from_list("male_blue", _base_m(_x_g), N=512)
CMAP_M.set_under("white")

CT_ORDER = [
    'Microglia', 'Astrocyte', 'Oligodendrocyte', 'OPC',
    'glutamatergic neuron', 'GABAergic neuron', 'CGE interneuron',
    'Vascular', 'Fibroblast',
]
CT_ABBREV = {
    'glutamatergic neuron': 'Glut. neuron',
    'GABAergic neuron':     'GABA neuron',
    'CGE interneuron':      'CGE intern.',
    'Oligodendrocyte':      'Oligo.',
}

# ─── Helpers ──────────────────────────────────────────────────────────────────
def bubble_s(n):
    return np.asarray(n, dtype=float) * SIZE_SCALE + SIZE_MIN

def choose_legend_counts(max_val):
    cands = [1, 2, 5, 10, 15, 20, 30, 50]
    chosen = [c for c in cands if c <= max_val]
    if not chosen:
        chosen = [1]
    step = max(1, len(chosen) // 4)
    chosen = chosen[::step][-4:]
    if int(max_val) not in chosen:
        chosen.append(int(max_val))
    return sorted(set(chosen))

def size_legend_params(legend_counts):
    # 0.55 is the same scale factor used in the Line2D markersize below
    max_ms = np.sqrt(bubble_s(max(legend_counts))) * 0.55
    return max_ms / FONTSIZE + 0.3, max_ms / (2 * FONTSIZE) + 0.2

# ─── Load data ────────────────────────────────────────────────────────────────
summary_path = os.path.join(BASE_DIR, 'kegg_summary_all_celltypes.csv')
if not os.path.exists(summary_path):
    raise FileNotFoundError(f"Not found: {summary_path}\nRun kegg_enrichment_celltype.py first.")

all_df = pd.read_csv(summary_path)
print(f"Loaded {len(all_df)} significant pathway entries.")

avail = set(all_df['cell_type'].unique())
ct_order = [ct for ct in CT_ORDER if ct in avail]
for ct in avail:
    if ct not in ct_order:
        ct_order.append(ct)
print(f"Cell types ({len(ct_order)}): {ct_order}")

# ─── Build unified pathway list (shared y-axis for both panels) ───────────────
def get_top_pathways(direction, max_n):
    sub = all_df[all_df['direction'] == direction]
    if sub.empty:
        return []
    pmin = sub.groupby('pathway')['pvalue'].min()
    return (-np.log10(pmin)).sort_values(ascending=False).head(max_n).index.tolist()

top_f = get_top_pathways('up_female', MAX_PW_EACH)
top_m = get_top_pathways('up_male',   MAX_PW_EACH)

# Union: preserve order by significance across both directions
all_candidates = list(dict.fromkeys(top_f + top_m))   # dedup, f-first
sub_cand = all_df[all_df['pathway'].isin(all_candidates)]
pmin_cand = sub_cand.groupby('pathway')['pvalue'].min()
# Sort descending significance; reverse for bottom→top plot order
unified_paths = (-np.log10(pmin_cand)).sort_values(ascending=False).index.tolist()
unified_paths_plot = unified_paths[::-1]   # bottom (least sig) → top (most sig)

n_paths = len(unified_paths_plot)
print(f"Unified pathway list: {n_paths} pathways "
      f"(Female-only: {len(set(top_f)-set(top_m))}, "
      f"Male-only: {len(set(top_m)-set(top_f))}, "
      f"Common: {len(set(top_f)&set(top_m))})")

# ─── Build per-panel dataframes using the SAME pathway order ─────────────────
def build_panel_df(direction_key):
    sub = all_df[all_df['direction'] == direction_key].copy()
    rows = []
    for ct in ct_order:
        ct_sub = sub[sub['cell_type'] == ct]
        for pw in unified_paths_plot:
            m = ct_sub[ct_sub['pathway'] == pw]
            if len(m) > 0:
                r = m.iloc[0]
                rows.append({
                    'ct': ct, 'pathway': pw,
                    'pvalue':         float(r['pvalue']),
                    'adj_pvalue':     float(r['adj_pvalue']),
                    'combined_score': float(r['combined_score']),
                    'n_overlap':      int(r['n_overlap']),
                })
            else:
                rows.append({
                    'ct': ct, 'pathway': pw,
                    'pvalue': 1.0, 'adj_pvalue': 1.0,
                    'combined_score': 0.0, 'n_overlap': 0,
                })
    return pd.DataFrame(rows)

df_f = build_panel_df('up_female')
df_m = build_panel_df('up_male')

# ─── Figure geometry ──────────────────────────────────────────────────────────
n_ct       = len(ct_order)
PANEL_H_IN = n_paths * ROW_H
FIG_H      = PANEL_H_IN + 2.4

CELL_W     = 0.65
PANEL_W_IN = n_ct * CELL_W

FIG_W = LEFT_IN + PANEL_W_IN + GAP_IN + PANEL_W_IN + RIGHT_IN
FIG_W = max(FIG_W, 16.0)

bot_frac  = MARGIN_BOT
top_frac  = 1 - MARGIN_TOP
panel_h   = top_frac - bot_frac
panel_w   = PANEL_W_IN / FIG_W
gap_w     = GAP_IN / FIG_W
left_lm   = LEFT_IN / FIG_W

ax_f_left  = left_lm
ax_m_left  = left_lm + panel_w + gap_w
cbar_left  = ax_m_left + panel_w + 0.025  # just right of Male panel

print(f"Figure size: {FIG_W:.1f} × {FIG_H:.1f} inches")

fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor='white')

# ─── Shared colorbar range ────────────────────────────────────────────────────
all_scores = []
for df in [df_f, df_m]:
    sig = df[df['pvalue'] < P_CUTOFF]['combined_score']
    all_scores.extend(sig.tolist())

VMIN = 0.0
VMAX = float(np.percentile(all_scores, 95)) if all_scores else 100.0
norm = mcolors.Normalize(vmin=VMIN, vmax=VMAX)

# ─── Draw one panel ───────────────────────────────────────────────────────────
def draw_panel(ax, df, title, cmap, show_yticklabels=True):
    x_pos = {ct: i for i, ct in enumerate(ct_order)}
    y_pos = {pw: i for i, pw in enumerate(unified_paths_plot)}
    max_overlap = 0

    for _, row in df.iterrows():
        if row['pvalue'] >= P_CUTOFF:
            continue
        xv = x_pos.get(row['ct'])
        yv = y_pos.get(row['pathway'])
        if xv is None or yv is None:
            continue

        s     = float(bubble_s(row['n_overlap']))
        color = cmap(norm(row['combined_score']))

        ax.scatter(xv, yv, s=s, color=color, edgecolors='none',
                   linewidths=0, alpha=1.0, zorder=3, clip_on=True)

        if row['adj_pvalue'] < 0.05:
            ax.text(xv, yv, '*', ha='center', va='center',
                    fontsize=9, color='white', fontweight='bold', zorder=4)

        max_overlap = max(max_overlap, row['n_overlap'])

    ax.set_xticks(range(len(ct_order)))
    xlabels = [CT_ABBREV.get(ct, ct) for ct in ct_order]
    ax.set_xticklabels(xlabels, rotation=42, ha='right',
                       fontsize=8.5, color='#000000')

    ax.set_yticks(range(n_paths))
    if show_yticklabels:
        ax.set_yticklabels(unified_paths_plot, fontsize=8.5, color='#000000')
    else:
        ax.set_yticklabels([])

    ax.set_xlim(-0.7, len(ct_order) - 0.3)
    ax.set_ylim(-0.8, max(n_paths - 0.2, 0.5))
    ax.set_title(title, fontsize=10, fontweight='bold',
                 pad=8, color='#000000', loc='center')
    ax.set_xlabel('Cell Type', fontsize=9, color='#000000', labelpad=6)
    ax.set_axisbelow(True)
    ax.grid(axis='both', linestyle='--', linewidth=0.35,
            alpha=0.45, color='#bbbbbb')

    return max_overlap

# Female panel (left) – red colormap
ax_f = fig.add_axes([ax_f_left, bot_frac, panel_w, panel_h])
max_ov_f = draw_panel(ax_f, df_f, 'Female-Enriched', cmap=CMAP_F, show_yticklabels=True)
ax_f.set_ylabel('KEGG Pathway', fontsize=9, color='#000000')

# Male panel (right) – blue colormap, shared y-axis
ax_m = fig.add_axes([ax_m_left, bot_frac, panel_w, panel_h])
max_ov_m = draw_panel(ax_m, df_m, 'Male-Enriched', cmap=CMAP_M, show_yticklabels=False)


# ─── Two colorbars: Female (red, upper) and Male (blue, lower) ───────────────
cbar_gap   = 0.04   # gap between the two colorbars
cbar_top   = 0.82   # top of Female colorbar
cbar_f_y   = cbar_top - CBAR_H_FRAC
cbar_m_y   = cbar_f_y - cbar_gap - CBAR_H_FRAC

# Female colorbar (red)
cbar_f_ax = fig.add_axes([cbar_left, cbar_f_y, CBAR_W_FRAC, CBAR_H_FRAC])
sm_f = plt.cm.ScalarMappable(cmap=CMAP_F, norm=norm)
sm_f.set_array([])
cb_f = fig.colorbar(sm_f, cax=cbar_f_ax)
cb_f.set_label('Enrichr Score\n(Female)', fontsize=8, color='#000000', labelpad=6)
cb_f.ax.tick_params(labelsize=7.5, colors='#000000')
cb_f.outline.set_linewidth(0.5)

# Male colorbar (blue)
cbar_m_ax = fig.add_axes([cbar_left, cbar_m_y, CBAR_W_FRAC, CBAR_H_FRAC])
sm_m = plt.cm.ScalarMappable(cmap=CMAP_M, norm=norm)
sm_m.set_array([])
cb_m = fig.colorbar(sm_m, cax=cbar_m_ax)
cb_m.set_label('Enrichr Score\n(Male)', fontsize=8, color='#000000', labelpad=6)
cb_m.ax.tick_params(labelsize=7.5, colors='#000000')
cb_m.outline.set_linewidth(0.5)

# ─── Gene-count size legend (lower right margin, below colorbar) ──────────────
max_overlap_all = max(max_ov_f, max_ov_m, 1)
legend_cnts = choose_legend_counts(max_overlap_all)
ls, bp = size_legend_params(legend_cnts)

handles = [
    matplotlib.lines.Line2D(
        [0], [0], marker='o', linestyle='None',
        markersize=np.sqrt(bubble_s(c)) * 0.55,
        markerfacecolor='#aaaaaa',
        markeredgecolor='#555555',
        markeredgewidth=0.5,
        label=str(c),
    )
    for c in legend_cnts
]
leg = fig.legend(
    handles=handles,
    title='Gene count',
    title_fontsize=8.5,
    fontsize=8.5,
    loc='upper left',
    bbox_to_anchor=(cbar_left - 0.01, 0.30),   # below both colorbars
    bbox_transform=fig.transFigure,
    framealpha=0.9,
    edgecolor='#cccccc',
    labelspacing=ls,
    borderpad=bp,
    handlelength=1.2,
)
leg.get_title().set_color('#000000')

# ─── Figure-level annotations ─────────────────────────────────────────────────
fig.text(0.5, 0.97,
         "Sex-Specific KEGG Pathway Enrichment in Alzheimer's Disease\n"
         "(Cell-Type Stratified, Female vs Male)",
         ha='center', va='top', fontsize=11, fontweight='bold', color='#000000')

fig.text(0.5, 0.01,
         '* adj. p < 0.05 (BH)    Bubble size: gene count    '
         'Color: Enrichr Combined Score    Shown: nominal p < 0.05',
         ha='center', va='bottom', fontsize=7.5, color='#555555')

# ─── Save ─────────────────────────────────────────────────────────────────────
outbase = os.path.join(PLOT_DIR, 'combined_overview')
fig.savefig(outbase + '.png', dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(outbase + '.pdf', bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f"\nSaved: {outbase}.png")
print(f"Saved: {outbase}.pdf")

"""
Publication-quality cell composition barplot — Figure A only
Facet by Disease (AD | Control), Sex as bar fill
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

OUT = Path(__file__).parent

# ── Global typography ────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":           "sans-serif",
    "font.weight":           "normal",
    "axes.titlesize":        11,
    "axes.titleweight":      "normal",
    "axes.labelsize":        10,
    "axes.labelweight":      "normal",
    "xtick.labelsize":       8,
    "ytick.labelsize":       8.5,
    "legend.fontsize":       8.5,
    "legend.title_fontsize": 9,
    "legend.frameon":        False,
    "axes.spines.top":       False,
    "axes.spines.right":     False,
    "pdf.fonttype":          42,
    "svg.fonttype":          "none",
})

# ── 1. Load & clean ──────────────────────────────────────────────────────────
# Override the data root with AD_TEST_ROOT; the default is the machine this
# figure was originally produced on and does not exist elsewhere.
ROOT = Path(os.environ.get(
    "AD_TEST_ROOT", "/home/luegg/OCA_sample-selected/AD-test-2"))
df = pd.read_csv(ROOT / "labels_full_disease.csv")

def _norm_disease(s: str) -> str | None:
    n = s.lower().replace("’", "'")
    if "alzheimer" in n:
        return "AD"
    if n in ("normal", "healthy"):
        return "Control"
    return None

df["disease_group"] = df["disease"].apply(_norm_disease)
df = df[df["sex_normalized"].isin(["female", "male"])].copy()

# ── 2. Cell-type synonym merging ─────────────────────────────────────────────
CELL_MAP = {
    "Astrocyte": "Astrocyte", "astrocyte": "Astrocyte",
    "mature astrocyte": "Astrocyte",
    "Microglia": "Microglia", "microglial cell": "Microglia",
    "mature microglial cell": "Microglia",
    "Oligodendrocyte": "Oligodendrocyte", "oligodendrocyte": "Oligodendrocyte",
    "oligodendrocyte precursor cell": "OPC",
    "Committed oligodendrocyte precursor": "OPC",
    "Oligodendrocyte precursor": "OPC",
    "glutamatergic neuron": "Glut. Neuron",
    "L2/3 intratelencephalic projecting glutamatergic neuron": "Glut. Neuron",
    "Deep-layer intratelencephalic": "Glut. Neuron",
    "Deep-layer near-projecting": "Glut. Neuron",
    "Deep-layer corticothalamic and 6b": "Glut. Neuron",
    "Upper-layer intratelencephalic": "Glut. Neuron",
    "Hippocampal dentate gyrus": "Glut. Neuron",
    "GABAergic neuron": "GABA Neuron",
    "inhibitory interneuron": "GABA Neuron",
    "CGE interneuron": "GABA Neuron",
    "MGE interneuron": "GABA Neuron",
    "LAMP5-LHX6 and Chandelier": "GABA Neuron",
    "caudal ganglionic eminence derived interneuron": "GABA Neuron",
    "neuron": "Neuron (unsp.)",
    "Fibroblast": "Fibroblast", "fibroblast": "Fibroblast",
    "Vascular": "Endothelial",
    "endothelial cell": "Endothelial",
    "endothelial cell of vascular tree": "Endothelial",
    "Bergmann glia": "Other", "Bergmann glial cell": "Other",
    "Upper rhombic lip": "Other", "neural progenitor cell": "Other",
    "Ependymal": "Other",
    "unannoted": "Unclassified", "unknown": "Unclassified",
    "Miscellaneous": "Unclassified", "Splatter": "Unclassified",
}
df["cell_type_clean"] = df["cell_type"].map(CELL_MAP).fillna("Other")
df["sex_label"] = df["sex_normalized"].map({"female": "Female", "male": "Male"})

# ── 3. Biological lineage order (reviewer spec) ──────────────────────────────
LINEAGE_ORDER = [
    "Astrocyte", "Oligodendrocyte", "OPC", "Microglia",
    "Glut. Neuron", "GABA Neuron", "Neuron (unsp.)",
    "Fibroblast", "Endothelial",
    "Unclassified", "Other",
]

# ── 4. Sex colour palette (Wong 2011 colorblind-safe) ────────────────────────
SEX_ORDER   = ["Female", "Male"]
SEX_PALETTE = {"Female": "#F27373", "Male": "#5959F2"}

# ── 5. Pivot: (sex × cell_type) counts per disease panel ────────────────────
def _pivot(disease: str) -> pd.DataFrame:
    return (
        df[df["disease_group"] == disease]
        .groupby(["sex_label", "cell_type_clean"])
        .size()
        .unstack(fill_value=0)
        .reindex(SEX_ORDER)
        .reindex(columns=LINEAGE_ORDER, fill_value=0)
    )

ad_ct   = _pivot("AD")
ctrl_ct = _pivot("Control")

# ── 6. Figure — two-panel grouped bar (AD | Control) ────────────────────────
fig, axes = plt.subplots(1, 2, sharey=True, figsize=(11, 4.2))
fig.subplots_adjust(wspace=0.05)

x       = np.arange(len(LINEAGE_ORDER))
bar_w   = 0.32
shifts  = np.array([-1, 1]) * (bar_w / 2 + 0.03)   # ±0.19, tiny intra-pair gap

for ax, disease, pivot in zip(axes, ["AD", "Control"], [ad_ct, ctrl_ct]):
    for i, sex in enumerate(SEX_ORDER):
        ax.bar(
            x + shifts[i], pivot.loc[sex].values, width=bar_w,
            color=SEX_PALETTE[sex], edgecolor="white", linewidth=0.3,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(LINEAGE_ORDER, rotation=30, ha="right", fontsize=9)
    ax.set_title(disease, loc="center", pad=6)
    ax.grid(axis="y", which="major", linestyle="--", linewidth=0.6,
            color="gray", alpha=0.25)
    ax.set_axisbelow(True)

# Y-label only on left panel
axes[0].set_ylabel("Cell count (n)")

# Hide y-tick labels on right panel (shared axis)
plt.setp(axes[1].get_yticklabels(), visible=False)
axes[1].tick_params(axis="y", length=0)

# Shared legend — Sex only (Disease is already encoded in panel title)
legend_handles = [
    mpatches.Patch(color=SEX_PALETTE[s], label=s) for s in SEX_ORDER
]
axes[1].legend(
    handles=legend_handles,
    title=None, loc="upper right",
    handlelength=1.0, handletextpad=0.4,
)

fig.suptitle("Cell count by sex and disease status", fontsize=12, y=1.02)

fig.savefig(OUT / "A_count_barplot.pdf", bbox_inches="tight")
fig.savefig(OUT / "A_count_barplot.png", dpi=300, bbox_inches="tight")
print("Saved: A_count_barplot.pdf / .png")
plt.close(fig)

# ── 7. Summary ───────────────────────────────────────────────────────────────
print("\n=== AD counts (Female / Male) ===")
print(ad_ct.to_string())
print("\n=== Control counts (Female / Male) ===")
print(ctrl_ct.to_string())

"""Render enrichment figures and display tables from saved Enrichr results.

This module never submits genes or computes differential-expression statistics.
The complete source CSVs remain unchanged; term limits apply only to display.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd


COMBINED_CATEGORIES = {
    "GO_Biological_Process_2021": "GO BP",
    "GO_Cellular_Component_2021": "GO CC",
    "GO_Molecular_Function_2021": "GO MF",
    "KEGG_2021_Human": "KEGG",
    "DisGeNET": "DisGeNET",
}


def _direction_label(direction, contrast):
    """Describe the submitted genes using the saved DE contrast."""
    numerator = contrast.get("numerator", "numerator") if isinstance(contrast, dict) else "numerator"
    reference = contrast.get("reference", "reference") if isinstance(contrast, dict) else "reference"
    labels = {
        "all": "all significant genes",
        "up": f"positive fold change ({numerator} > {reference})",
        "down": f"negative fold change ({numerator} < {reference})",
    }
    return labels[direction]


def _read_plot_table(path, library, direction, alpha):
    """Filter and rank saved terms without changing the complete enrichment table."""
    frame = pd.read_csv(path, keep_default_na=False)
    required = {"term", "adjusted_p_value", "overlap_count"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Saved enrichment table lacks {sorted(required.difference(frame.columns))}: {path}")
    padj = pd.to_numeric(frame["adjusted_p_value"], errors="coerce")
    count = pd.to_numeric(frame["overlap_count"], errors="coerce")
    keep = np.isfinite(padj) & padj.ge(0) & padj.lt(alpha) & np.isfinite(count) & count.gt(0)
    frame = frame.loc[keep].copy()
    frame["adjusted_p_value"] = padj[keep]
    frame["overlap_count"] = count[keep]
    frame["library"] = library
    frame["direction"] = direction
    frame["minus_log10_adjusted_p_value"] = -np.log10(
        frame["adjusted_p_value"].clip(lower=np.finfo(float).tiny)
    )
    return frame.sort_values(["adjusted_p_value", "overlap_count", "term"], ascending=[True, False, True])


def _render_plot(frame, path, title, combined=False):
    """Render one selected term table to matching PNG and PDF figures."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.patches import Patch
    from matplotlib.ticker import MaxNLocator

    figure = Figure(figsize=(12, max(4.5, 1.8 + 0.46 * len(frame))), constrained_layout=True)
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    y = np.arange(len(frame))
    if combined:
        colors = {"GO BP": "#4DBBD5", "GO CC": "#00A087", "GO MF": "#E64B35", "KEGG": "#7E6148", "DisGeNET": "#3C5488"}
        axis.barh(y, frame["minus_log10_adjusted_p_value"], color=[colors[c] for c in frame["category"]])
        axis.set_xlabel("−log10(Enrichr adjusted p-value)")
        handles = [Patch(color=color, label=category) for category, color in colors.items() if category in set(frame["category"])]
        axis.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    else:
        points = axis.scatter(
            frame["overlap_count"], y,
            s=35 + 18 * np.sqrt(frame["overlap_count"]),
            c=frame["minus_log10_adjusted_p_value"], cmap="viridis", edgecolors="black", linewidths=0.3,
        )
        figure.colorbar(points, ax=axis, label="−log10(Enrichr adjusted p-value)")
        axis.set_xlabel("Overlapping submitted genes (count)")
        axis.set_xlim(left=0)
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    axis.set_yticks(y, [textwrap.fill(str(term), width=65) for term in frame["term"]], fontsize=9)
    axis.invert_yaxis()
    axis.set_title(title, fontsize=11, pad=14)
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="x", alpha=0.2)
    axis.set_axisbelow(True)
    files = []
    for suffix in (".png", ".pdf"):
        destination = path.with_suffix(suffix)
        figure.savefig(destination, dpi=180)
        files.append(str(destination.resolve()))
    figure.clear()
    return files


def regenerate_enrichment_plots(comparison_dir, alpha=0.05, top_kegg=20, top_per_category=5):
    """Create direction-specific KEGG and combined plots from complete saved CSVs.

    The combined categories match the previous KEGG plotting script: GO BP/CC/MF,
    KEGG, and DisGeNET. Term caps affect display only. Missing or failed library
    results are explicitly recorded, and no figure is made for an empty set of
    significant terms. Exported plot-data CSVs identify every displayed term.
    """
    # Import at call time so enrichment.py can retain its public plotting exports.
    from tools.omic_tools.pseudobulk_pipeline.enrichment import (
        DIRECTIONS, TABLE_COLUMNS, _hash_file, _utc_now, _validate_options, _write_json,
    )

    _validate_options(alpha, 0, 1, 0)
    if any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in (top_kegg, top_per_category)):
        raise ValueError("plot term limits must be positive integers")
    comparison_dir = Path(comparison_dir).resolve()
    enrichment_dir = comparison_dir / "enrichment"
    metadata = json.loads((enrichment_dir / "metadata.json").read_text())
    plot_root = enrichment_dir / "plots"
    plot_root.mkdir(exist_ok=True)
    manifest = {
        "status": "empty", "generated_at": _utc_now(), "alpha": alpha,
        "adjusted_p_value_operator": "<", "source": "Saved enrichment tables only",
        "top_kegg": top_kegg, "top_per_category": top_per_category,
        "zero_adjusted_p_values_clipped_to": float(np.finfo(float).tiny),
        "interpretation": "Gene-set overrepresentation; fold-change direction does not establish pathway activity.",
        "directions": {},
    }
    for direction in DIRECTIONS:
        plot_dir = plot_root / direction
        plot_dir.mkdir(exist_ok=True)
        tables = {}
        info = {"sources": [], "unavailable_libraries": {}, "plots": {}}
        manifest["directions"][direction] = info
        for library in COMBINED_CATEGORIES:
            path = enrichment_dir / direction / "tables" / f"{library}.csv"
            if not path.is_file():
                info["unavailable_libraries"][library] = "No successful saved table"
                continue
            try:
                tables[library] = _read_plot_table(path, library, direction, alpha)
                info["sources"].append({"path": str(path), "sha256": _hash_file(path)})
            except (ValueError, pd.errors.ParserError) as exc:
                info["unavailable_libraries"][library] = f"{type(exc).__name__}: {exc}"
        kegg = tables.get("KEGG_2021_Human", pd.DataFrame(columns=TABLE_COLUMNS)).head(top_kegg).copy()
        combined_parts = []
        for library, category in COMBINED_CATEGORIES.items():
            if library in tables:
                part = tables[library].head(top_per_category).copy()
                part["category"] = category
                combined_parts.append(part)
        combined = pd.concat(combined_parts, ignore_index=True) if combined_parts else pd.DataFrame(columns=(*TABLE_COLUMNS, "category"))
        label = _direction_label(direction, metadata.get("contrast", {}))
        de_alpha = metadata.get("thresholds", {}).get("alpha", "unknown")
        minimum_fc = metadata.get("thresholds", {}).get("log2fc_min", "unknown")
        for name, frame in (("kegg_dotplot", kegg), ("pathway_combined_plot", combined)):
            data_path = plot_dir / f"{name}_data.csv"
            frame.to_csv(data_path, index=False)
            item = {"status": "empty", "term_count": len(frame), "data": str(data_path), "files": []}
            info["plots"][name] = item
            if frame.empty:
                item["reason"] = "No saved terms with finite adjusted p-value below alpha and positive overlap count."
                # Remove only previously generated figures from this new run so a
                # stricter regeneration cannot leave an old plot that looks current.
                for suffix in (".png", ".pdf"):
                    (plot_dir / f"{name}{suffix}").unlink(missing_ok=True)
                continue
            title = (
                f"{comparison_dir.name}: {label}\n"
                f"{'Combined categories' if name == 'pathway_combined_plot' else 'KEGG_2021_Human'}; "
                f"Enrichr adjusted p < {alpha:g}\n"
                f"Input DE adjusted p < {de_alpha}; |log2 fold change| ≥ {minimum_fc}"
            )
            item["files"] = _render_plot(frame, plot_dir / name, title, combined=name == "pathway_combined_plot")
            item["status"] = "success"
            manifest["status"] = "success"
    _write_json(plot_root / "manifest.json", manifest)
    return manifest

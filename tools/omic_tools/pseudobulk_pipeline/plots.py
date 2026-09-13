"""Plot saved DESeq2 results and donor profiles without fitting any DE model.

``plot_comparison`` writes a new ``plots/`` directory beneath a comparison.
Every dot is a donor; exported point tables retain the donor metadata and saved
DESeq2 statistics.  The input files and any pre-existing plot directory are
never modified.  PCA is descriptive: genes are centered, not scaled, and are
selected by variance in the saved VST matrix independently of the DE results.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


_RESULT_NUMERIC = ("baseMean", "log2FoldChange", "lfcSE", "stat", "pvalue", "padj")
_COLORS = ("#3B6B9A", "#B84C4C")


def _read_csv(path: Path, *, identifier: str) -> pd.DataFrame:
    # pandas otherwise silently renames repeated column labels, losing evidence
    # of a duplicate donor column. Read the original header before parsing.
    with path.open(newline="") as handle:
        header = next(csv.reader(handle), [])
    if not header or len(set(header)) != len(header):
        raise ValueError(f"{path.name} has an empty or duplicate column header")
    frame = pd.read_csv(path, dtype={identifier: str}, keep_default_na=False)
    if identifier not in frame:
        raise ValueError(f"{path.name} must contain {identifier}")
    if frame[identifier].str.strip().eq("").any() or frame[identifier].duplicated().any():
        raise ValueError(f"{path.name} must have nonempty, unique {identifier} values")
    return frame


def _read_matrix(path: Path, sample_ids: list[str], *, nonnegative: bool) -> pd.DataFrame:
    frame = _read_csv(path, identifier="gene")
    if frame.columns[0] != "gene":
        raise ValueError(f"{path.name} must have gene as its first column")
    frame = frame.set_index("gene")
    if set(frame.columns) != set(sample_ids):
        missing = sorted(set(sample_ids) - set(frame.columns))
        extra = sorted(set(frame.columns) - set(sample_ids))
        raise ValueError(f"{path.name} donor sample IDs differ from metadata: missing={missing}, extra={extra}")
    frame = frame.loc[:, sample_ids].apply(pd.to_numeric, errors="raise")
    values = frame.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (nonnegative and (values < 0).any()):
        raise ValueError(f"{path.name} must contain finite {'nonnegative ' if nonnegative else ''}values")
    return frame


def _figure(*, figsize=(7.2, 5.6)) -> Figure:
    figure = Figure(figsize=figsize)
    FigureCanvasAgg(figure)
    return figure


def _save_figure(figure: Figure, directory: Path, name: str) -> list[str]:
    names = []
    try:
        for extension in ("png", "pdf"):
            filename = f"{name}.{extension}"
            # Exclusive creation protects files even if another process writes
            # into this newly reserved directory while plotting is in progress.
            with (directory / filename).open("xb") as handle:
                figure.savefig(handle, format=extension, dpi=180, bbox_inches="tight")
            names.append(filename)
    finally:
        figure.clear()
    return names


def _write_frame(frame: pd.DataFrame, directory: Path, name: str) -> None:
    with (directory / name).open("x", newline="") as handle:
        frame.to_csv(handle, index=False)


def _write_manifest(directory: Path, manifest: dict) -> dict:
    with (directory / "plot_manifest.json").open("x") as handle:
        json.dump(manifest, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return manifest


def _blank_axes(axes, title: str, reason: str) -> None:
    axes.set_title(title, fontweight="bold")
    axes.text(0.5, 0.5, reason.replace("_", " "), ha="center", va="center", transform=axes.transAxes)
    axes.set_axis_off()


def _style_axes(axes) -> None:
    axes.spines[["top", "right"]].set_visible(False)
    axes.grid(alpha=0.15, zorder=0)


def _pca(vst: pd.DataFrame, metadata: pd.DataFrame, contrast: dict, directory: Path) -> dict:
    """Plot donors from saved VST expression; gene selection uses variance, not DE."""
    genes = pd.DataFrame(columns=["gene", "vst_variance", "rank"])
    scores = metadata.copy()
    scores["PC1"] = np.nan
    scores["PC2"] = np.nan
    variance_table = pd.DataFrame(columns=["component", "variance", "explained_variance_ratio"])
    summary = {
        "status": "skipped",
        "input": "vst_expression.csv",
        "sample_unit": "donor",
        "n_donors": len(metadata),
        "gene_selection": {
            "method": "highest donor VST variance; positive variance only",
            "maximum_genes": 500,
            "variance_ddof": 1,
            "uses_de_statistics": False,
            "tie_breaker": "gene ascending",
        },
        "center_genes": True,
        "scale_genes": False,
        "color_by": contrast["column"],
    }
    if len(metadata) < 2:
        summary["reason"] = "fewer_than_two_donors"
    else:
        gene_variance = vst.var(axis=1, ddof=1).rename("vst_variance")
        ranked = gene_variance[gene_variance > 0].reset_index()
        ranked = ranked.sort_values(["vst_variance", "gene"], ascending=[False, True], kind="stable")
        genes = ranked.head(500).copy()
        genes["rank"] = np.arange(1, len(genes) + 1)
        if genes.empty:
            summary["reason"] = "no_variable_genes"
        else:
            # Donors are observations (rows), VST genes are features (columns).
            matrix = vst.loc[genes["gene"]].to_numpy(dtype=float).T
            centered = matrix - matrix.mean(axis=0, keepdims=True)
            left, singular, loadings = np.linalg.svd(centered, full_matrices=False)
            components = min(len(metadata) - 1, len(genes))
            left, singular, loadings = left[:, :components], singular[:components], loadings[:components]
            tolerance = singular[0] * max(centered.shape) * np.finfo(float).eps
            singular[singular <= tolerance] = 0.0
            # A consistent sign makes repeat exports easier to compare while
            # leaving the PCA distances and variance unchanged.
            signs = np.sign(loadings[np.arange(components), np.abs(loadings).argmax(axis=1)])
            signs[signs == 0] = 1
            values = left * singular * signs
            eigenvalues = singular**2 / (len(metadata) - 1)
            ratios = eigenvalues / eigenvalues.sum()
            scores = pd.concat([
                metadata.copy(),
                pd.DataFrame(values, index=metadata.index,
                             columns=[f"PC{i + 1}" for i in range(components)]),
            ], axis=1)
            if components == 1:
                scores["PC2"] = 0.0
                eigenvalues = np.append(eigenvalues, 0.0)
                ratios = np.append(ratios, 0.0)
            variance_table = pd.DataFrame(
                {"component": [f"PC{i + 1}" for i in range(len(ratios))],
                 "variance": eigenvalues, "explained_variance_ratio": ratios}
            )
            summary.update(status="success", n_components=components,
                           effective_rank=int(np.count_nonzero(singular)), rank_tolerance=float(tolerance))
    summary["n_genes"] = len(genes)
    figure = _figure()
    axes = figure.subplots()
    if summary["status"] == "success":
        for group, color in zip((contrast["reference"], contrast["numerator"]), _COLORS):
            subset = scores.loc[scores[contrast["column"]] == group]
            axes.scatter(subset["PC1"], subset["PC2"], s=46, color=color, edgecolors="white", linewidths=0.5,
                         label=f"{group} (n={len(subset)})", zorder=3)
        if len(scores) <= 16:
            for row in scores.itertuples(index=False):
                axes.annotate(row.sample_id, (row.PC1, row.PC2), xytext=(4, 4), textcoords="offset points", fontsize=6)
        percentages = variance_table["explained_variance_ratio"].to_numpy() * 100
        axes.set(xlabel=f"PC1 ({percentages[0]:.1f}% variance)", ylabel=f"PC2 ({percentages[1]:.1f}% variance)",
                 title=f"Donor VST PCA | {len(genes)} most variable genes")
        axes.legend(frameon=False)
        _style_axes(axes)
    else:
        _blank_axes(axes, "Donor VST PCA", summary["reason"])
    figure.text(0.5, 0.01, "Each point is one donor; genes selected without DE statistics.", ha="center", fontsize=8)
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    _write_frame(genes, directory, "pca_genes.csv")
    _write_frame(scores, directory, "pca_scores.csv")
    _write_frame(variance_table, directory, "pca_variance_explained.csv")
    summary["files"] = _save_figure(figure, directory, "pca") + ["pca_genes.csv", "pca_scores.csv", "pca_variance_explained.csv"]
    return summary


def _valid_results(results: pd.DataFrame) -> pd.DataFrame:
    valid = np.isfinite(results["log2FoldChange"]) & np.isfinite(results["padj"]) & results["padj"].between(0, 1)
    if "tested" in results:
        valid &= results["tested"].astype(str).str.lower().isin(["true", "1"])
    return results.loc[valid].copy()


def _volcano(results: pd.DataFrame, contrast: dict, alpha: float, directory: Path) -> dict:
    """Plot saved DESeq2 effect estimates and adjusted P-values without refitting."""
    points = _valid_results(results)
    positives = points.loc[points["padj"] > 0, "padj"]
    floor = max(float(positives.min()) / 10, np.finfo(float).tiny) if not positives.empty else 1e-300
    points["negative_log10_padj"] = -np.log10(points["padj"].clip(lower=floor))
    points["significant"] = points["padj"] < alpha
    summary = {
        "status": "success" if len(points) else "skipped",
        "input": "results.csv",
        "statistical_test": "saved DESeq2 results only",
        "n_genes": len(points),
        "n_omitted_genes": len(results) - len(points),
        "n_significant": int(points["significant"].sum()),
        "zero_padj_count": int(points["padj"].eq(0).sum()),
        "padj_floor_for_display": floor,
        "alpha": alpha,
    }
    figure = _figure()
    axes = figure.subplots()
    if len(points):
        categories = (
            (~points["significant"], "#A3A8AE", "Not significant"),
            (points["significant"] & (points["log2FoldChange"] < 0), _COLORS[0], f"Higher in {contrast['reference']}"),
            (points["significant"] & (points["log2FoldChange"] >= 0), _COLORS[1], f"Higher in {contrast['numerator']}"),
        )
        for mask, color, label in categories:
            subset = points.loc[mask]
            if len(subset):
                axes.scatter(subset["log2FoldChange"], subset["negative_log10_padj"], s=12, c=color, alpha=0.7,
                             edgecolors="none", label=f"{label} (n={len(subset)})", rasterized=True, zorder=3)
        axes.axhline(-np.log10(alpha), color="#777777", linestyle="--", linewidth=0.8)
        axes.axvline(0, color="#999999", linewidth=0.6)
        axes.set(xlabel=f"Saved DESeq2 log₂ fold change ({contrast['numerator']} / {contrast['reference']})",
                 ylabel="−log₁₀(saved DESeq2 adjusted P value)", title="Donor pseudobulk differential expression")
        axes.legend(frameon=False, fontsize=8)
        _style_axes(axes)
    else:
        summary["reason"] = "no_finite_saved_results"
        _blank_axes(axes, "Donor pseudobulk differential expression", summary["reason"])
    note = f"Saved DESeq2 results; adjusted P < {alpha:g}."
    if summary["zero_padj_count"]:
        note += f" Zero adjusted P displayed at {floor:.2g}."
    figure.text(0.5, 0.01, note, ha="center", fontsize=8)
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    _write_frame(points, directory, "volcano_points.csv")
    summary["files"] = _save_figure(figure, directory, "volcano") + ["volcano_points.csv"]
    return summary


def _format_stat(value: float) -> str:
    return f"{value:.3g}" if np.isfinite(value) else "NA"


def _gene_expression(results: pd.DataFrame, counts: pd.DataFrame, metadata: pd.DataFrame,
                     contrast: dict, alpha: float, top_genes: int, directory: Path) -> dict:
    eligible = _valid_results(results)
    eligible = eligible.loc[eligible["gene"].isin(counts.index)].copy()
    eligible["_absolute_lfc"] = eligible["log2FoldChange"].abs()
    ranked = eligible.sort_values(["padj", "_absolute_lfc", "gene"], ascending=[True, False, True], kind="stable")
    selected = ranked.head(top_genes).drop(columns="_absolute_lfc")
    parts = []
    for _, result in selected.iterrows():
        part = metadata.copy()
        part.insert(0, "gene", result["gene"])
        part["normalized_count"] = counts.loc[result["gene"]].to_numpy(dtype=float)
        part["log2_normalized_count_plus_1"] = np.log2(part["normalized_count"] + 1)
        for statistic in _RESULT_NUMERIC:
            part[f"deseq2_{statistic}"] = result[statistic]
        parts.append(part)
    columns = ["gene", *metadata.columns, "normalized_count", "log2_normalized_count_plus_1", *[f"deseq2_{s}" for s in _RESULT_NUMERIC]]
    points = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns)
    summary = {
        "status": "success" if len(points) else "skipped",
        "input": "normalized_counts.csv",
        "sample_unit": "donor",
        "display_transform": "log2(DESeq2 normalized count + 1)",
        "statistical_test": "saved DESeq2 results only",
        "selection": "lowest saved DESeq2 padj",
        "tie_breaker": "absolute log2FoldChange descending, gene ascending",
        "top_genes_requested": top_genes,
        "genes": selected["gene"].tolist(),
        "n_donors": len(metadata),
        "n_significant": int((selected["padj"] < alpha).sum()),
        "alpha": alpha,
        "group_order": [contrast["reference"], contrast["numerator"]],
    }
    if points.empty:
        summary["reason"] = "no_donors" if metadata.empty else ("top_genes_zero" if top_genes == 0 else "no_finite_saved_results_with_counts")
        figure = _figure()
        _blank_axes(figure.subplots(), "Donor gene expression", summary["reason"])
    else:
        ncols = min(3, len(selected))
        nrows = (len(selected) + ncols - 1) // ncols
        figure = _figure(figsize=(max(5.2, ncols * 4.2), 3.5 * nrows + 1.1))
        axes = figure.subplots(nrows, ncols, squeeze=False).ravel()
        for axes_i, (_, result) in zip(axes, selected.iterrows()):
            gene_points = points.loc[points["gene"] == result["gene"]]
            for x, (group, color) in enumerate(zip(summary["group_order"], _COLORS)):
                subset = gene_points.loc[gene_points[contrast["column"]] == group]
                values = subset["log2_normalized_count_plus_1"].to_numpy()
                # Gaussian KDE is undefined for singleton or constant groups.
                # Their donor dots remain visible, without inventing a density.
                if len(values) > 1 and np.ptp(values) > 0:
                    violin = axes_i.violinplot(values, positions=[x], widths=0.75, showextrema=False)
                    for body in violin["bodies"]:
                        body.set_facecolor(color)
                        body.set_edgecolor(color)
                        body.set_alpha(0.22)
                jitter = np.random.default_rng(0).uniform(-0.11, 0.11, len(values)) if len(values) > 1 else np.zeros(len(values))
                axes_i.scatter(x + jitter, values, s=25, color=color, alpha=0.85, edgecolors="white", linewidths=0.35, zorder=3)
            labels = [f"{group}\n(n={int((metadata[contrast['column']] == group).sum())})" for group in summary["group_order"]]
            axes_i.set_xticks([0, 1], labels)
            axes_i.set_xlim(-0.6, 1.6)
            axes_i.set_ylabel("log₂(DESeq2 normalized count + 1)", fontsize=8)
            axes_i.set_title(f"{result['gene']}\nSaved DESeq2: log₂FC={_format_stat(result['log2FoldChange'])}; "
                             f"Wald z={_format_stat(result['stat'])}\npadj={_format_stat(result['padj'])}", fontsize=9)
            _style_axes(axes_i)
        for unused in axes[len(selected):]:
            unused.set_axis_off()
        figure.suptitle("Donor gene expression", fontweight="bold", y=0.99)
        note = "Each dot is one donor; annotations come from the saved DESeq2 model."
        if summary["n_significant"] == 0:
            note += f"\nNo selected gene meets adjusted P < {alpha:g}."
        figure.text(0.5, 0.015, note, ha="center", fontsize=8)
        figure.tight_layout(rect=(0, 0.065, 1, 0.94))
    _write_frame(points, directory, "gene_expression_points.csv")
    _write_frame(selected, directory, "gene_expression_genes.csv")
    summary["files"] = _save_figure(figure, directory, "gene_expression") + ["gene_expression_points.csv", "gene_expression_genes.csv"]
    return summary


def plot_comparison(comparison_dir: str | Path, top_genes: int = 12) -> dict:
    """Create auditable donor plots from one saved DESeq2 comparison.

    The expected status schema is ``status``, ``contrast`` (``column``,
    ``numerator``, ``reference``), ``formula``, and optional ``alpha`` (0.05).
    A skipped or failed DE analysis creates only a skipped plot manifest.
    Successful comparisons require result, normalized-count, VST, and donor
    metadata CSVs. Matrices must have ``gene`` first and exactly the metadata's
    unique ``sample_id`` columns; their donor order is aligned before plotting.

    Raises ``FileExistsError`` if ``plots/`` already exists, and ``ValueError``
    for invalid inputs. The plotting directory is reserved only after all
    successful-comparison inputs have been validated.
    """
    if isinstance(top_genes, bool) or not isinstance(top_genes, (int, np.integer)) or top_genes < 0:
        raise ValueError("top_genes must be a nonnegative integer")
    top_genes = int(top_genes)
    comparison = Path(comparison_dir)
    directory = comparison / "plots"
    if directory.exists() or directory.is_symlink():
        raise FileExistsError(f"Refusing to overwrite existing plot output: {directory}")
    with (comparison / "status.json").open() as handle:
        status = json.load(handle)
    manifest = {
        "schema_version": 1,
        "status": "skipped",
        "comparison": comparison.name,
        "sample_unit": "donor",
        "differential_expression_rerun": False,
        "deseq2_status": status.get("status"),
        "contrast": status.get("contrast"),
        "formula": status.get("formula"),
        "software_versions": {"numpy": np.__version__, "pandas": pd.__version__, "matplotlib": matplotlib.__version__},
    }
    if status.get("status") != "success":
        manifest["reason"] = status.get("reason") or "deseq2_comparison_not_successful"
        directory.mkdir()
        return _write_manifest(directory, manifest)
    contrast = status.get("contrast", {})
    if any(not isinstance(contrast.get(key), str) or not contrast[key] for key in ("column", "numerator", "reference")):
        raise ValueError("status.json requires contrast column, numerator, and reference")
    if contrast["numerator"] == contrast["reference"]:
        raise ValueError("Contrast numerator and reference must differ")
    alpha = float(status.get("alpha", 0.05))
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("status.json alpha must be between 0 and 1")
    metadata = _read_csv(comparison / "donor_metadata.csv", identifier="sample_id")
    if contrast["column"] not in metadata:
        raise ValueError(f"Donor metadata is missing contrast column {contrast['column']}")
    groups = set(metadata[contrast["column"]])
    if not groups.issubset({contrast["numerator"], contrast["reference"]}):
        raise ValueError("Donor metadata includes values outside the saved contrast")
    sample_ids = metadata["sample_id"].tolist()
    counts = _read_matrix(comparison / "normalized_counts.csv", sample_ids, nonnegative=True)
    vst = _read_matrix(comparison / "vst_expression.csv", sample_ids, nonnegative=False)
    results = _read_csv(comparison / "results.csv", identifier="gene")
    if not {"log2FoldChange", "padj"}.issubset(results):
        raise ValueError("results.csv requires saved log2FoldChange and padj columns")
    for column in _RESULT_NUMERIC:
        results[column] = pd.to_numeric(results[column], errors="coerce") if column in results else np.nan
    inputs = ["status.json", "donor_metadata.csv", "normalized_counts.csv", "vst_expression.csv", "results.csv"]
    manifest["input_sha256"] = {name: hashlib.sha256((comparison / name).read_bytes()).hexdigest() for name in inputs}
    manifest["n_donors"] = len(metadata)
    manifest["donor_order"] = sample_ids
    manifest["alpha"] = alpha
    # Atomic directory reservation prevents concurrent runs from overwriting one
    # another. Partial outputs are left in place if rendering fails.
    directory.mkdir()
    _write_frame(metadata, directory, "plotted_donors.csv")
    with matplotlib.rc_context({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "axes.titlesize": 11}):
        manifest["pca"] = _pca(vst, metadata, contrast, directory)
        manifest["volcano"] = _volcano(results, contrast, alpha, directory)
        manifest["gene_expression"] = _gene_expression(results, counts, metadata, contrast, alpha, top_genes, directory)
    manifest["status"] = "success"
    manifest["files"] = ["plot_manifest.json", "plotted_donors.csv"]
    for name in ("pca", "volcano", "gene_expression"):
        manifest["files"].extend(manifest[name]["files"])
    return _write_manifest(directory, manifest)

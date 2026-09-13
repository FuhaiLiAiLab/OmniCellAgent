"""Summarize saved DESeq2 sensitivity fits; no differential testing here."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RIBOSOME_FOCUS = ["RPLP0", "RPL13A", "RPS18", "RPL3", "RPS4X"]


def _save(fig, directory, stem):
    for suffix in ["png", "pdf"]:
        fig.savefig(directory / f"{stem}.{suffix}", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _valid(frame):
    mask = np.isfinite(frame.log2FoldChange)
    if "tested" in frame:
        mask &= frame.tested.astype(str).str.lower().isin(["true", "1"])
    return frame.loc[mask]


def _concordance(a, b, tables):
    if a not in tables or b not in tables:
        return pd.DataFrame(), {"reference": a, "comparison": b, "status": "unavailable"}
    joint = _valid(tables[a]).merge(_valid(tables[b]), on="gene", suffixes=("_reference", "_comparison"))
    if len(joint) < 3:
        return joint, {"reference": a, "comparison": b, "status": "insufficient_genes", "n_genes": len(joint)}
    x, y = joint.log2FoldChange_reference, joint.log2FoldChange_comparison
    rho = x.corr(y, method="spearman")
    summary = {"reference": a, "comparison": b, "status": "success", "n_genes": len(joint),
               "spearman": float(rho) if np.isfinite(rho) else None,
               "same_sign_fraction": float((np.sign(x) == np.sign(y)).mean()),
               "median_absolute_lfc_change": float((y - x).abs().median())}
    return joint, summary


def build_report(source_run, output_dir):
    source = Path(source_run) / "comparisons/ad_vs_control"
    output = Path(output_dir)
    directory = output / "summary"
    directory.mkdir(exist_ok=False)
    manifest = json.loads((output / "manifest.json").read_text())
    order = manifest["model_order"]
    alpha = float(manifest.get("settings", {}).get("alpha", .05))
    tables, rows = {}, []
    for name in order:
        model_dir = output / "models" / name
        status = json.loads((model_dir / "status.json").read_text())
        row = {"model": name, "status": status["status"], "formula": status.get("formula"),
               "AD_donors": status.get("group_counts", {}).get("AD", 0),
               "control_donors": status.get("group_counts", {}).get("control", 0),
               "count_filter_genes": status.get("filter", {}).get("n_retained_genes"),
               "valid_tests": status.get("filter", {}).get("n_valid_tests"),
               "nonconverged_genes": status.get("filter", {}).get("n_nonconverged"),
               "unavailable_pvalues": status.get("filter", {}).get("n_unavailable_pvalues"),
               "omitted_study": status.get("omitted_study"), "reason": status.get("reason")}
        path = model_dir / "results.csv"
        if status["status"] == "success" and path.exists():
            frame = pd.read_csv(path)
            tables[name] = frame
            significant = _valid(frame).loc[lambda data: data.padj.lt(alpha)]
            row.update(significant_genes=len(significant), up=int(significant.log2FoldChange.gt(0).sum()),
                       down=int(significant.log2FoldChange.lt(0).sum()))
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(directory / "model_summary.csv", index=False)
    effects = pd.concat([frame.assign(model=name) for name, frame in tables.items()], ignore_index=True) if tables else pd.DataFrame()
    effects.to_csv(directory / "gene_effects.csv", index=False)

    pairs = [("full_A", "full_B"), ("full_B", "full_C"), ("full_C", "shared_C")]
    all_pairs = pairs + [("shared_C", name) for name in order if name.startswith("leave_one_out_")]
    comparisons = [_concordance(a, b, tables)[1] for a, b in all_pairs]
    pd.DataFrame(comparisons).to_csv(directory / "effect_concordance.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for axis, (a, b) in zip(axes, pairs):
        joined, stats = _concordance(a, b, tables)
        if len(joined):
            x, y = joined.log2FoldChange_reference, joined.log2FoldChange_comparison
            axis.scatter(x, y, s=5, alpha=.22, color="#365f8d", rasterized=True)
            lim = max(float(x.abs().max()), float(y.abs().max()), 1)
            axis.plot([-lim, lim], [-lim, lim], "--", color="grey", linewidth=.8)
            rho = stats.get("spearman")
            label = f"n={len(joined):,}; Spearman={rho:.3f}" if rho is not None else f"n={len(joined):,}"
            axis.set_title(label)
        else:
            axis.text(.5, .5, "Comparison unavailable", ha="center", transform=axis.transAxes)
        axis.set(xlabel=f"{a}: AD/control log2FC", ylabel=f"{b}: AD/control log2FC")
    fig.suptitle("Donor-level effect concordance | all jointly tested genes")
    fig.tight_layout()
    _save(fig, directory, "fold_change_concordance")

    original = pd.read_csv(source / "results.csv")
    present = set(original.gene)
    focus = [gene for gene in RIBOSOME_FOCUS if gene in present]
    focus += [gene for gene in original.dropna(subset=["padj"]).sort_values(["padj", "gene"]).gene if gene not in focus][:5]
    pd.DataFrame({"gene": focus, "selection": ["prespecified ribosome example" if gene in RIBOSOME_FOCUS
                                                else "top gene in original adjusted AD result" for gene in focus]}).to_csv(directory / "focus_genes.csv", index=False)
    successful = [name for name in order if name in tables]
    fig, axes = plt.subplots(max(1, int(np.ceil(len(focus) / 2))), 2,
                             figsize=(13, max(4, 2.8 * np.ceil(len(focus) / 2))), squeeze=False)
    forest = []
    for axis, gene in zip(axes.flat, focus):
        for i, name in enumerate(successful):
            match = tables[name].loc[tables[name].gene.eq(gene)]
            if match.empty:
                continue
            record = match.iloc[0]
            value = record.log2FoldChange
            lo = record.get("lfc_ci_low", value - 1.96 * record.lfcSE)
            hi = record.get("lfc_ci_high", value + 1.96 * record.lfcSE)
            forest.append({"gene": gene, "model": name, "log2FoldChange": value,
                           "lfc_ci_low": lo, "lfc_ci_high": hi, "padj": record.padj,
                           "tested": str(record.get("tested", True)).lower() in {"true", "1"},
                           "filter_reason": record.get("filter_reason", "")})
            if forest[-1]["tested"] and np.isfinite([value, lo, hi]).all():
                axis.errorbar(value, i, xerr=[[max(0, value-lo)], [max(0, hi-value)]], fmt="o",
                              color="#9b352f" if name == "shared_C" else "#365f8d", markersize=3, capsize=2)
        axis.axvline(0, color="grey", linewidth=.7)
        axis.set(yticks=np.arange(len(successful)), yticklabels=successful, title=gene, xlabel="AD/control log2FC (95% Wald interval)")
        axis.invert_yaxis()
    for axis in list(axes.flat)[len(focus):]:
        axis.axis("off")
    fig.tight_layout()
    _save(fig, directory, "focused_effect_forest")
    pd.DataFrame(forest).to_csv(directory / "focused_effects.csv", index=False)

    metadata = pd.read_csv(source / "donor_metadata.csv")
    normalized = pd.read_csv(source / "normalized_counts.csv", index_col=0)
    if set(normalized.columns) != set(metadata.sample_id) or metadata.sample_id.duplicated().any():
        raise ValueError("Source normalized counts and unique donor metadata do not align")
    shared = manifest["shared_studies"]
    within = []
    for study in ["pooled_all_donors"] + shared:
        donors = metadata if study == "pooled_all_donors" else metadata.loc[metadata.study.eq(study)]
        ad = donors.loc[donors.disease.eq("AD"), "sample_id"]
        control = donors.loc[donors.disease.eq("control"), "sample_id"]
        for gene in focus:
            mean_ad = normalized.loc[gene, ad].mean() if len(ad) else np.nan
            mean_control = normalized.loc[gene, control].mean() if len(control) else np.nan
            lfc = np.log2(mean_ad / mean_control) if mean_ad > 0 and mean_control > 0 else np.nan
            within.append({"study": study, "gene": gene, "AD_donors": len(ad), "control_donors": len(control),
                           "AD_mean_normalized_count": mean_ad, "control_mean_normalized_count": mean_control,
                           "descriptive_log2fc": lfc, "inferential_test": "none"})
    within = pd.DataFrame(within)
    within.to_csv(directory / "within_study_effects.csv", index=False)
    lookup = pd.DataFrame({"label": [f"Study {i+1}" for i in range(len(shared))], "study": shared})
    lookup.to_csv(directory / "study_labels.csv", index=False)
    fig, axis = plt.subplots(figsize=(max(6, len(shared) + 3), max(3, .45*len(focus)+2)))
    if focus:
        matrix = within.pivot(index="gene", columns="study", values="descriptive_log2fc").reindex(index=focus, columns=["pooled_all_donors"]+shared)
        values = matrix.to_numpy()
        finite = values[np.isfinite(values)]
        bound = max(float(np.abs(finite).max()), .1) if len(finite) else 1
        picture = axis.imshow(np.ma.masked_invalid(values), cmap="RdBu_r", vmin=-bound, vmax=bound, aspect="auto")
        axis.set(yticks=np.arange(len(focus)), yticklabels=focus,
                 xticks=np.arange(len(shared)+1), xticklabels=["Pooled"]+lookup.label.tolist())
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                axis.text(j, i, f"{values[i,j]:.2f}" if np.isfinite(values[i,j]) else "NA", ha="center", va="center", fontsize=8)
        fig.colorbar(picture, ax=axis, label="Descriptive AD/control log2 mean ratio")
    axis.set_title("Within-study donor expression effects\nDescriptive means; no per-study DE test")
    fig.tight_layout()
    _save(fig, directory, "within_study_effects")

    info = {"de_recomputed_by_reporting": False, "sample_unit": "donor",
            "alpha": alpha, "model_order": order, "focus_genes": focus,
            "confidence_intervals": "Approximate 95% Wald intervals; not multiplicity-adjusted",
            "within_study_effects": "Descriptive ratios using original full-cohort normalized donor counts; no new DE tests",
            "concordance": comparisons}
    (directory / "reporting_manifest.json").write_text(json.dumps(info, indent=2, allow_nan=False) + "\n")
    text = ["# AD donor sensitivity analysis", "", "All fits compare AD against control at the donor level.", "",
            "| Model | Formula | AD | Control | Genes at fixed-family FDR < 0.05 | Status |",
            "|---|---|---:|---:|---:|---|"]
    for row in rows:
        text.append(f"| {row['model']} | `{row['formula']}` | {row['AD_donors']} | {row['control_donors']} | {row.get('significant_genes', '—')} | {row['status']} |")
    text += ["", "Full A/B/C use identical donors, count-filter genes and size factors. Subsets re-estimate normalization.",
             "The sensitivity audit uses a fixed BH family after count filtering and disables adaptive independent filtering. Its FDR counts need not equal the original pipeline's 356 genes.", "",
             "Compare effect direction, magnitude and uncertainty. Loss of significance after removing donors can reflect lower power; it is not by itself evidence of inconsistency.", "",
             "`gene_effects.csv` contains complete saved effects. `effect_concordance.csv` describes all jointly tested genes. Forest plots show prespecified ribosome examples and five leading genes from the original adjusted analysis.", "",
             "`within_study_effects.csv` and its heatmap use simple donor means, not independently fitted DE tests. `study_labels.csv` maps short labels to exact study identifiers.", "",
             "No enrichment was rerun and no original results were overwritten.", ""]
    (directory / "report.md").write_text("\n".join(text))
    return info

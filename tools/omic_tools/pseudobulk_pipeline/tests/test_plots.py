"""Behavioral checks for plots derived only from saved donor-level outputs."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _plot_comparison():
    """Load just the plotting module, without importing unrelated workflows."""
    path = Path(__file__).parents[1] / "plots.py"
    assert path.is_file(), "The saved-result plotting component is not implemented"
    spec = importlib.util.spec_from_file_location("pseudobulk_plots_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.plot_comparison


def _write_comparison(path, *, variable_genes=3, donors=4, significant=True):
    path.mkdir()
    ids = [f"donor_{i}" for i in range(donors)]
    # Metadata and the two matrices deliberately use different donor orders.
    metadata = pd.DataFrame(
        {
            "sample_id": ids[::-1],
            "disease": ["AD" if i % 2 else "control" for i in range(donors)][::-1],
            "sex": ["male" if i % 2 else "female" for i in range(donors)][::-1],
            "study": ["study_A"] * donors,
            "source": ["source_A"] * donors,
            "age": list(range(60, 60 + donors))[::-1],
        }
    )
    metadata.to_csv(path / "donor_metadata.csv", index=False)
    genes = [f"VAR_{i:03d}" for i in range(variable_genes)] + ["DE_ONLY"]
    vst_values = [np.arange(donors, dtype=float) * (i + 1) for i in range(variable_genes)]
    vst_values.append(np.full(donors, 7.0))
    vst = pd.DataFrame(vst_values, index=genes, columns=ids)
    vst.index.name = "gene"
    vst.to_csv(path / "vst_expression.csv")
    counts = pd.DataFrame(
        [[float((i + 1) * (j + 1)) for j in range(donors)] for i in range(len(genes))],
        index=genes,
        columns=ids,
    )
    counts.index.name = "gene"
    counts.loc[:, ids[1:] + ids[:1]].to_csv(path / "normalized_counts.csv")
    results = pd.DataFrame(
        {
            "gene": genes,
            "baseMean": [50.0] * len(genes),
            "log2FoldChange": [0.5] * variable_genes + [2.0],
            "lfcSE": [0.2] * len(genes),
            "stat": [2.5] * variable_genes + [10.0],
            "pvalue": [0.3] * variable_genes + [1e-20 if significant else 0.2],
            "padj": [0.8] * variable_genes + [1e-18 if significant else 0.7],
            "tested": [True] * len(genes),
            "filter_reason": [""] * len(genes),
        }
    )
    results.to_csv(path / "results.csv", index=False)
    status = {
        "status": "success",
        "contrast": {"column": "disease", "numerator": "AD", "reference": "control"},
        "formula": "~ disease",
        "alpha": 0.05,
        "included_donors": metadata["sample_id"].tolist(),
    }
    (path / "status.json").write_text(json.dumps(status))
    return metadata, counts, results


def _hashes(path):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in path.iterdir() if p.is_file()}


def test_pca_uses_variable_vst_genes_and_preserves_donor_alignment(tmp_path):
    comparison = tmp_path / "ad_vs_control"
    metadata, counts, _ = _write_comparison(comparison, variable_genes=501)
    before = _hashes(comparison)

    manifest = _plot_comparison()(comparison, top_genes=1)

    assert _hashes(comparison) == before
    assert manifest["sample_unit"] == "donor"
    assert manifest["differential_expression_rerun"] is False
    genes = pd.read_csv(comparison / "plots/pca_genes.csv")
    assert len(genes) == 500
    assert genes.iloc[0]["gene"] == "VAR_500"
    assert "DE_ONLY" not in genes["gene"].tolist()
    assert "VAR_000" not in genes["gene"].tolist()
    scores = pd.read_csv(comparison / "plots/pca_scores.csv")
    assert scores["sample_id"].tolist() == metadata["sample_id"].tolist()
    assert scores["disease"].tolist() == metadata["disease"].tolist()
    assert scores["age"].tolist() == metadata["age"].tolist()
    # These donor profiles vary along one dimension, in ID order.
    assert scores["PC1"].is_monotonic_increasing or scores["PC1"].is_monotonic_decreasing
    # A null PC must not amplify floating-point roundoff into apparent structure.
    assert scores["PC2"].eq(0).all()
    variance = pd.read_csv(comparison / "plots/pca_variance_explained.csv")
    assert variance.iloc[0]["explained_variance_ratio"] == pytest.approx(1.0)
    points = pd.read_csv(comparison / "plots/gene_expression_points.csv")
    assert points["gene"].unique().tolist() == ["DE_ONLY"]
    assert points["sample_id"].tolist() == metadata["sample_id"].tolist()
    assert points["normalized_count"].tolist() == [2008.0, 1506.0, 1004.0, 502.0]
    assert points["log2_normalized_count_plus_1"].to_numpy() == pytest.approx(np.log2([2009, 1507, 1005, 503]))
    assert points["deseq2_padj"].tolist() == [1e-18] * 4
    assert points["deseq2_stat"].tolist() == [10.0] * 4
    assert manifest["pca"]["gene_selection"]["uses_de_statistics"] is False
    assert manifest["gene_expression"]["statistical_test"] == "saved DESeq2 results only"
    plotted = pd.read_csv(comparison / "plots/plotted_donors.csv")
    assert plotted["sample_id"].tolist() == metadata["sample_id"].tolist()
    for name in ("pca", "volcano", "gene_expression"):
        assert (comparison / "plots" / f"{name}.png").read_bytes().startswith(b"\x89PNG")
        assert (comparison / "plots" / f"{name}.pdf").read_bytes().startswith(b"%PDF")
    assert json.loads((comparison / "plots/plot_manifest.json").read_text()) == manifest


def test_no_significant_genes_still_shows_saved_effects_and_donor_counts(tmp_path):
    comparison = tmp_path / "no_significant"
    _, _, results = _write_comparison(comparison, significant=False)
    # Incomplete saved estimates are omitted rather than filled or recomputed.
    results.loc[0, "padj"] = np.nan
    results.loc[1, "log2FoldChange"] = np.nan
    results.to_csv(comparison / "results.csv", index=False)
    manifest = _plot_comparison()(comparison, top_genes=2)
    assert manifest["volcano"]["n_significant"] == 0
    assert manifest["gene_expression"]["n_significant"] == 0
    assert manifest["gene_expression"]["selection"] == "lowest saved DESeq2 padj"
    points = pd.read_csv(comparison / "plots/volcano_points.csv")
    assert set(points["gene"]) == {"VAR_002", "DE_ONLY"}
    assert points.set_index("gene").loc["DE_ONLY", "log2FoldChange"] == 2.0
    assert points.set_index("gene").loc["DE_ONLY", "negative_log10_padj"] == pytest.approx(-np.log10(0.7))


def test_zero_padj_has_finite_display_coordinate_and_preserves_statistic(tmp_path):
    comparison = tmp_path / "zero_padj"
    _, _, results = _write_comparison(comparison)
    results.loc[results["gene"] == "DE_ONLY", "padj"] = 0.0
    results.to_csv(comparison / "results.csv", index=False)
    manifest = _plot_comparison()(comparison, top_genes=1)
    points = pd.read_csv(comparison / "plots/volcano_points.csv")
    row = points.set_index("gene").loc["DE_ONLY"]
    assert row["padj"] == 0.0
    assert np.isfinite(row["negative_log10_padj"])
    assert manifest["volcano"]["zero_padj_count"] == 1


@pytest.mark.parametrize("donors,variable_genes,reason", [(1, 3, "fewer_than_two_donors"), (4, 0, "no_variable_genes"), (0, 0, "fewer_than_two_donors")])
def test_small_or_zero_variance_data_has_explicit_pca_skip(tmp_path, donors, variable_genes, reason):
    comparison = tmp_path / "small"
    _write_comparison(comparison, donors=donors, variable_genes=variable_genes)
    manifest = _plot_comparison()(comparison, top_genes=1)
    assert manifest["pca"]["status"] == "skipped"
    assert manifest["pca"]["reason"] == reason
    assert pd.read_csv(comparison / "plots/pca_genes.csv").empty
    scores = pd.read_csv(comparison / "plots/pca_scores.csv")
    assert len(scores) == donors
    assert scores["PC1"].isna().all()


def test_empty_saved_results_are_explicit_and_do_not_block_pca(tmp_path):
    comparison = tmp_path / "empty"
    _, _, results = _write_comparison(comparison)
    results.iloc[:0].to_csv(comparison / "results.csv", index=False)
    manifest = _plot_comparison()(comparison)
    assert manifest["pca"]["status"] == "success"
    assert manifest["volcano"]["status"] == "skipped"
    assert manifest["gene_expression"]["status"] == "skipped"
    assert pd.read_csv(comparison / "plots/volcano_points.csv").empty
    assert pd.read_csv(comparison / "plots/gene_expression_points.csv").empty


def test_existing_plot_output_is_never_overwritten(tmp_path):
    comparison = tmp_path / "existing"
    _write_comparison(comparison)
    (comparison / "plots").mkdir()
    sentinel = comparison / "plots/pca.png"
    sentinel.write_bytes(b"prior plot")
    before = _hashes(comparison / "plots")
    with pytest.raises(FileExistsError):
        _plot_comparison()(comparison)
    assert _hashes(comparison / "plots") == before


def test_mismatched_donor_ids_are_rejected_before_creating_outputs(tmp_path):
    comparison = tmp_path / "misaligned"
    _write_comparison(comparison)
    matrix = pd.read_csv(comparison / "vst_expression.csv")
    matrix.rename(columns={"donor_0": "unexpected_donor"}).to_csv(comparison / "vst_expression.csv", index=False)
    with pytest.raises(ValueError, match="sample|donor"):
        _plot_comparison()(comparison)
    assert not (comparison / "plots").exists()


def test_skipped_comparison_requires_no_expression_files(tmp_path):
    comparison = tmp_path / "skipped"
    comparison.mkdir()
    (comparison / "status.json").write_text(json.dumps({"status": "skipped", "reason": "non_estimable_design"}))
    manifest = _plot_comparison()(comparison)
    assert manifest["status"] == "skipped"
    assert manifest["reason"] == "non_estimable_design"
    assert manifest["sample_unit"] == "donor"
    assert list((comparison / "plots").iterdir()) == [comparison / "plots/plot_manifest.json"]

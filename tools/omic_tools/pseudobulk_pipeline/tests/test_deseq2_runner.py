"""Behavioral tests for the real R runner; no substitute DE implementation."""

import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "deseq2_analysis.R"
COMPARISONS = ("ad_vs_control", "ad_male_vs_female", "control_male_vs_female")


@pytest.fixture(scope="module")
def rscript():
    executable = os.environ.get("PSEUDOBULK_RSCRIPT") or shutil.which("Rscript")
    if not executable:
        pytest.skip("Set PSEUDOBULK_RSCRIPT to Rscript with jsonlite and DESeq2")
    probe = subprocess.run(
        [executable, "--vanilla", "-e", 'quit(status=if(requireNamespace("jsonlite",quietly=TRUE)) 0 else 1)'],
        text=True, capture_output=True, timeout=60,
    )
    if probe.returncode:
        pytest.skip("R/jsonlite unavailable: " + probe.stderr)
    return executable


@pytest.fixture(scope="module")
def has_deseq2(rscript):
    probe = subprocess.run(
        [rscript, "--vanilla", "-e", 'quit(status=if(requireNamespace("DESeq2",quietly=TRUE)) 0 else 1)'],
        text=True, capture_output=True, timeout=120,
    )
    return probe.returncode == 0


def cohort(n_each=6, n_genes=300):
    """Balanced donors, independent age, known positive AD and male effects."""
    rng = np.random.default_rng(1031)
    disease = np.repeat(["control", "AD"], n_each * 2)
    sex = np.tile(np.repeat(["female", "male"], n_each), 2)
    n = len(disease)
    ids = [f"donor_{i:02d}" for i in range(n)]
    age = np.tile([60, 68, 73, 65, 78, 82], n // 6 + 1)[:n]
    metadata = pd.DataFrame({
        "sample_id": ids, "source": "synthetic", "dataset_id": "dataset",
        "donor_id": ids, "study": "one_study", "disease": disease,
        "sex": sex, "age": age, "n_metacells": 10,
        "sex_conflict": False, "age_conflict": False, "disease_conflict": False,
    })
    baseline = rng.uniform(100, 700, (n_genes, 1))
    library = np.tile([0.6, 0.8, 1.0, 1.2, 1.5, 1.8], n // 6 + 1)[:n]
    mean = baseline * library[None, :]
    mean[0:8, disease == "AD"] *= 8
    mean[8:16, sex == "male"] *= 8
    counts = rng.negative_binomial(20, 20 / (20 + mean))
    counts[-1, :] = 0
    counts[-2, :] = 1
    frame = pd.DataFrame(counts, index=[f"GENE_{i:04d}" for i in range(n_genes)], columns=ids)
    frame.index.name = "gene"
    metadata["library_size"] = frame.sum(axis=0).to_numpy()
    return frame, metadata


def invoke(tmp_path, rscript, counts, metadata, *, min_group=3, name="run"):
    case = tmp_path / name
    case.mkdir()
    counts_path = case / "counts.csv"
    metadata_path = case / "metadata.csv"
    counts.to_csv(counts_path)
    metadata.to_csv(metadata_path, index=False)
    output = case / "output"
    completed = subprocess.run(
        [rscript, "--vanilla", str(SCRIPT), str(counts_path), str(metadata_path),
         str(output), "0.05", "10", "3", str(min_group)],
        text=True, capture_output=True, timeout=180,
    )
    statuses = {
        comparison: json.loads((output / comparison / "status.json").read_text())
        for comparison in COMPARISONS
    }
    return output, statuses, completed


def test_confounded_study_is_reported_without_simplifying_model(tmp_path, rscript):
    counts, metadata = cohort()
    metadata["study"] = metadata["disease"]
    output, statuses, _ = invoke(tmp_path, rscript, counts, metadata)
    status = statuses["ad_vs_control"]
    assert status["status"] == "skipped"
    assert status["reason_code"] == "design_not_full_rank"
    assert "study" in status["formula"]
    assert status["design"]["rank"] < status["design"]["n_columns"]
    assert not (output / "ad_vs_control" / "results.csv").exists()


@pytest.mark.parametrize("value", [-1, 1.5, float("inf"), 2147483648])
def test_invalid_counts_produce_explicit_errors_for_every_comparison(tmp_path, rscript, value):
    counts, metadata = cohort()
    counts = counts.astype(float)
    counts.iloc[0, 0] = value
    _, statuses, completed = invoke(tmp_path, rscript, counts, metadata)
    assert completed.returncode != 0
    assert all(s["status"] == "error" for s in statuses.values())
    assert all(s["reason_code"] == "invalid_counts" for s in statuses.values())


def test_count_metadata_sample_mismatch_is_rejected(tmp_path, rscript):
    counts, metadata = cohort()
    metadata.loc[0, "sample_id"] = "unmatched_donor"
    _, statuses, completed = invoke(tmp_path, rscript, counts, metadata)
    assert completed.returncode != 0
    assert statuses["ad_vs_control"]["reason_code"] == "sample_mismatch"


def test_insufficient_donors_after_conflict_and_zero_library_exclusions(tmp_path, rscript):
    counts, metadata = cohort(n_each=3)
    metadata.loc[9, "sex_conflict"] = True
    metadata.loc[10, "age"] = np.nan
    counts.iloc[:, 11] = 0
    metadata["library_size"] = counts.sum(axis=0).to_numpy()
    _, statuses, _ = invoke(tmp_path, rscript, counts, metadata)
    status = statuses["ad_male_vs_female"]
    assert status["status"] == "skipped"
    assert status["reason_code"] == "insufficient_group_donors"
    exclusions = {item["sample_id"]: item["reasons"] for item in status["excluded_donors"]}
    assert "sex_conflict" in exclusions["donor_09"]
    assert "missing_age" in exclusions["donor_10"]
    assert "zero_library" in exclusions["donor_11"]
    assert not {"donor_09", "donor_10", "donor_11"} & set(status["included_donors"])


def test_saturated_design_is_not_fitted(tmp_path, rscript):
    counts, metadata = cohort(n_each=3)
    metadata = metadata.iloc[:6].copy()
    counts = counts.loc[:, metadata["sample_id"]]
    metadata["study"] = ["s0", "s1", "s2", "s3", "s0", "s1"]
    metadata["age"] = [60, 62, 67, 65, 79, 71]
    _, statuses, _ = invoke(tmp_path, rscript, counts, metadata)
    status = statuses["control_male_vs_female"]
    assert status["status"] == "skipped"
    assert status["reason_code"] == "no_residual_degrees_of_freedom"
    assert status["design"]["residual_df"] == 0


def test_refuses_existing_outputs_and_does_not_drop_missing_age(tmp_path, rscript):
    counts, metadata = cohort()
    metadata["age"] = np.nan
    output, statuses, completed = invoke(tmp_path, rscript, counts, metadata)
    assert completed.returncode == 0
    assert all(s["reason_code"] == "no_complete_case_donors" for s in statuses.values())
    saved = {path: path.read_bytes() for path in output.rglob("*") if path.is_file()}
    rerun = subprocess.run(
        [rscript, "--vanilla", str(SCRIPT), str(output.parent / "counts.csv"),
         str(output.parent / "metadata.csv"), str(output), "0.05", "10", "3", "3"],
        text=True, capture_output=True, timeout=60,
    )
    assert rerun.returncode != 0
    assert {path: path.read_bytes() for path in output.rglob("*") if path.is_file()} == saved


@pytest.fixture(scope="module")
def fitted_run(tmp_path_factory, rscript, has_deseq2):
    if not has_deseq2:
        pytest.skip("Real DESeq2 runtime required")
    counts, metadata = cohort()
    # Deliberately shuffled metadata tests alignment by sample ID.
    metadata = metadata.sample(frac=1, random_state=31)
    output, statuses, completed = invoke(tmp_path_factory.mktemp("deseq2_real"), rscript, counts, metadata)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert all(s["status"] == "success" for s in statuses.values()), statuses
    return counts, metadata, output, statuses


def test_explicit_contrast_directions_and_complete_results(fitted_run):
    counts, _, output, statuses = fitted_run
    for comparison, marker in zip(COMPARISONS, ["GENE_0000", "GENE_0008", "GENE_0008"]):
        status = statuses[comparison]
        results = pd.read_csv(output / comparison / "results.csv", keep_default_na=False).set_index("gene")
        assert list(results.index) == list(counts.index)
        assert float(results.loc[marker, "log2FoldChange"]) > 2
        assert float(results.loc[marker, "padj"]) < 0.05
        assert results.loc["GENE_0299", "filter_reason"] == "all_zero"
        assert results.loc["GENE_0298", "filter_reason"] == "low_count"
        assert results.loc["GENE_0299", "pvalue"] == ""
        assert not results.loc["GENE_0299", "tested"]
        assert results.loc[marker, "tested"]
        assert status["contrast"]["numerator"] == ("AD" if comparison == "ad_vs_control" else "male")
        assert status["contrast"]["reference"] == ("control" if comparison == "ad_vs_control" else "female")
        assert status["normalization"]["type"] == "ratio"
        assert status["alpha"] == 0.05
        assert "study" in status["dropped_constant_terms"]
        assert (output / comparison / "fit.log").is_file()


def test_size_factors_equal_median_of_ratios_and_exports_align(fitted_run):
    counts, _, output, statuses = fitted_run
    for comparison in COMPARISONS:
        directory = output / comparison
        donors = pd.read_csv(directory / "donor_metadata.csv")["sample_id"].tolist()
        subset = counts.loc[:, donors]
        retained = subset.loc[(subset >= 10).sum(axis=1) >= 3]
        positive = retained.loc[(retained > 0).all(axis=1)].to_numpy(dtype=float)
        expected = np.exp(np.median(np.log(positive) - np.log(positive).mean(axis=1)[:, None], axis=0))
        factors = pd.read_csv(directory / "size_factors.csv").set_index("sample_id")
        np.testing.assert_allclose(factors.loc[donors, "size_factor"], expected, rtol=1e-10)
        normalized = pd.read_csv(directory / "normalized_counts.csv", index_col="gene")
        np.testing.assert_allclose(normalized, subset / expected, rtol=1e-10)
        vst = pd.read_csv(directory / "vst_expression.csv", index_col="gene")
        assert list(vst.index) == list(retained.index)
        assert list(vst.columns) == donors == statuses[comparison]["included_donors"]
        assert np.isfinite(vst.to_numpy()).all()
        design = pd.read_csv(directory / "design_matrix.csv")
        assert design["sample_id"].tolist() == donors


def test_ratio_normalization_is_not_replaced_when_every_gene_has_zero(tmp_path, rscript, has_deseq2):
    if not has_deseq2:
        pytest.skip("Real DESeq2 runtime required")
    counts, metadata = cohort()
    for gene_index in range(len(counts)):
        counts.iloc[gene_index, gene_index % counts.shape[1]] = 0
    metadata["library_size"] = counts.sum(axis=0).to_numpy()
    output, statuses, _ = invoke(tmp_path, rscript, counts, metadata)
    status = statuses["ad_vs_control"]
    assert status["status"] == "skipped"
    assert status["reason_code"] == "ratio_normalization_unavailable"
    assert status["normalization"]["type"] == "ratio"
    assert not (output / "ad_vs_control" / "results.csv").exists()

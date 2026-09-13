"""Real DESeq2 tests for controlled donor sensitivity comparisons."""

import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "models.R"


@pytest.fixture(scope="module")
def rscript():
    executable = os.environ.get("PSEUDOBULK_RSCRIPT") or shutil.which("Rscript")
    if not executable:
        pytest.skip("Set PSEUDOBULK_RSCRIPT to the isolated DESeq2 Rscript")
    check = subprocess.run(
        [executable, "--vanilla", "-e",
         'quit(status=if(requireNamespace("DESeq2",quietly=TRUE)&&requireNamespace("jsonlite",quietly=TRUE)) 0 else 1)'],
        text=True, capture_output=True, timeout=90,
    )
    if check.returncode:
        pytest.skip("Real DESeq2/jsonlite runtime unavailable: " + check.stderr)
    return executable


def fixture_source(root):
    """Two shared studies, a control-only study, and an AD-only study."""
    rng = np.random.default_rng(817)
    n = 48
    samples = [f"donor_{i:02d}" for i in range(n)]
    study = np.repeat(["shared_1", "shared_2", "only_control", "only_AD"], 12)
    disease = np.array((["control"] * 6 + ["AD"] * 6) * 2 + ["control"] * 12 + ["AD"] * 12)
    sex = np.tile(["female", "male"], 24)
    age = np.tile([61, 70, 66, 78, 74, 82], 8)
    meta = pd.DataFrame({
        "sample_id": samples, "source": "fixture", "dataset_id": study,
        "donor_id": samples, "study": study, "disease": disease, "sex": sex,
        "age": age, "n_metacells": 10, "sex_conflict": False,
        "age_conflict": False, "disease_conflict": False,
    })
    baseline = rng.uniform(120, 800, (240, 1))
    scale = np.tile([0.6, 0.8, 1.0, 1.2, 1.5, 1.9], 8)
    means = baseline * scale[None, :]
    means[:8, disease == "AD"] *= 8
    means[8:16, study == "only_AD"] *= 10
    counts = rng.negative_binomial(20, 20 / (20 + means))
    counts[-1, :] = 0
    counts[-2, :] = 1
    counts[-3, :] = np.where(study == "only_control", 100, 0)
    matrix = pd.DataFrame(counts, index=[f"GENE_{i:04d}" for i in range(240)], columns=samples)
    matrix.index.name = "gene"
    meta["library_size"] = matrix.sum(axis=0).to_numpy()
    # The source comparison's order differs from count CSV order.
    meta = meta.sample(frac=1, random_state=98).reset_index(drop=True)
    included = matrix.loc[:, meta.sample_id]
    retained = included.loc[(included >= 10).sum(axis=1) >= 3]
    positive = retained.loc[(retained > 0).all(axis=1)].to_numpy(dtype=float)
    factors = np.exp(np.median(np.log(positive) - np.log(positive).mean(axis=1)[:, None], axis=0))
    (root / "inputs").mkdir(parents=True)
    comparison = root / "comparisons/ad_vs_control"
    comparison.mkdir(parents=True)
    # One additional source donor must not leak into any sensitivity model.
    source_counts = matrix.assign(excluded_extra=100)
    source_counts.to_csv(root / "inputs/pseudobulk_counts.csv")
    meta.to_csv(comparison / "donor_metadata.csv", index=False)
    pd.DataFrame({"sample_id": meta.sample_id, "size_factor": factors}).to_csv(comparison / "size_factors.csv", index=False)
    (comparison / "status.json").write_text(json.dumps({"status": "success", "contrast": {
        "column": "disease", "numerator": "AD", "reference": "control"}}))
    return matrix, meta, factors


def run_models(rscript, source, output):
    completed = subprocess.run(
        [rscript, "--vanilla", str(SCRIPT), str(source), str(output), "0.05", "10", "3", "3"],
        text=True, capture_output=True, timeout=240,
    )
    manifest = json.loads((output / "manifest.json").read_text())
    return manifest, completed


def test_fixed_bh_family_penalizes_unavailable_and_nonconverged_tests(rscript):
    code = (
        f"source({json.dumps(str(SCRIPT))});"
        'cat(jsonlite::toJSON(fixed_family_bh(c(0.01,0.001,NA,0.2),c(TRUE,FALSE,FALSE,TRUE)),na="null"))'
    )
    completed = subprocess.run([rscript, "--vanilla", "-e", code], text=True, capture_output=True, timeout=30)
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == [0.04, None, None, 0.4]


def test_source_effect_validation_separates_nonconverged_estimates(rscript):
    code = (
        f"source({json.dumps(str(SCRIPT))});"
        'old<-data.frame(gene=c("A","B"),log2FoldChange=c(1,100),filter_reason=c("","nonconverged"));'
        'new<-data.frame(gene=c("A","B"),log2FoldChange=c(1,5),beta_converged=c(TRUE,FALSE));'
        'cat(jsonlite::toJSON(compare_source_effects(new,old,"fixture"),auto_unbox=TRUE))'
    )
    completed = subprocess.run([rscript, "--vanilla", "-e", code], text=True, capture_output=True, timeout=30)
    assert completed.returncode == 0, completed.stderr
    summary = json.loads(completed.stdout)
    assert summary["max_abs_lfc_difference"] == 95
    assert summary["n_converged_compared_genes"] == 1
    assert summary["max_abs_lfc_difference_converged"] == 0
    assert summary["within_tolerance_converged"] is True


@pytest.fixture(scope="module")
def fitted(tmp_path_factory, rscript):
    case = tmp_path_factory.mktemp("sensitivity_models")
    counts, metadata, factors = fixture_source(case / "source")
    manifest, completed = run_models(rscript, case / "source", case / "output")
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert manifest["status"] == "success", manifest
    return case, counts, metadata, factors, manifest


def test_full_models_hold_donors_genes_and_original_size_factors_fixed(fitted):
    case, counts, metadata, factors, manifest = fitted
    expected_genes = counts.index[(counts >= 10).sum(axis=1) >= 3].tolist()
    assert manifest["model_order"] == ["full_A", "full_B", "full_C", "shared_C", "leave_one_out_01", "leave_one_out_02"]
    exported_factor_files = []
    for name in ["full_A", "full_B", "full_C"]:
        directory = case / "output/models" / name
        donors = pd.read_csv(directory / "donor_metadata.csv")
        sizes = pd.read_csv(directory / "size_factors.csv")
        results = pd.read_csv(directory / "results.csv")
        assert donors.sample_id.tolist() == metadata.sample_id.tolist() == sizes.sample_id.tolist()
        # R CSV serializes approximately 15 significant digits; factors are
        # numerically preserved from the source and identical across models.
        np.testing.assert_allclose(sizes.size_factor, factors, rtol=1e-12)
        exported_factor_files.append((directory / "size_factors.csv").read_bytes())
        assert results.loc[results.in_count_filter, "gene"].tolist() == expected_genes
        assert results.gene.tolist() == counts.index.tolist()
        assert manifest["models"][name]["normalization"]["source"] == "original_full_cohort_factors"
        assert manifest["models"][name]["normalization"]["original_factors_validated"] is True
        assert manifest["models"][name]["filter"]["independent_filtering"] is False
    assert len(set(exported_factor_files)) == 1


def test_model_directions_fixed_bh_and_wald_intervals(fitted):
    case, _, _, _, manifest = fitted
    z95 = 1.959963984540054
    for name in manifest["model_order"]:
        status = manifest["models"][name]
        assert status["contrast"]["numerator"] == "AD"
        assert status["contrast"]["reference"] == "control"
        result = pd.read_csv(case / "output/models" / name / "results.csv").set_index("gene")
        assert result.loc["GENE_0000", "log2FoldChange"] > 2
        assert result.loc["GENE_0000", "padj"] < 0.05
        assert result.loc["GENE_0239", "filter_reason"] == "all_zero"
        assert result.loc["GENE_0238", "filter_reason"] == "low_count"
        assert result.loc[~result.tested, "padj"].isna().all()
        assert result.loc[result.tested, "beta_converged"].all()
        np.testing.assert_allclose(
            result.loc[result.tested, "lfc_ci_low"],
            result.loc[result.tested, "log2FoldChange"] - z95 * result.loc[result.tested, "lfcSE"], rtol=1e-10,
        )
        retained = result.loc[result.in_count_filter]
        values = retained.pvalue_for_bh.to_numpy()
        order = np.argsort(values)
        adjusted = np.minimum.accumulate((values[order] * len(values) / np.arange(1, len(values) + 1))[::-1])[::-1].clip(0, 1)
        expected = np.empty_like(adjusted)
        expected[order] = adjusted
        np.testing.assert_allclose(retained.loc[retained.tested, "padj"], expected[retained.tested], rtol=1e-10)


def test_shared_and_leave_one_study_out_models_use_only_eligible_studies(fitted):
    case, counts, metadata, _, manifest = fitted
    assert manifest["shared_studies"] == ["shared_1", "shared_2"]
    support = pd.read_csv(case / "output/cohort_support.csv")
    assert support.groupby("study").n_donors.sum().to_dict() == {s: 12 for s in metadata.study.unique()}
    for name in ["shared_C", "leave_one_out_01", "leave_one_out_02"]:
        status = manifest["models"][name]
        donors = pd.read_csv(case / "output/models" / name / "donor_metadata.csv")
        expected = metadata.loc[metadata.study.isin(["shared_1", "shared_2"])]
        if status["omitted_study"]:
            expected = expected.loc[expected.study != status["omitted_study"]]
        assert donors.sample_id.tolist() == expected.sample_id.tolist()
        subset = counts.loc[:, donors.sample_id]
        retained = subset.loc[(subset >= 10).sum(axis=1) >= 3]
        results = pd.read_csv(case / "output/models" / name / "results.csv")
        assert results.loc[results.in_count_filter, "gene"].tolist() == retained.index.tolist()
        assert not results.set_index("gene").loc["GENE_0237", "in_count_filter"]
        positive = retained.loc[(retained > 0).all(axis=1)].to_numpy(dtype=float)
        expected_factors = np.exp(np.median(np.log(positive) - np.log(positive).mean(axis=1)[:, None], axis=0))
        factors = pd.read_csv(case / "output/models" / name / "size_factors.csv")
        np.testing.assert_allclose(factors.size_factor, expected_factors, rtol=1e-10)
        assert status["normalization"]["source"] == "reestimated_subset_ratio_factors"


def test_confounded_adjusted_models_skip_without_removing_study(tmp_path, rscript):
    _, metadata, _ = fixture_source(tmp_path / "source")
    metadata["study"] = metadata["disease"]
    metadata.to_csv(tmp_path / "source/comparisons/ad_vs_control/donor_metadata.csv", index=False)
    manifest, _ = run_models(rscript, tmp_path / "source", tmp_path / "output")
    for name in ["full_B", "full_C"]:
        status = manifest["models"][name]
        assert status["status"] == "skipped"
        assert status["reason_code"] == "design_not_full_rank"
        assert "study" in status["formula"]
    assert manifest["shared_studies"] == []
    assert manifest["models"]["shared_C"]["status"] == "skipped"


def test_original_size_factor_mismatch_aborts_controlled_comparison(tmp_path, rscript):
    fixture_source(tmp_path / "source")
    path = tmp_path / "source/comparisons/ad_vs_control/size_factors.csv"
    factors = pd.read_csv(path)
    factors.loc[0, "size_factor"] *= 1.5
    factors.to_csv(path, index=False)
    manifest, completed = run_models(rscript, tmp_path / "source", tmp_path / "output")
    assert completed.returncode != 0
    assert manifest["status"] == "error"
    assert manifest["reason_code"] == "original_size_factor_mismatch"
    assert not list((tmp_path / "output").rglob("results.csv"))


def test_missing_covariates_abort_without_changing_fixed_donor_set(tmp_path, rscript):
    _, metadata, _ = fixture_source(tmp_path / "source")
    metadata.loc[0, "age"] = np.nan
    metadata.to_csv(tmp_path / "source/comparisons/ad_vs_control/donor_metadata.csv", index=False)
    manifest, completed = run_models(rscript, tmp_path / "source", tmp_path / "output")
    assert completed.returncode != 0
    assert manifest["reason_code"] == "fixed_cohort_not_complete"
    assert manifest["details"] == [{"sample_id": metadata.loc[0, "sample_id"], "reasons": ["missing_age"]}]
    assert not list((tmp_path / "output").rglob("results.csv"))


def test_existing_output_is_preserved(fitted, rscript):
    case, _, _, _, _ = fitted
    manifest_path = case / "output/manifest.json"
    saved = manifest_path.read_bytes()
    _, completed = run_models(rscript, case / "source", case / "output")
    assert completed.returncode != 0
    assert manifest_path.read_bytes() == saved

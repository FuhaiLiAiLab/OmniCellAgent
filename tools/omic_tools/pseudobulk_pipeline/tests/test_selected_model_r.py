"""Real R regression checks: selection must bound actual DESeq2 execution."""

import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "selected_model.R"
MODELS = SCRIPT.parent.parent / "donor_sensitivity/models.R"


@pytest.fixture(scope="module")
def rscript():
    executable = os.environ.get("PSEUDOBULK_RSCRIPT") or shutil.which("Rscript")
    if not executable:
        pytest.skip("Set PSEUDOBULK_RSCRIPT to the isolated DESeq2 runtime")
    check = subprocess.run([executable, "--vanilla", "-e",
                            'quit(status=if(requireNamespace("DESeq2",quietly=TRUE)&&requireNamespace("jsonlite",quietly=TRUE)) 0 else 1)'],
                           text=True, capture_output=True, timeout=90)
    if check.returncode:
        pytest.skip("DESeq2/jsonlite unavailable: " + check.stderr)
    return executable


def raw_source(root):
    rng = np.random.default_rng(817)
    n = 48
    samples = [f"donor_{i:02d}" for i in range(n)]
    studies = np.repeat(["shared_1", "shared_2", "only_control", "only_AD"], 12)
    disease = np.array((["control"] * 6 + ["AD"] * 6) * 2 + ["control"] * 12 + ["AD"] * 12)
    means = rng.uniform(120, 800, (240, 1)) * np.tile([0.6, 0.8, 1, 1.2, 1.5, 1.9], 8)
    means[:8, disease == "AD"] *= 8
    values = rng.negative_binomial(20, 20 / (20 + means))
    values[-1] = 0
    values[-2] = 1
    values[-3] = np.where(studies == "only_control", 100, 0)
    counts = pd.DataFrame(values, index=[f"GENE_{i:04d}" for i in range(240)], columns=samples)
    counts.index.name = "gene"
    metadata = pd.DataFrame({"sample_id": samples, "source": "fixture", "dataset_id": studies,
                             "donor_id": samples, "study": studies, "disease": disease,
                             "sex": np.tile(["female", "male"], 24),
                             "age": np.tile([61, 70, 66, 78, 74, 82], 8), "n_metacells": 10,
                             "sex_conflict": False, "age_conflict": False, "disease_conflict": False,
                             "library_size": counts.sum(axis=0).to_numpy()})
    # Disease-only must retain the original complete-case exclusions.
    metadata.loc[24, "age"] = np.nan
    metadata.loc[25, "sex_conflict"] = True
    metadata.loc[36, "study"] = ""
    metadata = metadata.sample(frac=1, random_state=98)
    (root / "inputs").mkdir(parents=True)
    counts.to_csv(root / "inputs/pseudobulk_counts.csv")
    metadata.to_csv(root / "inputs/donor_metadata.csv", index=False)
    return counts, metadata


def invoke(rscript, source, output, model="full_A", export=False, baseline=None):
    calls = output.parent / (output.name + "_calls.txt")
    args = [str(source), str(output), model, "0.05", "10", "3", "3", str(export).lower()]
    # Trace the real namespace functions; neither call is mocked or replaced.
    code = f'''
    calls <- {json.dumps(str(calls))}
    check_full_factors <- {str(model.startswith('full_')).upper()}
    exported_factors <- {json.dumps(str(output / 'cohort/size_factors.csv'))}
    trace("DESeq", where=asNamespace("DESeq2"), tracer=quote({{
      cat("DESeq\\n", file=calls, append=TRUE)
      if (check_full_factors) {{
        saved <- utils::read.csv(exported_factors)$size_factor
        stopifnot(identical(unname(DESeq2::sizeFactors(object)), unname(saved)))
      }}
    }}), print=FALSE)
    trace("varianceStabilizingTransformation", where=asNamespace("DESeq2"), tracer=quote(cat("VST\\n", file=calls, append=TRUE)), print=FALSE)
    env <- new.env()
    source({json.dumps(str(SCRIPT))}, local=env)
    result <- env$main(unlist(jsonlite::fromJSON({json.dumps(json.dumps(args))})))
    untrace("DESeq", where=asNamespace("DESeq2"))
    untrace("varianceStabilizingTransformation", where=asNamespace("DESeq2"))
    '''
    if baseline is not None:
        # Reconstruct the historical sensitivity source using the exact same
        # fixed donor cohort and independent NumPy factors, then fit its model.
        code += f'''
        old <- new.env()
        sys.source({json.dumps(str(MODELS))}, envir=old)
        settings <- list(alpha=.05, min_count=10, min_gene_donors=3, min_group_donors=3)
        source <- old$read_source({json.dumps(str(baseline))}, settings)
        definitions <- old$model_specs(source$metadata)
        spec <- Filter(function(x) x$name == {json.dumps(model)}, definitions$models)[[1L]]
        status <- old$fit_model(spec, source, definitions$shared_studies, settings, {json.dumps(str(baseline / 'fitted'))})
        stopifnot(status$status == "success")
        '''
    code += "quit(status=result)"
    completed = subprocess.run([rscript, "--vanilla", "-e", code], text=True, capture_output=True, timeout=240)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    observed = calls.read_text().splitlines() if calls.exists() else []
    return completed, manifest, observed


@pytest.mark.parametrize("model,export", [("full_A", False), ("full_B", True), ("shared_C", False), ("leave_one_out_01", False)])
def test_selected_fit_matches_historical_model_and_executes_once(tmp_path, rscript, model, export):
    counts, metadata = raw_source(tmp_path / "source")
    donors = [name for name in counts.columns if name not in {"donor_24", "donor_25", "donor_36"}]
    included = metadata.set_index("sample_id").loc[donors].reset_index()
    matrix = counts.loc[:, donors]
    retained = matrix.loc[(matrix >= 10).sum(axis=1) >= 3]
    positive = retained.loc[(retained > 0).all(axis=1)].to_numpy(dtype=float)
    expected_factors = np.exp(np.median(np.log(positive) - np.log(positive).mean(axis=1)[:, None], axis=0))
    baseline = tmp_path / "baseline"
    comparison = baseline / "comparisons/ad_vs_control"
    comparison.mkdir(parents=True)
    (baseline / "inputs").mkdir()
    counts.to_csv(baseline / "inputs/pseudobulk_counts.csv")
    included.to_csv(comparison / "donor_metadata.csv", index=False)
    pd.DataFrame({"sample_id": donors, "size_factor": expected_factors}).to_csv(comparison / "size_factors.csv", index=False)
    output = tmp_path / "selected"
    completed, manifest, calls = invoke(rscript, tmp_path / "source", output, model, export, baseline)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert manifest["status"] == "success", manifest
    assert calls == (["DESeq", "VST"] if export else ["DESeq"])
    assert manifest["model_order"] == [model]
    assert list(manifest["models"]) == [model]
    assert manifest["fitted_model_count"] == 1
    assert manifest["full_cohort"]["n_donors"] == 45
    assert manifest["shared_cohort"]["n_donors"] == 24
    assert sorted(p.name for p in (output / "models").iterdir()) == [model]
    audit = pd.read_csv(output / "cohort/donor_exclusions.csv").set_index("sample_id")
    assert audit.loc["donor_24", "exclusion_reasons"] == "missing_age"
    assert audit.loc["donor_25", "exclusion_reasons"] == "sex_conflict"
    assert audit.loc["donor_36", "exclusion_reasons"] == "missing_study"
    assert audit.loc[audit.included].index.tolist() == donors
    np.testing.assert_allclose(pd.read_csv(output / "cohort/size_factors.csv").size_factor, expected_factors, rtol=1e-10)
    directory = output / "models" / model
    for name in ("results.csv", "size_factors.csv", "donor_metadata.csv"):
        # Historical factors cross a CSV boundary; rounding at ~15 digits can
        # perturb DESeq2's iterative optimizer below its 1e-6 effect tolerance.
        pd.testing.assert_frame_equal(pd.read_csv(directory / name), pd.read_csv(baseline / "fitted" / name), rtol=1e-6, atol=1e-7)
    if model.startswith("full_"):
        assert manifest["models"][model]["normalization"]["source"] == "estimated_full_cohort_ratio_factors"
    assert (directory / "normalized_counts.csv").exists() == export
    assert (directory / "vst_expression.csv").exists() == export
    if export:
        factors = pd.read_csv(directory / "size_factors.csv").set_index("sample_id").size_factor
        normalized = pd.read_csv(directory / "normalized_counts.csv", index_col="gene")
        np.testing.assert_allclose(normalized, counts.loc[:, factors.index] / factors, rtol=1e-10)
        vst = pd.read_csv(directory / "vst_expression.csv", index_col="gene")
        assert vst.columns.tolist() == donors
        assert np.isfinite(vst.to_numpy()).all()
    before = (output / "manifest.json").read_bytes()
    rerun, _, _ = invoke(rscript, tmp_path / "source", output, model)
    assert rerun.returncode != 0
    assert (output / "manifest.json").read_bytes() == before


@pytest.mark.parametrize("case", ["invalid_counts", "confounded", "unknown_model"])
def test_invalid_or_unidentifiable_model_never_fits(tmp_path, rscript, case):
    counts, metadata = raw_source(tmp_path / "source")
    model = "full_B"
    if case == "invalid_counts":
        counts = counts.astype(float)
        counts.iloc[0, 0] = 1.5
        counts.to_csv(tmp_path / "source/inputs/pseudobulk_counts.csv")
    elif case == "confounded":
        metadata["study"] = metadata.disease
        metadata.to_csv(tmp_path / "source/inputs/donor_metadata.csv", index=False)
    else:
        model = "leave_one_out_99"
    completed, manifest, calls = invoke(rscript, tmp_path / "source", tmp_path / "selected", model)
    assert calls == []
    assert manifest["fitted_model_count"] == 0
    if case == "confounded":
        status = manifest["models"][model]
        assert status["status"] == "skipped"
        assert status["reason_code"] == "design_not_full_rank"
        assert "study" in status["formula"]
    else:
        assert completed.returncode != 0
        assert manifest["reason_code"] == case

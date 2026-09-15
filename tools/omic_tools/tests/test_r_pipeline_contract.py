"""Exercise the real R CLI and compare with the existing panels Rmd DE.

Run inside a Slurm allocation with Rscript, limma, statmod and data.table.
Missing R dependencies are failures, not silently skipped acceptance checks.
"""
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import pytest

OMIC = Path(__file__).resolve().parents[1]
RUNNER = OMIC / "run_casestudy.R"
COLS = ["Name", "log2_fold_change", "effect_size", "p_value", "FDR",
        "is_significant", "abs_log2_fc"]
FILES = ["unpaired_differential_expression_results.csv",
         "significant_genes_by_fdr.csv", "significant_genes_by_fc.csv",
         "significant_upregulated_genes.csv", "significant_downregulated_genes.csv"]


@pytest.fixture
def cohort(tmp_path):
    """Known positive contrast plus one zero-variance gene, on CP10K scale."""
    root = tmp_path / "patient's cohort"
    root.mkdir()
    rng = np.random.default_rng(20260915)
    ref = rng.poisson(20, (40, 8)).astype(float)
    alt = rng.poisson(20, (40, 8)).astype(float)
    alt[:5] += 40
    ref[-1] = alt[-1] = 0
    for matrix, name in [(ref, "reference"), (alt, "alternate")]:
        matrix *= 10000 / matrix.sum(axis=0)
        frame = pd.DataFrame(matrix, columns=[f"{name}_{i}" for i in range(8)])
        frame.insert(0, "Name", [f"G{i}" for i in range(1, 41)])
        frame.to_csv(root / f"{name}.csv", index=False)
    return root


def run_de(root, *options, output="result"):
    return subprocess.run(
        ["Rscript", str(RUNNER), "--stage", "de", str(root), str(root / output),
         "--ref-csv", str(root / "reference.csv"),
         "--alt-csv", str(root / "alternate.csv"), *map(str, options)],
        capture_output=True, text=True, timeout=120,
    )


def read_main(root):
    return pd.read_csv(root / "differential_expression" / FILES[0], comment="#")


def test_de_stage_matches_existing_r_and_python_contract(cohort):
    result = run_de(cohort)
    assert result.returncode == 0, result.stdout + result.stderr
    reference = subprocess.run(
        ["Rscript", str(OMIC / "tests/fixtures/existing_panels_de.R"),
         str(OMIC / "casestudy_panels.Rmd"), str(cohort), str(cohort / "original")],
        capture_output=True, text=True, timeout=120,
    )
    assert reference.returncode == 0, reference.stdout + reference.stderr
    old = pd.read_csv(cohort / "original/DE_results_table.csv").set_index("Gene").sort_index()
    new = pd.read_csv(cohort / "result/DE_results_table.csv").set_index("Gene").sort_index()
    assert new.index.equals(old.index)
    for col in ["logFC", "P.Value", "adj.P.Val"]:
        np.testing.assert_allclose(new[col], old[col], rtol=1e-8, atol=0)
    main = read_main(cohort)
    assert main["Name"].tolist() == [f"G{i}" for i in range(1, 40)]
    assert main["is_significant"].dtype == bool
    assert main["is_significant"].equals(main["FDR"] < 0.05)
    np.testing.assert_allclose(main["abs_log2_fc"], main["log2_fold_change"].abs())
    mapped = main.set_index("Name").sort_index()
    for compat_col, native_col in [("log2_fold_change", "logFC"), ("p_value", "P.Value"), ("FDR", "adj.P.Val")]:
        np.testing.assert_allclose(mapped[compat_col], new[native_col], rtol=1e-12, atol=0)
    assert (main.set_index("Name").loc[["G1", "G2", "G3", "G4", "G5"], "log2_fold_change"] > 0).all()
    for filename in FILES:
        assert pd.read_csv(cohort / "differential_expression" / filename, comment="#").columns.tolist() == COLS
    assert (cohort / "result/analysis_state.rds").is_file()
    assert not list(cohort.rglob("*.png"))


def test_effect_size_keeps_original_linear_cohens_d(cohort):
    result = run_de(cohort)
    assert result.returncode == 0, result.stderr
    ref = pd.read_csv(cohort / "reference.csv").set_index("Name")
    alt = pd.read_csv(cohort / "alternate.csv").set_index("Name")
    expected = (alt.mean(axis=1) - ref.mean(axis=1)) / (
        np.sqrt((alt.var(axis=1, ddof=1) + ref.var(axis=1, ddof=1)) / 2) + 1e-8
    )
    result = read_main(cohort).set_index("Name")
    np.testing.assert_allclose(result["effect_size"], expected.loc[result.index], rtol=1e-10)


def test_top_n_only_limits_significant_exports_and_preserves_utf8(cohort):
    diagnostic = cohort / "diagnostics.txt"
    diagnostic.write_text("COHORT DIAGNOSTICS: CAUTION\n  样本来自疾病组\n", encoding="utf-8")
    result = run_de(cohort, "--top-n", 3, "--diagnostics-file", diagnostic)
    assert result.returncode == 0, result.stderr
    main = read_main(cohort)
    assert len(main) == 39
    assert main["is_significant"].sum() > 3
    subsets = {name: pd.read_csv(cohort / "differential_expression" / name, comment="#") for name in FILES[1:]}
    assert all(len(table) <= 3 for table in subsets.values())
    assert subsets[FILES[1]]["Name"].tolist() == main.loc[main.is_significant].sort_values("FDR").head(3)["Name"].tolist()
    assert (subsets[FILES[3]]["log2_fold_change"] > 0).all()
    assert (subsets[FILES[4]]["log2_fold_change"] < 0).all()
    text = (cohort / "differential_expression" / FILES[0]).read_text(encoding="utf-8")
    assert text.startswith("# COHORT DIAGNOSTICS: CAUTION\n#   样本来自疾病组\n")


def test_no_significant_genes_keeps_empty_csv_headers(cohort):
    result = run_de(cohort, "--de-fdr", 0)
    assert result.returncode == 0, result.stderr
    assert len(read_main(cohort)) == 39
    for filename in FILES[1:]:
        table = pd.read_csv(cohort / "differential_expression" / filename)
        assert table.empty
        assert table.columns.tolist() == COLS


@pytest.mark.parametrize("fdr", [0.05, 0], ids=["with_significant_genes", "without_significant_genes"])
def test_csv_reader_preserves_dtypes_for_all_five_results(cohort, fdr, monkeypatch):
    """A header-only R CSV must retain the numeric/bool Python contract."""
    monkeypatch.syspath_prepend(str(OMIC))
    from de_results_io import read_de_results

    expected_dtypes = {
        "Name": "object",
        "log2_fold_change": "float64",
        "effect_size": "float64",
        "p_value": "float64",
        "FDR": "float64",
        "is_significant": "bool",
        "abs_log2_fc": "float64",
    }
    diagnostic = cohort / "diagnostics.txt"
    diagnostic.write_text("COHORT DIAGNOSTICS: CAUTION\n  空结果类型检查\n", encoding="utf-8")
    result = run_de(cohort, "--de-fdr", fdr, "--diagnostics-file", diagnostic)
    assert result.returncode == 0, result.stdout + result.stderr
    for index, filename in enumerate(FILES):
        path = cohort / "differential_expression" / filename
        original = path.read_bytes()
        table = read_de_results(path)
        assert path.read_bytes() == original, "Reading must not rewrite the CSV"
        assert table.columns.tolist() == COLS
        actual_dtypes = table.dtypes.astype(str).to_dict()
        print(f"{filename}: rows={len(table)}, dtypes={actual_dtypes}")
        assert actual_dtypes == expected_dtypes
        if index == 0:
            assert len(table) == 39
        elif fdr == 0:
            assert table.empty
        else:
            assert not table.empty
        if len(table):
            expected_flags = table["FDR"].to_numpy() < fdr
            np.testing.assert_array_equal(table["is_significant"].to_numpy(), expected_flags)


def test_reversing_groups_reverses_fc_without_changing_pvalues(cohort):
    result = run_de(cohort)
    assert result.returncode == 0, result.stderr
    forward = read_main(cohort).set_index("Name").sort_index()
    ref = (cohort / "reference.csv").read_bytes()
    alt = (cohort / "alternate.csv").read_bytes()
    (cohort / "reference.csv").write_bytes(alt)
    (cohort / "alternate.csv").write_bytes(ref)
    result = run_de(cohort, "--ref-label", "Diseased", "--alt-label", "Healthy", output="reversed")
    assert result.returncode == 0, result.stderr
    reverse = read_main(cohort).set_index("Name").sort_index()
    np.testing.assert_allclose(reverse.log2_fold_change, -forward.log2_fold_change, rtol=1e-10)
    for column in ["p_value", "FDR"]:
        np.testing.assert_allclose(reverse[column], forward[column], rtol=1e-8, atol=0)


def test_already_log2_input_is_not_transformed_twice(cohort):
    result = run_de(cohort)
    assert result.returncode == 0, result.stderr
    linear = read_main(cohort).set_index("Name").sort_index()
    for name in ["reference", "alternate"]:
        path = cohort / f"{name}.csv"
        frame = pd.read_csv(path)
        frame.iloc[:, 1:] = np.log2(frame.iloc[:, 1:] + 1)
        frame.to_csv(path, index=False)
    result = run_de(cohort, "--input-scale", "log2_already", output="log-input")
    assert result.returncode == 0, result.stderr
    logged = read_main(cohort).set_index("Name").sort_index()
    for column in ["log2_fold_change", "effect_size", "p_value", "FDR"]:
        np.testing.assert_allclose(logged[column], linear[column], rtol=1e-8, atol=0)


def test_source_does_not_run_cli():
    result = subprocess.run(
        ["Rscript", "-e", "source(commandArgs(TRUE)[1]); stopifnot(is.function(run_de))", str(RUNNER)],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("invalid", ["scale", "duplicate", "one_cell", "group_names", "no_overlap", "output_is_file"])
def test_invalid_inputs_fail_before_export(cohort, invalid):
    options = []
    ref = pd.read_csv(cohort / "reference.csv")
    if invalid == "scale":
        ref.iloc[:, 1:] *= 0.1
    elif invalid == "duplicate":
        ref.loc[1, "Name"] = ref.loc[0, "Name"]
    elif invalid == "one_cell":
        ref = ref.iloc[:, :2]
    elif invalid == "no_overlap":
        ref["Name"] = [f"OTHER{i}" for i in range(len(ref))]
    elif invalid == "output_is_file":
        conflict = cohort / "output-file"
        conflict.write_text("keep this file", encoding="utf-8")
        options = ["--de-dir", str(conflict)]
    else:
        options = ["--ref-label", "same", "--alt-label", "same"]
    ref.to_csv(cohort / "reference.csv", index=False)
    result = run_de(cohort, *options)
    assert result.returncode != 0
    assert not (cohort / "differential_expression" / FILES[0]).exists()
    if invalid == "output_is_file":
        assert conflict.read_text(encoding="utf-8") == "keep this file"

"""Python -> real R DE -> typed results -> existing enrichment preparation."""
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

OMIC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OMIC))
sys.path.insert(0, str(OMIC.parents[1]))
import omic_analysis_components as analysis
from de_results_io import read_de_results


def seed_historical_results(directory):
    """Valid old CSVs would be readable if the adapter selected the wrong run."""
    directory.mkdir(parents=True, exist_ok=True)
    old = pd.DataFrame({"Name": ["HISTORICAL_ONLY"], "log2_fold_change": [1.0],
                        "effect_size": [0.5], "p_value": [0.001], "FDR": [0.01],
                        "is_significant": [True], "abs_log2_fc": [1.0]})
    snapshots = {}
    for filename in ("unpaired_differential_expression_results.csv", "significant_genes_by_fdr.csv",
                     "significant_genes_by_fc.csv", "significant_upregulated_genes.csv",
                     "significant_downregulated_genes.csv"):
        path = directory / filename
        old.to_csv(path, index=False)
        snapshots[path.resolve()] = path.read_bytes()
    return snapshots


def test_analysis_component_supports_package_import_from_repo_root():
    result = subprocess.run(
        [sys.executable, "-c", "from tools.omic_tools.omic_analysis_components import perform_unpaired_differential_expression; assert callable(perform_unpaired_differential_expression)"],
        cwd=OMIC.parents[1], capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr


@pytest.fixture
def frames():
    rng = np.random.default_rng(20260915)
    matrices = [rng.poisson(20, (40, 8)).astype(float) for _ in range(2)]
    matrices[1][:5] += 40
    output = []
    for matrix, prefix in zip(matrices, ["ref", "alt"]):
        matrix[-1] = 0
        matrix *= 10000 / matrix.sum(axis=0)
        frame = pd.DataFrame(matrix, columns=[f"{prefix}{i}" for i in range(8)])
        frame.insert(0, "Name", [f"G{i}" for i in range(1, 41)])
        output.append(frame)
    return output


@pytest.mark.parametrize("fdr", [0.05, 0], ids=["nonempty", "empty"])
def test_python_de_uses_r_result_and_typed_subsets(frames, tmp_path, fdr):
    ref, alt = frames
    subsets, table = analysis.perform_unpaired_differential_expression(
        alt, ref, p_value_threshold=fdr, sig_top_n=3, de_output_dir=str(tmp_path / "de"),
        session_dir=str(tmp_path), ref_label="Female", alt_label="Male", r_timeout=30,
    )
    native = pd.read_csv(tmp_path / "casestudy_R/DE_results_table.csv").set_index("Gene").sort_index()
    comparable = table.set_index("Name").sort_index()
    assert comparable.index.equals(native.index)
    assert len(table) == 39
    assert table["is_significant"].dtype == bool
    for field, r_field in [("log2_fold_change", "logFC"), ("p_value", "P.Value"), ("FDR", "adj.P.Val")]:
        np.testing.assert_allclose(comparable[field], native[r_field], rtol=1e-10, atol=0)
    assert set(subsets) == {"all", "up", "down"}
    for name, filename in {"all": "significant_genes_by_fdr.csv", "up": "significant_upregulated_genes.csv", "down": "significant_downregulated_genes.csv"}.items():
        pd.testing.assert_frame_equal(subsets[name], read_de_results(tmp_path / "de" / filename))
        assert subsets[name].dtypes.equals(table.dtypes)
        assert len(subsets[name]) <= 3
        assert subsets[name].empty if fdr == 0 else not subsets[name].empty


def test_analysis_feeds_real_r_genes_to_existing_enrichment_without_plots(frames, tmp_path, monkeypatch):
    ref, alt = frames
    calls = {}
    def remote_enrichr(genes, sample_id, enrich_output_dir=None, databases=None, request_timeout=60):
        calls[sample_id] = list(genes)
        return {}
    def unexpected_plot(*args, **kwargs):
        raise AssertionError("enable_plotting=False started a plot")
    monkeypatch.setattr(analysis, "enrichr_analysis", remote_enrichr)
    monkeypatch.setattr(analysis, "render_analysis_plots", unexpected_plot)
    output = analysis.omic_analysis(
        "test contrast", {"normal_omic_feature": ref.iloc[:, 1:].to_numpy().T,
                          "disease_omic_feature": alt.iloc[:, 1:].to_numpy().T},
        enable_plotting=False, session_dir=str(tmp_path), gene_names=ref.Name.tolist(),
        ref_label="Normal", alt_label="Alzheimer's Disease", r_timeout=30,
    )
    assert set(output) == {"data_dir", "differential_expression_dir", "volcano_plots_dir", "enrichment_results_dir", "enrichment_plots_dir", "enrichment_success", "enrichment_status", "de_plots_status", "de_plot_files", "de_plots_error"}
    for group, filename in {"all": "significant_genes_by_fdr.csv", "up": "significant_upregulated_genes.csv", "down": "significant_downregulated_genes.csv"}.items():
        expected = read_de_results(Path(output["differential_expression_dir"]) / filename).Name.tolist()
        assert calls[f"test_contrast_{group}_regulated"] == expected
    assert not list(tmp_path.rglob("*.png"))
    assert (tmp_path / "casestudy_R/analysis_state.rds").exists()


@pytest.mark.parametrize("script_body,error_type,timeout", [
    ('stop("R failed")', subprocess.CalledProcessError, 30),
    ('Sys.sleep(5)', subprocess.TimeoutExpired, 0.2),
    ('quit(status=0)', FileNotFoundError, 30),
], ids=["nonzero_exit", "timeout", "missing_outputs"])
def test_failed_or_missing_r_output_cannot_reuse_stale_results(frames, tmp_path, monkeypatch, script_body, error_type, timeout):
    ref, alt = frames
    script = tmp_path / "fake.R"
    script.write_text(script_body, encoding="utf-8")
    original_get_path = analysis.get_path
    def configured_path(key, **kwargs):
        return str(script) if key == "analysis.r_script" else original_get_path(key, **kwargs)
    monkeypatch.setattr(analysis, "get_path", configured_path)
    de_dir = tmp_path / "de"
    snapshots = seed_historical_results(de_dir)
    read_attempts = []
    original_reader = analysis.read_de_results
    def observed_reader(path):
        path = Path(path).resolve()
        read_attempts.append(path)
        assert path not in snapshots, "Attempted to read a historical DE result"
        return original_reader(path)
    monkeypatch.setattr(analysis, "read_de_results", observed_reader)
    with pytest.raises(error_type) as error:
        analysis.perform_unpaired_differential_expression(
            alt, ref, de_output_dir=str(de_dir), session_dir=str(tmp_path), r_timeout=timeout)
    assert not any(path in snapshots for path in read_attempts)
    if error_type is not FileNotFoundError:
        assert read_attempts == []
    else:
        assert len(read_attempts) == 1
    assert all(path.read_bytes() == contents for path, contents in snapshots.items())
    print(f"exception={type(error.value).__name__}; read_attempts={len(read_attempts)}; historical_read_attempts=0; historical_files_unchanged={len(snapshots)}")


@pytest.mark.parametrize("failure", ["none", "de", "de_timeout", "de_empty", "kegg", "kegg_empty", "panels", "real_plots"],
                         ids=["real_r", "r_failure", "r_timeout", "r_missing_outputs", "kegg_failure", "kegg_missing_outputs", "panels_failure", "real_r_plots"])
def test_workflow_routes_real_r_and_reports_analysis_status(frames, tmp_path, monkeypatch, failure):
    import omic_fetch_analysis_workflow as workflow

    ref, alt = frames
    x = pd.DataFrame(np.vstack([ref.iloc[:, 1:].to_numpy().T, alt.iloc[:, 1:].to_numpy().T]),
                     columns=ref.Name.tolist())
    labels = np.array([0] * 8 + [1] * 8)
    metadata = pd.DataFrame({"source": ["fixture"] * 16, "dataset_id": ["study"] * 16,
                             "donor_id": [f"donor{i // 2}" for i in range(16)],
                             "suspension_type": ["nucleus"] * 16})
    mapping = {"normal": 0, "Alzheimer's Disease": 1}
    def existing_data(*args, **kwargs):
        return x.copy(), labels.copy(), metadata.copy(), {}, True, "disease", "", mapping
    monkeypatch.setattr(workflow, "omic_fetch_with_new_loader", existing_data)
    def remote_enrichr(genes, sample_id, enrich_output_dir=None, databases=None, request_timeout=60):
        (Path(enrich_output_dir) / sample_id).mkdir(parents=True, exist_ok=True)
        return {library: {library: [[1, "fixture term", 0.001, 2.0, 12.0, genes[:2], 0.01, 0, 0]]}
                for library in databases}
    monkeypatch.setattr(analysis, "enrichr_analysis", remote_enrichr)
    plot_enabled = failure in ("kegg", "kegg_empty", "panels", "real_plots")
    if failure == "real_plots":
        monkeypatch.setenv("COMPOSITE", "false")
        monkeypatch.setenv("PERMUTATIONS", "19")
    de_failure = failure in ("de", "de_timeout", "de_empty")
    if de_failure:
        script = tmp_path / "failure.R"
        script.write_text({"de": 'stop("workflow DE failure")', "de_timeout": 'Sys.sleep(5)', "de_empty": 'quit(status=0)'}[failure], encoding="utf-8")
        original_get_path = analysis.get_path
        monkeypatch.setattr(analysis, "get_path", lambda key, **kw: str(script) if key == "analysis.r_script" else original_get_path(key, **kw))
        snapshots = seed_historical_results(tmp_path / "differential_expression")
        read_attempts = []
        original_reader = analysis.read_de_results
        def observed_reader(path):
            path = Path(path).resolve()
            read_attempts.append(path)
            assert path not in snapshots, "Workflow attempted to read a historical DE result"
            return original_reader(path)
        monkeypatch.setattr(analysis, "read_de_results", observed_reader)
        monkeypatch.setattr(workflow, "read_de_results", observed_reader)
    elif plot_enabled and failure != "real_plots":
        script = tmp_path / "kegg-failure.R"
        script.write_text('stop("KEGG failure")' if failure == "kegg" else 'quit(status=0)', encoding="utf-8")
        original_get_path = workflow.get_path
        monkeypatch.setattr(workflow, "get_path", lambda key, **kw: str(script) if key == "enrichment.kegg_script" else original_get_path(key, **kw))
        monkeypatch.setattr(analysis, "render_analysis_plots", lambda *a, **k: {"status": "success", "files": []})
        if failure == "panels":
            def failed_panels(*args, **kwargs):
                raise RuntimeError("panel renderer failed")
            monkeypatch.setattr(analysis, "render_analysis_plots", failed_panels)
        stale_plot = tmp_path / "plots/kegg_dotplot.png"
        stale_plot.parent.mkdir()
        stale_plot.write_bytes(b"old PNG must not imply a successful current R run")
    result = workflow.omic_fetch_analysis_workflow(
        disease="Alzheimer's Disease", session_dir=str(tmp_path), enable_plotting=plot_enabled,
        r_timeout=0.2 if failure == "de_timeout" else 120)
    assert result["retrieval_success"] is True
    assert result["analysis_success"] is (not de_failure)
    assert result["kegg_success"] is (failure == "real_plots")
    if de_failure:
        assert result["analysis_paths"] is None
        assert result["top_genes_by_fdr"] == []
        assert not any(path in snapshots for path in read_attempts)
        assert all(path.read_bytes() == contents for path, contents in snapshots.items())
        print("analysis_success=False; analysis_paths=None; top_genes_by_fdr=[]; historical_read_attempts=0; historical_files_unchanged=5")
    else:
        assert result["top_genes_by_fdr"]
        state = tmp_path / "casestudy_R/analysis_state.rds"
        probe = subprocess.run(["Rscript", "-e", 's <- readRDS(commandArgs(TRUE)[1]); stopifnot(dir.exists(s$config$out_dir), dir.exists(s$config$de_dir), file.exists(s$config$state_file), file.exists(s$config$ref_csv), file.exists(s$config$alt_csv)); cat(s$contrast)', str(state)],
                               check=True, capture_output=True, text=True, timeout=30)
        assert probe.stdout == "Alzheimer.s.Disease - normal"
        assert Path(result["analysis_paths"]["differential_expression_dir"]).is_dir()
    if plot_enabled and failure != "real_plots":
        assert stale_plot.read_bytes() == b"old PNG must not imply a successful current R run"
    if failure == "panels":
        assert result["success"] is False
        assert result["de_plots_status"] == "failed"
        assert "panel renderer failed" in result["de_plots_error"]
        assert result["plots_for_report"]["volcano_plots"] == []
    if failure == "real_plots":
        assert result["success"] is True
        assert result["plot_success"] is True
        assert result["de_plots_status"] == "success"
        assert result["enrichment_plot_status"] == "success"
        report = result["plots_for_report"]
        assert report["volcano_plots"] == []
        assert len(report["case_study_plots"]) >= 3
        assert report["enrichment_bar_plots"] == []
        assert len(report["kegg_pathway_plots"]) == 6
        assert all((tmp_path / p).is_file() for p in report["all_plots"])

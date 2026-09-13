"""Network-isolated checks for saved-results Enrichr analysis."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import requests


@pytest.fixture
def enrichment():
    path = Path(__file__).resolve().parents[1] / "enrichment.py"
    assert path.is_file(), "The saved-results Enrichr component has not been implemented"
    spec = importlib.util.spec_from_file_location("pseudobulk_enrichment_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save_results(directory, rows=None):
    directory.mkdir(parents=True, exist_ok=True)
    if rows is None:
        rows = [
            ("UP", 2.0, 0.001, True),
            ("DOWN", -2.0, 0.002, True),
            ("ZERO", 0.0, 0.003, True),
            ("BOUNDARY", 1.0, 0.05, True),
            ("UNTESTED", 1.0, 0.001, False),
            ("NAN", 1.0, np.nan, True),
            ("INF", np.inf, 0.001, True),
        ]
    frame = pd.DataFrame(rows, columns=["gene", "log2FoldChange", "padj", "tested"])
    frame["baseMean"] = 30.0
    frame["lfcSE"] = 0.5
    frame["stat"] = 2.0
    frame["pvalue"] = 0.001
    frame["filter_reason"] = ""
    frame.to_csv(directory / "results.csv", index=False)
    (directory / "status.json").write_text(json.dumps({
        "status": "success",
        "contrast": {"column": "disease", "numerator": "AD", "reference": "control"},
    }))
    return directory


def response(payload, status=200):
    result = requests.Response()
    result.status_code = status
    result._content = (payload if isinstance(payload, str) else json.dumps(payload)).encode()
    result.url = "https://maayanlab.cloud/Enrichr/enrich"
    result.headers["Content-Type"] = "application/json"
    return result


def fake_api(monkeypatch, enrichment, rows=None):
    """Replace only HTTP transport; all analysis, files, and plots remain real."""
    uploads, queries = [], []
    if rows is None:
        rows = []

    def post(url, *, files, data, timeout):
        assert url == "https://maayanlab.cloud/Enrichr/addList"
        assert set(files) == {"list"}
        assert set(data) == {"description"}
        assert timeout > 0
        uploads.append(files["list"][1].splitlines())
        return response({"userListId": 100 + len(uploads), "shortId": "abc"})

    def get(url, *, params, timeout):
        assert url == "https://maayanlab.cloud/Enrichr/enrich"
        assert set(params) == {"userListId", "backgroundType"}
        assert timeout > 0
        queries.append((params["userListId"], params["backgroundType"]))
        return response({params["backgroundType"]: rows})

    monkeypatch.setattr(enrichment.requests, "post", post)
    monkeypatch.setattr(enrichment.requests, "get", get)
    return uploads, queries


def test_filters_tested_finite_adjusted_p_values_and_preserves_direction(tmp_path, monkeypatch, enrichment):
    comparison = save_results(tmp_path / "ad_vs_control")
    uploads, queries = fake_api(monkeypatch, enrichment)
    result = enrichment.run_enrichment(comparison, retries=0)

    assert uploads == [["UP", "DOWN", "ZERO"], ["UP"], ["DOWN"]]
    assert len(queries) == 42
    assert {library for _, library in queries} == {
        "GO_Biological_Process_2021", "GO_Molecular_Function_2021",
        "GO_Cellular_Component_2021", "KEGG_2021_Human", "Reactome_2022",
        "WikiPathways_2019_Human", "MSigDB_Hallmark_2020", "DisGeNET",
        "OMIM_Disease", "OMIM_Expanded", "Human_Phenotype_Ontology",
        "Jensen_DISEASES", "GTEx_Tissue_Expression_Down", "GTEx_Tissue_Expression_Up",
    }
    assert result["status"] == "success"
    assert result["directions"]["all"]["gene_count"] == 3
    assert (comparison / "enrichment/up/genes.txt").read_text() == "UP\n"
    assert (comparison / "enrichment/down/genes.txt").read_text() == "DOWN\n"
    background = result["background"]
    assert background["mode"] == "server_default"
    assert background["custom_background_submitted"] is False
    assert background["server_universe_size"] is None
    assert background["server_universe_membership"] is None
    assert background["documented_nominal_size"] == 20000
    assert result["source"]["sha256"]
    assert result["contrast"]["numerator"] == "AD"


def test_no_gene_or_table_cap_and_preserves_api_scores(tmp_path, monkeypatch, enrichment):
    rows = [(f"G{i}", 1.0, 0.001, True) for i in range(1505)]
    comparison = save_results(tmp_path / "large", rows)
    api_rows = [[i + 1, f"Term {i}", 0.01, -1.5, 20.0, ["G0", "G1"], 0.25, 0, 0]
                for i in range(65)]
    api_rows[-1].append({"new_field": "retained"})
    uploads, _ = fake_api(monkeypatch, enrichment, api_rows)

    enrichment.run_enrichment(comparison, retries=0)

    assert len(uploads[0]) == 1505
    table = pd.read_csv(comparison / "enrichment/all/tables/KEGG_2021_Human.csv")
    assert len(table) == 65
    assert table.iloc[-1]["term"] == "Term 64"
    assert set(table["api_score_4"]) == {-1.5}
    assert not any("odds" in column.lower() for column in table.columns)
    assert json.loads(table.iloc[-1]["extra_api_fields"]) == [{"new_field": "retained"}]
    assert table["overlap_count"].tolist() == [2] * 65
    raw = json.loads((comparison / "enrichment/all/raw/KEGG_2021_Human.attempt-1.body.txt").read_text())
    assert raw["KEGG_2021_Human"] == api_rows


def test_empty_sets_do_not_upload(tmp_path, monkeypatch, enrichment):
    comparison = save_results(tmp_path / "empty", [("G", 1.0, 0.2, True)])
    uploads, queries = fake_api(monkeypatch, enrichment)

    result = enrichment.run_enrichment(comparison)

    assert result["status"] == "empty"
    assert not uploads and not queries
    for direction in ("all", "up", "down"):
        assert result["directions"][direction]["status"] == "empty"
        assert (comparison / f"enrichment/{direction}/genes.txt").read_text() == ""


def test_fold_change_minimum_applies_before_submission(tmp_path, monkeypatch, enrichment):
    comparison = save_results(tmp_path / "threshold", [
        ("POS", 1.0, 0.01, "TRUE"), ("NEG", -1.0, 0.01, "true"),
        ("SMALL", 0.5, 0.01, "True"), ("FALSE", 2.0, 0.01, "FALSE"),
    ])
    uploads, _ = fake_api(monkeypatch, enrichment)
    result = enrichment.run_enrichment(comparison, log2fc_min=1.0)
    assert uploads == [["POS", "NEG"], ["POS"], ["NEG"]]
    assert result["thresholds"]["log2fc_min"] == 1.0


def test_retries_upload_and_records_all_failed_attempts(tmp_path, monkeypatch, enrichment):
    comparison = save_results(tmp_path / "failed", [("G", 1.0, 0.01, True)])
    attempts = []

    def unavailable(*args, **kwargs):
        attempts.append(1)
        return response("server unavailable", status=503)

    monkeypatch.setattr(enrichment.requests, "post", unavailable)
    monkeypatch.setattr(enrichment.time, "sleep", lambda _: None)
    result = enrichment.run_enrichment(comparison, retries=1)
    assert len(attempts) == 4  # all and up each receive initial call + one retry
    assert result["status"] == "error"
    assert result["directions"]["all"]["status"] == "error"
    assert result["directions"]["down"]["status"] == "empty"
    raw = comparison / "enrichment/all/raw"
    assert (raw / "addList.attempt-1.body.txt").read_text() == "server unavailable"
    assert (raw / "addList.attempt-2.body.txt").read_text() == "server unavailable"
    assert result["directions"]["all"]["upload"]["attempts"] == 2
    assert (comparison / "enrichment/all/genes.txt").read_text() == "G\n"


def test_library_failure_preserves_other_tables_and_reports_partial(tmp_path, monkeypatch, enrichment):
    comparison = save_results(tmp_path / "partial", [("G", 1.0, 0.01, True)])
    fake_api(monkeypatch, enrichment)

    def get(url, *, params, timeout):
        library = params["backgroundType"]
        if library == "KEGG_2021_Human":
            return response({"wrong_library": []})
        return response({library: [[1, "Term", 0.01, 2.0, 4.0, ["G"], 0.25, 0, 0]]})

    monkeypatch.setattr(enrichment.requests, "get", get)
    result = enrichment.run_enrichment(comparison, retries=0)
    assert result["status"] == "partial"
    assert result["directions"]["all"]["libraries"]["KEGG_2021_Human"]["status"] == "error"
    assert result["directions"]["all"]["libraries"]["Reactome_2022"]["row_count"] == 1
    assert not (comparison / "enrichment/all/tables/KEGG_2021_Human.csv").exists()
    assert (comparison / "enrichment/all/tables/Reactome_2022.csv").is_file()


def test_invalid_results_or_existing_output_fail_before_any_upload(tmp_path, monkeypatch, enrichment):
    comparison = save_results(tmp_path / "invalid")
    frame = pd.read_csv(comparison / "results.csv")
    frame.loc[0, "gene"] = "bad\ngene"
    frame.to_csv(comparison / "results.csv", index=False)
    uploads, _ = fake_api(monkeypatch, enrichment)
    with pytest.raises(ValueError, match="gene"):
        enrichment.run_enrichment(comparison)
    assert not uploads

    comparison = save_results(tmp_path / "existing")
    (comparison / "enrichment").mkdir()
    with pytest.raises(FileExistsError):
        enrichment.run_enrichment(comparison)
    assert not uploads


def test_plot_regeneration_reads_saved_tables_and_uses_significant_terms_only(tmp_path, monkeypatch, enrichment):
    comparison = save_results(tmp_path / "plots")
    api_rows = [
        [1, "Significant", 0.001, 2, 20, ["UP"], 0.01, 0, 0],
        [2, "Raw p only", 0.001, 3, 30, ["UP"], 0.2, 0, 0],
        [3, "Boundary", 0.001, 4, 40, ["UP"], 0.05, 0, 0],
        [4, "Underflow", 0.0, 4, 40, ["UP"], 0.0, 0, 0],
    ]
    fake_api(monkeypatch, enrichment, api_rows)
    enrichment.run_enrichment(comparison, retries=0)
    (comparison / "results.csv").unlink()

    def network_forbidden(*args, **kwargs):
        raise AssertionError("Plot regeneration must not access the network")

    monkeypatch.setattr(enrichment.requests, "post", network_forbidden)
    monkeypatch.setattr(enrichment.requests, "get", network_forbidden)
    result = enrichment.regenerate_enrichment_plots(comparison)

    assert result["status"] == "success"
    for direction in ("all", "up", "down"):
        plot_dir = comparison / "enrichment/plots" / direction
        table = pd.read_csv(plot_dir / "kegg_dotplot_data.csv")
        assert set(table["term"]) == {"Significant", "Underflow"}
        assert set(table["direction"]) == {direction}
        assert np.isfinite(table["minus_log10_adjusted_p_value"]).all()
        assert (plot_dir / "kegg_dotplot.png").stat().st_size > 1000
        assert (plot_dir / "pathway_combined_plot.png").stat().st_size > 1000
        combined = pd.read_csv(plot_dir / "pathway_combined_plot_data.csv")
        assert set(combined["category"]) == {"GO BP", "GO CC", "GO MF", "KEGG", "DisGeNET"}
        assert set(combined["term"]) == {"Significant", "Underflow"}


@pytest.mark.parametrize("list_id", [None, True, -1, "invalid"])
def test_invalid_upload_id_saves_failed_direction_without_querying_libraries(
    tmp_path, monkeypatch, enrichment, list_id,
):
    comparison = save_results(tmp_path / "invalid_upload", [("G", 1.0, 0.01, True)])
    _, queries = fake_api(monkeypatch, enrichment)
    monkeypatch.setattr(
        enrichment.requests, "post", lambda *args, **kwargs: response({"userListId": list_id}),
    )

    result = enrichment.run_enrichment(comparison, retries=0)

    assert result["status"] == "error"
    assert not queries
    assert result["directions"]["down"]["status"] == "empty"
    for direction in ("all", "up"):
        saved = json.loads((comparison / f"enrichment/{direction}/status.json").read_text())
        assert saved == result["directions"][direction]
        assert saved["status"] == saved["upload"]["status"] == "error"
        assert saved["libraries"] == {}
        assert "valid userListId" in saved["reason"]


def test_stricter_saved_table_regeneration_removes_stale_figures(tmp_path, monkeypatch, enrichment):
    comparison = tmp_path / "saved_only"
    enrichment_dir = comparison / "enrichment"
    tables = enrichment_dir / "all/tables"
    tables.mkdir(parents=True)
    (enrichment_dir / "metadata.json").write_text(json.dumps({"contrast": {}}))
    pd.DataFrame({
        "term": ["Previously significant"], "adjusted_p_value": [0.01], "overlap_count": [3],
    }).to_csv(tables / "KEGG_2021_Human.csv", index=False)
    (tables / "GO_Biological_Process_2021.csv").write_text("wrong_column\nvalue\n")
    plot_dir = enrichment_dir / "plots/all"
    plot_dir.mkdir(parents=True)
    old_figures = [plot_dir / f"{name}{suffix}"
                   for name in ("kegg_dotplot", "pathway_combined_plot")
                   for suffix in (".png", ".pdf")]
    for path in old_figures:
        path.write_bytes(b"previous figure")

    def network_forbidden(*args, **kwargs):
        raise AssertionError("Saved-table regeneration must not access the network")

    monkeypatch.setattr(enrichment.requests, "get", network_forbidden)
    monkeypatch.setattr(enrichment.requests, "post", network_forbidden)
    result = enrichment.regenerate_enrichment_plots(comparison, alpha=0.001)

    assert result["status"] == "empty"
    assert all(not path.exists() for path in old_figures)
    all_direction = result["directions"]["all"]
    assert "Saved enrichment table lacks" in all_direction["unavailable_libraries"]["GO_Biological_Process_2021"]
    assert all_direction["unavailable_libraries"]["DisGeNET"] == "No successful saved table"
    for item in all_direction["plots"].values():
        assert item["status"] == "empty"
        assert item["files"] == []
        assert pd.read_csv(item["data"]).empty

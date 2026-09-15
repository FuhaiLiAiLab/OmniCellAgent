"""HTTP boundary tests; live service evidence is collected separately."""
import json
from pathlib import Path
import sys

import pandas as pd
import pytest
import requests

OMIC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OMIC))
import omic_analysis_components as analysis
from test_python_r_de import frames

LIBRARY = "KEGG_2021_Human"
ROW = [1, "Example pathway", 0.001, 2.5, 17.2, ["APOE", "APP"], 0.02, 0, 0]
DTYPES = {"Rank": "int64", "Term": "object", "P-value": "float64", "Odds Ratio": "float64",
          "Combined Score": "float64", "Genes": "object", "Adjusted P-value": "float64",
          "Old P-value": "float64", "Old Adjusted P-value": "float64"}


def response(payload, code=200):
    value = requests.Response()
    value.status_code = code
    value._content = json.dumps(payload).encode("utf-8")
    value.headers["Content-Type"] = "application/json"
    value.url = "https://maayanlab.cloud/Enrichr/enrich"
    return value


def test_empty_and_nonempty_enrichr_tables_have_same_dtypes(tmp_path, monkeypatch):
    from enrichr_client import read_enrichment_results
    monkeypatch.setattr(requests, "post", lambda *a, **k: response({"userListId": 123}))
    for rows, state in [([ROW], "success"), ([], "empty")]:
        monkeypatch.setattr(requests, "get", lambda *a, **k: response({LIBRARY: rows}))
        raw = analysis.enrichr_analysis(["APOE", "APP"], "case_all_regulated", str(tmp_path), [LIBRARY])
        assert raw[LIBRARY][LIBRARY] == rows
        folder = tmp_path / "case_all_regulated"
        table = read_enrichment_results(folder / f"{LIBRARY}_results.csv")
        assert table.dtypes.astype(str).to_dict() == DTYPES
        assert len(table) == len(rows)
        status = json.loads((folder / "enrichment_status.json").read_text())
        assert status["status"] == state
        assert status["libraries"][LIBRARY]["returned_rows"] == len(rows)
        if not rows:
            assert list(folder.glob("history/*/KEGG_2021_Human_results.csv"))
            assert "No returned terms" in (folder / "summary.txt").read_text()
            previous = next(folder.glob("history/*/enrichment_status.json"))
            old_status = json.loads(previous.read_text())
            raw_path = previous.parent / old_status["requests"][-1]["raw_body"]
            assert json.loads(raw_path.read_text())[LIBRARY] == [ROW]


@pytest.mark.parametrize("phase,failure", [("post", "timeout"), ("get", "timeout"),
                                         ("post", "connection"), ("get", "http")])
def test_request_failure_is_explicit_and_cannot_read_historical_results(tmp_path, monkeypatch, phase, failure):
    from enrichr_client import EnrichrError, read_enrichment_results
    folder = tmp_path / "case_all_regulated"
    folder.mkdir()
    old = folder / f"{LIBRARY}_results.csv"
    old.write_text("old historical table", encoding="utf-8")
    monkeypatch.setattr(requests, "post", lambda *a, **k: response({"userListId": 123}))
    monkeypatch.setattr(requests, "get", lambda *a, **k: response({LIBRARY: [ROW]}))
    def fail(*args, **kwargs):
        assert kwargs["timeout"] == 0.2
        if failure == "timeout":
            raise requests.Timeout("deliberate timeout")
        if failure == "connection":
            raise requests.ConnectionError("deliberate connection failure")
        return response({"error": "unavailable"}, 503)
    monkeypatch.setattr(requests, phase, fail)
    with pytest.raises(EnrichrError):
        analysis.enrichr_analysis(["APOE"], "case_all_regulated", str(tmp_path), [LIBRARY], request_timeout=0.2)
    status = json.loads((folder / "enrichment_status.json").read_text())
    assert status["status"] == "failed"
    assert status["error"]
    expected_error = {"timeout": "Timeout", "connection": "ConnectionError", "http": "HTTPError"}[failure]
    assert expected_error in status["error"]
    assert not old.exists()
    archived = list(folder.glob("history/*/KEGG_2021_Human_results.csv"))
    assert len(archived) == 1 and archived[0].read_text() == "old historical table"
    with pytest.raises(EnrichrError):
        read_enrichment_results(old)
    print(f"phase={phase}; failure={failure}; status={status['status']}; old_current_file_exists={old.exists()}")


@pytest.mark.parametrize("payload", [{}, {LIBRARY: None}, {LIBRARY: [[1, "broken"]]}])
def test_malformed_http_200_is_failure_not_empty(tmp_path, monkeypatch, payload):
    from enrichr_client import EnrichrError
    monkeypatch.setattr(requests, "post", lambda *a, **k: response({"userListId": 123}))
    monkeypatch.setattr(requests, "get", lambda *a, **k: response(payload))
    with pytest.raises(EnrichrError):
        analysis.enrichr_analysis(["APOE"], "case_all_regulated", str(tmp_path), [LIBRARY])
    status = json.loads((tmp_path / "case_all_regulated/enrichment_status.json").read_text())
    assert status["status"] == "failed"


def test_http_empty_result_and_no_significant_terms_are_distinct(tmp_path, monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *a, **k: response({"userListId": 123}))
    row = list(ROW)
    row[6] = 0.9
    monkeypatch.setattr(requests, "get", lambda *a, **k: response({LIBRARY: [row]}))
    analysis.enrichr_analysis(["APOE"], "case_all_regulated", str(tmp_path), [LIBRARY])
    status = json.loads((tmp_path / "case_all_regulated/enrichment_status.json").read_text())
    assert status["status"] == "success"
    assert status["libraries"][LIBRARY]["returned_rows"] == 1
    assert status["libraries"][LIBRARY]["significant_rows_fdr_0_05"] == 0


@pytest.mark.parametrize("failure", ["timeout", "connection", "http", "summary", "invalid_timeout"])
def test_workflow_reports_enrichment_failure_after_successful_de(frames, tmp_path, monkeypatch, failure):
    import numpy as np
    import omic_fetch_analysis_workflow as workflow
    from enrichr_client import EnrichrError, read_enrichment_results
    ref, alt = frames
    x = pd.DataFrame(np.vstack([ref.iloc[:, 1:].to_numpy().T, alt.iloc[:, 1:].to_numpy().T]), columns=ref.Name)
    labels = np.array([0] * 8 + [1] * 8)
    metadata = pd.DataFrame({"source": ["fixture"] * 16, "dataset_id": ["study"] * 16,
                             "donor_id": [f"donor{i}" for i in range(16)]})
    monkeypatch.setattr(workflow, "omic_fetch_with_new_loader", lambda *a, **k:
                        (x.copy(), labels.copy(), metadata.copy(), {}, True, "disease", "", {"normal": 0, "AD": 1}))
    monkeypatch.setattr(requests, "post", lambda *a, **k: response({"userListId": 123}))
    def fail(*args, **kwargs):
        if failure == "timeout":
            raise requests.Timeout("workflow network timeout")
        if failure == "connection":
            raise requests.ConnectionError("workflow connection failure")
        return response({"error": "unavailable"}, 503) if failure == "http" else response({LIBRARY: [ROW]})
    monkeypatch.setattr(requests, "get", fail)
    if failure == "summary":
        def broken_summary(*args, **kwargs):
            raise OSError("deliberate summary write failure")
        monkeypatch.setattr(analysis, "create_enrichment_summary", broken_summary)
    historical = tmp_path / "enrichment_results/AD_down_regulated" / f"{LIBRARY}_results.csv"
    historical.parent.mkdir(parents=True)
    historical.write_text("historical down results must not be used", encoding="utf-8")
    result = workflow.omic_fetch_analysis_workflow(disease="AD", session_dir=str(tmp_path),
                    enable_plotting=False, enrichment_timeout=0 if failure == "invalid_timeout" else 0.2,
                    enrichment_databases=[LIBRARY])
    assert result["success"] is False
    assert result["de_success"] is True
    assert result["analysis_success"] is False
    assert result["enrichment_success"] is False
    assert result["enrichment_status"] == "failed"
    assert result["enrichment_error"]
    assert result["kegg_success"] is False
    with pytest.raises(EnrichrError):
        read_enrichment_results(historical)
    assert historical.read_text() == "historical down results must not be used"
    print(f"{failure}: success={result['success']}, de_success={result['de_success']}, enrichment_status={result['enrichment_status']}")


def test_empty_current_enrichment_does_not_return_historical_plots(frames, tmp_path, monkeypatch):
    import numpy as np
    import omic_fetch_analysis_workflow as workflow
    ref, alt = frames
    x = pd.DataFrame(np.vstack([ref.iloc[:, 1:].to_numpy().T, alt.iloc[:, 1:].to_numpy().T]), columns=ref.Name)
    y = np.array([0] * 8 + [1] * 8)
    meta = pd.DataFrame({"donor_id": [str(i) for i in range(16)]})
    monkeypatch.setattr(workflow, "omic_fetch_with_new_loader", lambda *a, **k:
                        (x.copy(), y.copy(), meta.copy(), {}, True, "disease", "", {"normal": 0, "AD": 1}))
    monkeypatch.setattr(requests, "post", lambda *a, **k: response({"userListId": 123}))
    monkeypatch.setattr(requests, "get", lambda *a, **k: response({LIBRARY: []}))
    monkeypatch.setattr(analysis, "create_volcano_plot", lambda *a, **k: None)
    monkeypatch.setattr(workflow, "run_r_script", lambda *a, **k: "")
    old = tmp_path / "enrichment_results/enrichment_plots/KEGG_2021_Human_all_regulated.png"
    old.parent.mkdir(parents=True)
    old.write_bytes(b"historical plot")
    result = workflow.omic_fetch_analysis_workflow(disease="AD", session_dir=str(tmp_path),
                         enable_plotting=True, enrichment_databases=[LIBRARY])
    assert result["enrichment_status"] == "empty"
    assert result["plots_for_report"]["enrichment_bar_plots"] == []
    assert not old.exists()
    assert next(old.parent.glob("history/*/*.png")).read_bytes() == b"historical plot"

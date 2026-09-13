"""Regression tests for the direct-parameter-only omic workflow interface."""

import os
from pathlib import Path
import subprocess
import sys
import inspect

import pytest


WORKFLOW_SCRIPT = Path(__file__).resolve().parents[1] / "omic_fetch_analysis_workflow.py"
sys.path.insert(0, str(WORKFLOW_SCRIPT.parent))


def test_cli_help_supports_only_direct_query_parameters(tmp_path):
    """The CLI must start without the retired NER module or text interface."""
    env = os.environ.copy()
    env["NUMBA_CACHE_DIR"] = str(tmp_path / "numba")
    env["MPLCONFIGDIR"] = str(tmp_path / "matplotlib")

    completed = subprocess.run(
        [sys.executable, str(WORKFLOW_SCRIPT), "--help"],
        cwd=WORKFLOW_SCRIPT.parents[2],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--disease" in completed.stdout
    assert "--cell-type" in completed.stdout
    assert "--organ" in completed.stdout
    assert "--tissue" in completed.stdout
    assert "--sample-size" in completed.stdout
    assert "--text" not in completed.stdout
    assert "NER" not in completed.stdout


def test_workflow_api_has_only_direct_query_parameters():
    """The public workflow must not retain the retired free-text entry point."""
    from omic_fetch_analysis_workflow import omic_fetch_analysis_workflow

    parameters = inspect.signature(omic_fetch_analysis_workflow).parameters

    assert "text" not in parameters
    assert {"disease", "cell_type", "organ", "tissue", "gender"} <= set(parameters)


def test_workflow_rejects_legacy_positional_arguments():
    """Old free-text calls must fail instead of becoming silent disease queries."""
    from omic_fetch_analysis_workflow import omic_fetch_analysis_workflow

    with pytest.raises(TypeError, match="positional"):
        omic_fetch_analysis_workflow("analyze Alzheimer's disease in brain")


def test_failed_direct_query_reports_input_timing(monkeypatch, tmp_path):
    """Timing must name the direct-input stage even when retrieval finds nothing."""
    import omic_fetch_analysis_workflow as workflow

    def no_matching_data(fetch_dict, output_dir, label="disease", suspension_type=None,
                         sample_size=1000):
        return None, None, None, {}, False, label, None, None

    monkeypatch.setattr(workflow, "omic_fetch_with_new_loader", no_matching_data)

    result = workflow.omic_fetch_analysis_workflow(
        gender="female",
        session_dir=str(tmp_path),
        enable_differential_expression=False,
        enable_plotting=False,
    )

    assert result["retrieval_success"] is False
    assert set(result["timing"]) == {"input", "fetch", "total"}

"""Real R subprocess success and failure boundaries; run inside Slurm."""
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from subprocess_r import run_r_script


def test_r_receives_arguments_without_shell_expansion(tmp_path):
    script = tmp_path / "donor's script.R"
    script.write_text('cat(commandArgs(trailingOnly=TRUE), sep="\\n")', encoding="utf-8")
    argument = "donor's cohort; $(false)"
    assert run_r_script(str(script), [argument], timeout=30) == argument + "\n"


def test_r_nonzero_exit_propagates_stderr(tmp_path, capsys):
    script = tmp_path / "failed.R"
    script.write_text('stop("DE deliberately failed")', encoding="utf-8")
    with pytest.raises(subprocess.CalledProcessError) as error:
        run_r_script(str(script), timeout=30)
    assert error.value.returncode != 0
    assert "DE deliberately failed" in error.value.stderr
    assert "DE deliberately failed" in capsys.readouterr().out


def test_r_timeout_is_not_reported_as_success(tmp_path):
    script = tmp_path / "slow.R"
    script.write_text('Sys.sleep(5)', encoding="utf-8")
    with pytest.raises(subprocess.TimeoutExpired):
        run_r_script(str(script), timeout=0.2)


def test_missing_rscript_is_not_reported_as_success(tmp_path, monkeypatch):
    script = tmp_path / "ok.R"
    script.write_text('cat("ok")', encoding="utf-8")
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(FileNotFoundError):
        run_r_script(str(script))


def test_missing_script_is_not_reported_as_success(tmp_path):
    with pytest.raises(FileNotFoundError):
        run_r_script(str(tmp_path / "missing.R"))

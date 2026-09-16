"""Standalone plot stage integration; run inside a Slurm allocation."""
import json
import os
from pathlib import Path
import re
import subprocess

import pytest
from test_r_pipeline_contract import cohort, RUNNER


def invoke(root, stage, *options):
    args = ["Rscript", str(RUNNER), str(root), str(root / "result"),
            "--ref-csv", "reference.csv", "--alt-csv", "alternate.csv",
            "--ref-label", "Control group", "--alt-label", "Patient group"]
    if stage is not None:
        args += ["--stage", stage]
    return subprocess.run(args + list(options), capture_output=True, text=True, cwd=root,
                          timeout=600, env={**os.environ, "PERMUTATIONS": "19"})


def test_plots_reuses_state_and_publishes_complete_manifest(cohort):
    result = invoke(cohort, "de")
    assert result.returncode == 0, result.stderr
    protected = list(cohort.rglob("*.csv")) + [cohort / "result/analysis_state.rds"]
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in protected}
    assert not list(cohort.rglob("*.png"))
    result = invoke(cohort, "plots")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "DE input:" not in result.stderr and "DE complete:" not in result.stderr
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in protected}
    assert not (cohort / "Rplots.pdf").exists()
    manifest = json.loads((cohort / "result/plot_manifest.json").read_text())
    assert manifest["status"] == "success"
    files = [Path(p) for p in manifest["files"]]
    assert all(p.is_absolute() and p.is_file() and p.stat().st_size for p in files)
    expected = {f"DE_results_{panel}.{ext}" for panel in
                ["volcano", "corrgram_genes", "PCA_panel", "figure_ABCD"] for ext in ["png", "pdf"]}
    expected |= {"DE_results_violin_PC1_2groups.pdf", "DE_results_panel_A_labeled_genes.csv",
                 "DE_results_panel_B_corrgram_genes.csv"}
    assert not any(p.name.startswith("volcano_plot") for p in files)
    assert expected <= {p.name for p in files}
    for html in [p for p in files if p.suffix == ".html"]:
        text = html.read_text()
        assert "Control group" in text and "Patient group" in text
        assets = re.findall(r'(?:src|href)="([^"]+)"', text)
        local = [a for a in assets if not a.startswith(("http", "data:", "#"))]
        assert local
        for asset in local:
            assert not Path(asset).is_absolute()
            assert (html.parent / asset).resolve() in files


@pytest.mark.parametrize("mismatch", ["source", "label", "scale"])
def test_plot_state_mismatch_invalidates_manifest(cohort, mismatch):
    result = invoke(cohort, "de")
    assert result.returncode == 0, result.stderr
    manifest = cohort / "result/plot_manifest.json"
    manifest.write_text('{"status":"success","files":[]}')
    options = []
    if mismatch == "source":
        with (cohort / "reference.csv").open("a") as handle:
            handle.write("\n")
    elif mismatch == "label":
        # Change saved state, keeping CLI options unique.
        changed = subprocess.run(["Rscript", "-e", "p<-commandArgs(TRUE)[1]; s<-readRDS(p); s$config$ref_label<-'Other'; saveRDS(s,p)", str(cohort / "result/analysis_state.rds")], capture_output=True, text=True)
        assert changed.returncode == 0, changed.stderr
    else:
        options = ["--input-scale", "log2_already"]
    result = invoke(cohort, "plots", *options)
    assert result.returncode != 0
    assert "mismatch" in result.stderr.lower()
    assert not manifest.exists()


def test_default_runs_all_without_rmarkdown(cohort, monkeypatch):
    monkeypatch.setenv("COMPOSITE", "false")
    result = invoke(cohort, None)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (cohort / "result/analysis_state.rds").is_file()
    assert (cohort / "result/plot_manifest.json").is_file()
    assert not list(cohort.rglob("*.md"))


def test_redraw_rejects_pseudobulk_and_keeps_state(cohort, monkeypatch):
    result = invoke(cohort, "de")
    assert result.returncode == 0, result.stderr
    state = cohort / "result/analysis_state.rds"
    before = state.read_bytes()
    monkeypatch.setenv("PSEUDOBULK", "true")
    result = invoke(cohort, "plots")
    assert result.returncode != 0
    assert "PSEUDOBULK" in result.stderr
    assert state.read_bytes() == before
    assert not (cohort / "result/plot_manifest.json").exists()


def test_optional_composite_and_pc2_manifest(cohort, monkeypatch):
    monkeypatch.setenv("COMPOSITE", "false")
    monkeypatch.setenv("WHICH_PC", "PC2")
    stale = cohort / "volcano_plots/volcano_plot_files/stale.js"
    stale.parent.mkdir(parents=True)
    stale.write_text("previous dependency")
    result = invoke(cohort, None)
    assert result.returncode == 0, result.stdout + result.stderr
    manifest = json.loads((cohort / "result/plot_manifest.json").read_text())
    names = {Path(p).name for p in manifest["files"]}
    assert "DE_results_violin_PC2_2groups.pdf" in names
    assert not any("figure_ABCD" in name or "heatmap" in name for name in names)
    assert "stale.js" not in names


def test_all_preserves_explicit_optional_pseudobulk(cohort, monkeypatch):
    import pandas as pd
    pd.DataFrame({"disease": ["disease"] * 8 + ["healthy"] * 8,
                  "donor_id": [f"donor{i // 2}" for i in range(16)]}).to_csv(
        cohort / "labels.csv", index=False)
    monkeypatch.setenv("PSEUDOBULK", "true")
    monkeypatch.setenv("COMPOSITE", "false")
    result = invoke(cohort, "all")
    assert result.returncode == 0, result.stdout + result.stderr
    assert (cohort / "result/DE_results_table_pseudobulk_donor.csv").is_file()

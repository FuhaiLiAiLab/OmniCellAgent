"""Opt-in plotting checks on saved AD results; never fetch data or fit DE.

Set OMIC_EXISTING_PLOT_SESSION to an existing session with analysis_state.rds.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

MODULE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE))
from r_plotting import collect_plot_outputs, read_plot_manifest


@pytest.fixture
def existing_session():
    value = os.environ.get("OMIC_EXISTING_PLOT_SESSION")
    if not value:
        pytest.skip("Set OMIC_EXISTING_PLOT_SESSION; this test never creates DE results")
    session = Path(value).resolve()
    assert (session / "casestudy_R/analysis_state.rds").is_file()
    return session


def test_saved_ad_results_render_current_panels_without_legacy_volcano(existing_session, tmp_path):
    protected = list(existing_session.rglob("*.csv")) + [existing_session / "casestudy_R/analysis_state.rds"]
    def snapshot():
        return {str(p): [hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns]
                for p in protected}
    before = snapshot()
    out = tmp_path / "casestudy_R"
    result = subprocess.run([
        "Rscript", str(MODULE / "tests/fixtures/verify_panel_sources.R"),
        str(MODULE / "run_casestudy.R"), "--stage", "plots",
        str(existing_session), str(out), "--state-file",
        str(existing_session / "casestudy_R/analysis_state.rds"),
        "--ref-label", "normal", "--alt-label", "Alzheimer's Disease",
    ], capture_output=True, text=True, timeout=1200,
        env={**os.environ, "COMPOSITE": "true", "PSEUDOBULK": "false"})
    (tmp_path / "render.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    evidence = json.loads((out / "panel_source_verification.json").read_text())
    assert evidence["panel_A"]["renderer_labels_match"] is True
    assert evidence["panel_B"]["renderer_order_matches"] is True
    assert "DE complete:" not in result.stderr
    assert snapshot() == before
    manifest = read_plot_manifest(out / "plot_manifest.json", [out])
    names = {Path(p).name for p in manifest["files"]}
    assert not any(n.startswith("volcano_plot") for n in names)
    assert {"DE_results_volcano.png", "DE_results_corrgram_genes.png",
            "DE_results_PCA_panel.png", "DE_results_figure_ABCD.png",
            "DE_results_violin_PC1_2groups.pdf"} <= names
    html, report = collect_plot_outputs(tmp_path, manifest["files"], [])
    assert html == []
    assert report["volcano_plots"] == []
    assert len(report["case_study_plots"]) == 4
    (tmp_path / "verification.json").write_text(json.dumps({
        "protected_files": len(protected), "unchanged": True, "report": report}, indent=2))


def test_saved_expression_plot_letters_support_hyphenated_labels(existing_session, tmp_path):
    # Evaluate only the plotting helper and compute PCA from saved expression.
    code = r'''
args <- commandArgs(TRUE)
exprs <- parse(args[1])
run <- Filter(function(x) is.call(x) && identical(x[[1]], as.name("<-")) &&
              identical(x[[2]], as.name("run_plots")), as.list(exprs))[[1]]
eval(run)
helper <- Filter(function(x) is.call(x) && identical(x[[1]], as.name("<-")) &&
                 identical(x[[2]], as.name("cld_table")), as.list(body(run_plots)))[[1]]
library(multcompView)
s <- readRDS(args[2])
genes <- head(s$native$Gene[order(s$native$adj.P.Val)], 100)
pca <- prcomp(t(s$inputs$expr[genes,,drop=FALSE]), scale.=TRUE)
pcoadata <- data.frame(PC1=pca$x[,1], PC2=pca$x[,2], Subtype=s$inputs$group)
eval(helper)
original <- cld_table("PC1")
old <- levels(factor(pcoadata$Subtype))
new <- c("disease-with-hyphens", "normal-control")
pcoadata$Subtype <- factor(new[match(as.character(pcoadata$Subtype),old)],levels=new)
renamed <- cld_table("PC1")
renamed <- renamed[match(new[match(as.character(original$Subtype),old)],
                         as.character(renamed$Subtype)),,drop=FALSE]
stopifnot(identical(as.character(original$Letters), as.character(renamed$Letters)),
          isTRUE(all.equal(original$value_max,renamed$value_max)),
          setequal(renamed$Subtype,new))
print(renamed)
'''
    result = subprocess.run(["Rscript", "-e", code, str(MODULE / "run_casestudy.R"),
        str(existing_session / "casestudy_R/analysis_state.rds")],
        capture_output=True, text=True, timeout=120)
    (tmp_path / "labels.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr

"""R artifact publication and stale-manifest handling, without Python drawing."""
import json
from pathlib import Path
import sys
import os
import subprocess

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_plot_bridge_forwards_existing_input_paths_and_scale(tmp_path, monkeypatch):
    import r_plotting
    calls = []
    def renderer(script, args, **kwargs):
        calls.append(args)
        out = tmp_path / "casestudy_R"
        out.mkdir()
        artifact = out / "plot.png"
        artifact.write_bytes(b"current plot")
        (out / "plot_manifest.json").write_text(json.dumps(
            {"status": "success", "files": [str(artifact)]}))
    monkeypatch.setattr(r_plotting, "run_r_script", renderer)
    ref, alt = tmp_path / "chosen_ref.csv", tmp_path / "chosen_alt.csv"
    result = r_plotting.render_analysis_plots("run.R", tmp_path,
        ref_label="normal", alt_label="AD", ref_csv=ref, alt_csv=alt,
        input_scale="log2_already")
    args = calls[0]
    assert args[args.index("--stage") + 1] == "plots"
    assert args[args.index("--ref-csv") + 1] == str(ref)
    assert args[args.index("--alt-csv") + 1] == str(alt)
    assert args[args.index("--input-scale") + 1] == "log2_already"
    assert result["status"] == "success"


def test_silent_r_run_cannot_reuse_old_plot_manifest(tmp_path, monkeypatch):
    import r_plotting
    out = tmp_path / "casestudy_R"
    out.mkdir()
    old = out / "old.png"
    old.write_bytes(b"historical image")
    manifest = out / "plot_manifest.json"
    manifest.write_text(json.dumps({"status": "success", "files": [str(old)]}))
    monkeypatch.setattr(r_plotting, "run_r_script", lambda *a, **k: "")
    with pytest.raises(FileNotFoundError):
        r_plotting.render_analysis_plots("run.R", tmp_path, ref_label="normal", alt_label="AD")
    assert not manifest.exists()
    assert old.read_bytes() == b"historical image"


def test_manifest_rejects_artifact_outside_output_directories(tmp_path):
    from r_plotting import read_plot_manifest
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"image")
    manifest = allowed / "plot_manifest.json"
    manifest.write_text(json.dumps({"status": "success", "files": [str(outside)]}))
    with pytest.raises(ValueError):
        read_plot_manifest(manifest, [allowed])


def test_publish_keeps_html_dependencies_and_bars_in_their_own_directories(tmp_path):
    from r_plotting import publish_plot_files, read_plot_manifest
    plots = tmp_path / "staging/plots"
    bars = tmp_path / "staging/bars"
    (plots / "html_dependencies").mkdir(parents=True)
    bars.mkdir()
    html = plots / "kegg_dotplot.html"
    html.write_text('<script src="html_dependencies/widget.js"></script>')
    js = plots / "html_dependencies/widget.js"
    js.write_text("const widget = true;")
    png = bars / "KEGG_2021_Human_all_regulated.png"
    png.write_bytes(b"new image")
    manifest = plots / "plot_manifest.json"
    manifest.write_text(json.dumps({"status": "success", "files": [str(html), str(js), str(png)]}))
    parsed = read_plot_manifest(manifest, [plots, bars])
    files = publish_plot_files(parsed["files"], {plots: tmp_path / "final/plots", bars: tmp_path / "final/bars"})
    assert set(files) == {str(tmp_path / "final/plots/kegg_dotplot.html"),
                          str(tmp_path / "final/plots/html_dependencies/widget.js"),
                          str(tmp_path / "final/bars/KEGG_2021_Human_all_regulated.png")}
    assert (tmp_path / "final/plots/html_dependencies/widget.js").read_text() == "const widget = true;"
    assert (tmp_path / "final/plots/kegg_dotplot.html").read_text() == html.read_text()


def test_empty_manifest_is_explicit_not_a_missing_output_success(tmp_path):
    from r_plotting import read_plot_manifest
    manifest = tmp_path / "plot_manifest.json"
    manifest.write_text(json.dumps({"status": "empty", "files": []}))
    with pytest.raises(ValueError):
        read_plot_manifest(manifest, [tmp_path])
    assert read_plot_manifest(manifest, [tmp_path], allow_empty=True)["files"] == []


def test_report_uses_manifest_figures_not_historical_images_or_widget_assets(tmp_path):
    from r_plotting import collect_plot_outputs
    old = tmp_path / "volcano_plots/old.png"
    old.parent.mkdir()
    old.write_bytes(b"old")
    current = tmp_path / "casestudy_R/DE_results_PCA_panel.png"
    current.parent.mkdir()
    current.write_bytes(b"current")
    asset = tmp_path / "plots/html_dependencies/icon.png"
    asset.parent.mkdir(parents=True)
    asset.write_bytes(b"not a scientific figure")
    html_paths, report = collect_plot_outputs(tmp_path, [str(current)], [str(asset)])
    assert html_paths == []
    assert report["volcano_plots"] == []
    assert report["kegg_pathway_plots"] == []
    assert report["all_plots"] == ["casestudy_R/DE_results_PCA_panel.png"]
    assert report["case_study_plots"][0]["type"] == "case_study"


@pytest.mark.parametrize("failure", ["nonzero", "timeout", "missing", "failed_manifest"])
def test_group_failure_does_not_publish_partial_or_historical_plots(tmp_path, failure):
    from r_plotting import render_enrichment_plots
    inputs = tmp_path / "enrichment_results"
    for group in ("all", "up", "down"):
        (inputs / f"Case_{group}_regulated").mkdir(parents=True)
    output = tmp_path / "plots"
    output.mkdir()
    historical = output / "kegg_dotplot.png"
    historical.write_bytes(b"historical")
    (output / "plot_manifest.json").write_text(json.dumps({"status": "success", "files": [str(historical)]}))
    calls = []
    def runner(script, args, **kwargs):
        source, destination = map(Path, args)
        calls.append(source.name)
        destination.mkdir(parents=True, exist_ok=True)
        if source.name.endswith("up_regulated"):
            if failure == "nonzero":
                raise subprocess.CalledProcessError(1, [script])
            if failure == "timeout":
                raise subprocess.TimeoutExpired([script], 1)
            if failure == "failed_manifest":
                (destination / "plot_manifest.json").write_text('{"status":"failed","files":[]}')
            return
        png = destination / "kegg_dotplot.png"
        png.write_bytes(b"new")
        (destination / "plot_manifest.json").write_text(json.dumps({"status": "success", "files": [str(png)]}))
    with pytest.raises((subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError, ValueError)):
        render_enrichment_plots("original.R", tmp_path, inputs, "Case", runner=runner)
    assert calls == ["Case_all_regulated", "Case_up_regulated"]
    assert historical.read_bytes() == b"historical"
    assert not (output / "plot_manifest.json").exists()


def test_group_outputs_keep_original_names_and_exclude_removed_bars(tmp_path):
    from r_plotting import render_enrichment_plots, collect_plot_outputs
    inputs = tmp_path / "enrichment_results"
    for group in ("all", "up", "down"):
        (inputs / f"Case_A_{group}_regulated").mkdir(parents=True)
    calls = []
    def runner(script, args, **kwargs):
        assert len(args) == 2
        source, out = map(Path, args)
        calls.append(source.name)
        out.mkdir(parents=True, exist_ok=True)
        png = out / "pathway_combined_plot.png"
        png.write_bytes(source.name.encode())
        (out / "plot_manifest.json").write_text(json.dumps({"status":"success", "files":[str(png)]}))
    manifest = render_enrichment_plots("original.R", tmp_path, inputs, "Case A", runner=runner)
    assert calls == [f"Case_A_{g}_regulated" for g in ("all", "up", "down")]
    old = tmp_path / "enrichment_results/enrichment_plots/KEGG_2021_Human_up_regulated.png"
    _, report = collect_plot_outputs(tmp_path, [], manifest["files"] + [str(old)])
    assert report["enrichment_bar_plots"] == []
    assert len(report["kegg_pathway_plots"]) == 3
    for group, subdir in (("all", ""), ("up", "up"), ("down", "down")):
        assert (tmp_path / "plots" / subdir / "pathway_combined_plot.png").read_bytes() == f"Case_A_{group}_regulated".encode()


def test_existing_enrichment_uses_original_script_for_three_groups(tmp_path):
    import hashlib
    import re
    from r_plotting import render_enrichment_plots, collect_plot_outputs
    session = os.environ.get("OMIC_EXISTING_PLOT_SESSION")
    if not session:
        pytest.skip("Set OMIC_EXISTING_PLOT_SESSION to reuse existing enrichment CSVs")
    inputs = Path(session) / "enrichment_results"
    paths = list(inputs.glob("*regulated/*_results.csv")) + list(inputs.glob("*regulated/gene_list.txt"))
    def snapshot():
        return {str(p): (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns) for p in paths}
    before = snapshot()
    script = Path(__file__).resolve().parents[3] / "enrichment/kegg_simple.R"
    manifest = render_enrichment_plots(script, tmp_path, inputs, "Alzheimer's Disease")
    assert snapshot() == before
    assert manifest["groups"] == dict.fromkeys(("all", "up", "down"), "success")
    html, report = collect_plot_outputs(tmp_path, [], manifest["files"])
    assert len(html) == 6 and len(report["kegg_pathway_plots"]) == 6
    assert report["enrichment_bar_plots"] == []
    for page in html:
        refs = re.findall(r'(?:src|href)="([^"]+)"', Path(page).read_text())
        for ref in refs:
            if not ref.startswith(("http", "data:", "#")):
                assert (Path(page).parent / ref).is_file()
    (tmp_path / "verification.json").write_text(json.dumps({"inputs_unchanged": True,
        "protected_files": len(paths), "html": html, "report": report}, indent=2))

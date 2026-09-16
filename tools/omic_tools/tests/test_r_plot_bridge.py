"""R artifact publication and stale-manifest handling, without Python drawing."""
import json
from pathlib import Path
import sys

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

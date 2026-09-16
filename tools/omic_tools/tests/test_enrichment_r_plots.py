"""Real R rendering contract; run these tests inside a Slurm allocation."""
import csv
import json
from pathlib import Path
import re
import shutil
import subprocess

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "enrichment" / "kegg_simple.R"
COLUMNS = ["Rank", "Term", "P-value", "Odds Ratio", "Combined Score", "Genes",
           "Adjusted P-value", "Old P-value", "Old Adjusted P-value"]
DATABASES = ("Reactome_2022", "KEGG_2021_Human")
pytestmark = pytest.mark.skipif(shutil.which("Rscript") is None, reason="Rscript required")


def table(directory, database="KEGG_2021_Human", empty=False):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{database}_results.csv"
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(COLUMNS)
        if not empty:
            for i in range(12):
                writer.writerow([i + 1, f"Pathway {i} (human)", .01, 2, 4,
                                 "['APOE', 'APP']", 0 if i == 0 else i / 100, 0, 0])
    return path


def run(base, output, *args):
    result = subprocess.run(["Rscript", str(SCRIPT), str(base), str(output), *map(str, args)],
                            text=True, capture_output=True, timeout=180)
    manifest = output / "plot_manifest.json"
    return result, json.loads(manifest.read_text()) if manifest.exists() else None


def test_extended_cli_renders_all_directions_and_portable_html(tmp_path):
    root, output, bars = tmp_path / "input", tmp_path / "output", tmp_path / "bars"
    for direction in ("all", "up", "down"):
        for db in DATABASES:
            table(root / f"Case_A_{direction}_regulated", db)
    result, manifest = run(root / "Case_A_all_regulated", output,
                           "--enrichment-root", root, "--comparison-name", "Case A",
                           "--bar-output-dir", bars)
    assert result.returncode == 0, result.stderr
    assert manifest and manifest["status"] == "success"
    expected = {f"{db}_{direction}_regulated.{ext}" for db in DATABASES
                for direction in ("all", "up", "down") for ext in ("png", "pdf")}
    assert {p.name for p in bars.iterdir()} == expected
    for path in map(Path, manifest["files"]):
        assert path.is_absolute() and path.exists()
    for filename in ("kegg_dotplot.png", "pathway_combined_plot.png"):
        assert (output / filename).read_bytes().startswith(b"\x89PNG")
    for html in output.glob("*.html"):
        refs = re.findall(r'(?:src|href)="([^"]+)"', html.read_text())
        local = [ref for ref in refs if not ref.startswith(("http", "data:", "#"))]
        assert local
        assert all(not Path(ref).is_absolute() and (output / ref).exists() for ref in local)
    assert len(list(output.glob("*.html"))) == 2
    assert all(p.read_bytes().startswith(b"%PDF") for p in bars.glob("*.pdf"))


def test_legacy_cli_preserves_four_outputs(tmp_path):
    table(tmp_path / "input")
    result, manifest = run(tmp_path / "input", tmp_path / "output")
    assert result.returncode == 0, result.stderr
    assert manifest and manifest["status"] == "success"
    assert {"kegg_dotplot.png", "kegg_dotplot.html", "pathway_combined_plot.png",
            "pathway_combined_plot.html"} <= {Path(p).name for p in manifest["files"]}


@pytest.mark.parametrize("kind", ["missing", "empty", "failed", "running", "malformed"])
def test_empty_and_failed_inputs_never_publish_stale_success(tmp_path, kind):
    base, output = tmp_path / "input", tmp_path / "output"
    output.mkdir()
    (output / "plot_manifest.json").write_text('{"status":"success","files":["stale"]}')
    if kind != "missing":
        csv_path = table(base, empty=kind == "empty")
        if kind in ("failed", "running"):
            (base / "enrichment_status.json").write_text(json.dumps({"status": kind}))
        elif kind == "malformed":
            csv_path.write_text("Term,Genes\nBroken,APP\n")
    result, manifest = run(base, output)
    assert manifest and manifest["files"] == []
    if kind in ("failed", "running", "malformed"):
        assert result.returncode != 0
        assert manifest["status"] == "failed"
    else:
        assert result.returncode == 0, result.stderr
        assert manifest["status"] == "empty"


@pytest.mark.parametrize("scope", ["aggregate", "other_group", "library"])
def test_rejects_failure_status_before_any_render(tmp_path, scope):
    root, output = tmp_path / "input", tmp_path / "output"
    base = root / "Case_A_all_regulated"
    table(base)
    directory = root if scope == "aggregate" else root / "Case_A_up_regulated" if scope == "other_group" else base
    directory.mkdir(exist_ok=True)
    status = {"status": "failed"} if scope != "library" else {
        "status": "success", "libraries": {"KEGG_2021_Human": {"status": "running"}}}
    (directory / "enrichment_status.json").write_text(json.dumps(status))
    result, manifest = run(base, output, "--enrichment-root", root,
                           "--comparison-name", "Case A", "--bar-output-dir", tmp_path / "bars")
    assert result.returncode != 0
    assert manifest["status"] == "failed" and manifest["files"] == []
    assert not list(output.glob("*.png"))


def test_empty_status_ignores_stale_csv(tmp_path):
    base = tmp_path / "input"
    table(base)
    (base / "enrichment_status.json").write_text('{"status":"empty"}')
    result, manifest = run(base, tmp_path / "output")
    assert result.returncode == 0, result.stderr
    assert manifest["status"] == "empty" and manifest["files"] == []

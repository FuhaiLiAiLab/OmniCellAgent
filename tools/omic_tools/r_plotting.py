"""Invoke R renderers and publish their artifacts; no Python drawing code."""
import json
from pathlib import Path
import shutil

if __package__:
    from .subprocess_r import run_r_script
else:
    from subprocess_r import run_r_script


def read_plot_manifest(path, allowed_roots, *, allow_empty=False):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("status") not in ("success", "empty"):
        raise ValueError("R did not return a successful/empty plot manifest")
    files = payload.get("files")
    if not isinstance(files, list) or not all(isinstance(value, str) for value in files):
        raise ValueError("Plot manifest files must be an array of paths")
    if payload["status"] == "empty":
        if not allow_empty or files:
            raise ValueError("Unexpected empty plot result")
        return payload
    if not files:
        raise ValueError("Successful plot manifest contains no artifacts")
    roots = [Path(root).resolve() for root in allowed_roots]
    resolved = []
    for value in files:
        artifact = Path(value).resolve()
        if not any(artifact.is_relative_to(root) for root in roots):
            raise ValueError(f"Plot artifact is outside the requested output directories: {artifact}")
        if not artifact.is_file() or artifact.stat().st_size == 0:
            raise FileNotFoundError(f"Missing or empty R plot artifact: {artifact}")
        resolved.append(str(artifact))
    payload["files"] = list(dict.fromkeys(resolved))
    return payload


def publish_plot_files(files, directory_map):
    """Preserve relative paths for HTML/JS/CSS dependencies across publication."""
    mappings = [(Path(source).resolve(), Path(target).resolve()) for source, target in directory_map.items()]
    planned = []
    for value in files:
        source = Path(value).resolve()
        destinations = [target / source.relative_to(root) for root, target in mappings if source.is_relative_to(root)]
        if len(destinations) != 1:
            raise ValueError(f"Plot artifact does not map to exactly one destination: {source}")
        planned.append((source, destinations[0]))
    published = []
    for source, target in planned:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        published.append(str(target))
    return published


def render_analysis_plots(r_script, session_dir, *, ref_label, alt_label,
                          ref_csv=None, alt_csv=None, input_scale="linear_cp10k", timeout=3600):
    session = Path(session_dir).resolve()
    out = session / "casestudy_R"
    volcano = session / "volcano_plots"
    manifest = out / "plot_manifest.json"
    manifest.unlink(missing_ok=True)
    args = ["--stage", "plots", str(session), str(out),
                 "--state-file", str(out / "analysis_state.rds"),
                 "--volcano-dir", str(volcano), "--ref-label", ref_label,
                 "--alt-label", alt_label, "--input-scale", input_scale]
    for option, value in (("--ref-csv", ref_csv), ("--alt-csv", alt_csv)):
        if value is not None:
            args.extend([option, str(value)])
    run_r_script(str(r_script), args, timeout=timeout)
    return read_plot_manifest(manifest, [out, volcano])


def collect_plot_outputs(session_dir, de_files, enrichment_files):
    """Collect only declared figures, excluding widget assets and old files."""
    root = Path(session_dir).resolve()
    report = {"volcano_plots": [], "case_study_plots": [],
              "enrichment_bar_plots": [], "kegg_pathway_plots": [], "all_plots": []}
    html_paths = []
    for value in sorted(set(de_files + enrichment_files)):
        path = Path(value).resolve()
        relative = path.relative_to(root)
        parts = relative.parts
        if len(parts) == 2 and parts[0] == "casestudy_R":
            category, kind = "case_study_plots", "case_study"
        elif len(parts) == 3 and parts[:2] == ("enrichment_results", "enrichment_plots"):
            category, kind = "enrichment_bar_plots", "enrichment_bar"
        elif len(parts) == 2 and parts[0] == "plots":
            category, kind = "kegg_pathway_plots", "kegg_pathway"
        else:
            continue
        if path.suffix == ".html":
            html_paths.append(str(path))
        if path.suffix == ".png":
            report[category].append({"name": path.stem.replace("_", " ").title(),
                "filename": path.name, "relative_path": relative.as_posix(),
                "absolute_path": str(path), "type": kind, "format": "png"})
            report["all_plots"].append(relative.as_posix())
    return html_paths, report

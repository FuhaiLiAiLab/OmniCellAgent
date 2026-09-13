"""Scientific stages shared by the unified and initial-comparison entry points.

Read in order: create_raw_run → select_donor_cohort → retrieve_and_aggregate
→ fit_initial_comparisons. All count matrices remain raw until DESeq2.
"""

from dataclasses import dataclass
from types import SimpleNamespace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import traceback

import pandas as pd

from .aggregation import aggregate_raw_counts
from .metadata import apply_donor_exclusions, build_donor_metadata, select_cohort
from .retrieval import file_sha256, load_raw_gene_counts

COMPARISONS = ["ad_vs_control", "ad_male_vs_female", "control_male_vs_female"]


def _write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, default=str, allow_nan=False) + "\n", encoding="utf-8")


def _save_vocabularies(metadata, output_dir, disease, cell_type):
    for label, column, requested, filename in [
        ("disease", "disease_BMG_name", disease, "available_diseases.txt"),
        ("cell type", "CMT_name", cell_type, "available_cell_types.txt"),
    ]:
        values = sorted({str(value) for value in metadata[column].dropna().unique() if str(value).strip()},
                        key=lambda value: (value.lower(), value))
        matches = [value for value in values if requested and value.lower() == str(requested).lower()]
        header = ", ".join(json.dumps(value) for value in matches) or "NONE"
        (output_dir / filename).write_text(f"Matched {label} in list: {header}\n" + "\n".join(values) + "\n")


def _audit_previous_session(session, output_dir):
    if session is None:
        return None
    path = Path(session) / "labels.csv"
    if not path.exists():
        return {"status": "unavailable", "path": str(path)}
    labels = pd.read_csv(path, keep_default_na=False)
    donors, _, issues = build_donor_metadata(labels)
    issues.to_csv(output_dir / "previous_session_donor_issues.csv", index=False)
    report = {"path": str(path.resolve()), "sha256": file_sha256(path), "metacells": len(labels),
              "donor_keys": len(donors), "groups": donors.groupby("disease").size().to_dict(),
              "conflicts": {field: int(donors[f"{field}_conflict"].sum()) for field in ["sex", "age", "disease"]}}
    _write_json(output_dir / "previous_session_audit.json", report)
    return report


def _run_deseq2(rscript, counts, metadata, comparison_root, log_path, alpha,
                 min_count, min_gene_donors, min_group_donors, timeout=7200):
    executable = shutil.which(str(rscript)) if rscript else shutil.which("Rscript")
    if not executable:
        raise FileNotFoundError("Rscript not found; provide --rscript for an R environment containing DESeq2 and jsonlite")
    executable = Path(executable).resolve()
    env = os.environ.copy()
    # Do not mix an isolated conda R with an inherited, incompatible shared library.
    library = executable.parent.parent / "lib/R/library"
    if library.is_dir():
        env["R_LIBS"] = str(library)
        env["R_LIBS_USER"] = str(library)
        env["R_LIBS_SITE"] = str(library)
    env.pop("R_HOME", None)
    command = [str(executable), "--vanilla", str(Path(__file__).with_name("deseq2_analysis.R")),
               str(counts), str(metadata), str(comparison_root), str(alpha), str(min_count),
               str(min_gene_donors), str(min_group_donors)]
    print("[DESeq2] Running exactly three donor-level comparisons", flush=True)
    with Path(log_path).open("w", encoding="utf-8") as handle:
        result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT,
                                env=env, cwd=Path(log_path).parent, timeout=timeout, check=False)
    statuses = {}
    for name in COMPARISONS:
        path = comparison_root / name / "status.json"
        statuses[name] = (json.loads(path.read_text()) if path.exists() else
                          {"status": "error", "reason": f"Missing DESeq2 status; inspect {log_path}"})
    return {"command": command, "exit_code": result.returncode, "comparisons": statuses}



@dataclass
class DonorCohort:
    """Aligned metacell rows, donor rows, and metacell-to-donor assignments."""
    metacells: pd.DataFrame
    donors: pd.DataFrame
    assignments: pd.DataFrame


@dataclass
class RawPipelineRun:
    """Paths, settings and audit manifest for one newly created raw run."""
    output_dir: Path
    settings: SimpleNamespace
    manifest: dict

    @property
    def inputs_dir(self):
        return self.output_dir / "inputs"

    def save(self):
        _write_json(self.output_dir / "run_manifest.json", self.manifest)

    def finish(self, status):
        self.manifest.update(status=status, finished_utc=datetime.now(timezone.utc).isoformat())
        self.save()
        return self.manifest

    def fail(self, error):
        self.manifest["error"] = str(error)
        self.finish("error")
        (self.output_dir / "error.log").write_text(traceback.format_exc(), encoding="utf-8")


def create_raw_run(*, output_dir, data_root, celltosg_root, hgnc_reference,
                 disease="Alzheimer's Disease", cell_type="astrocyte", organ="brain", tissue=None,
                 rscript=None, chunk_size=64, alpha=0.05, min_count=10,
                 min_gene_donors=3, min_group_donors=3, prepare_only=False,
                 skip_enrichment=False, skip_plots=False, audit_session=None,
                 exclude_donor_keys=None):
    """Reserve a new folder and snapshot source identities before retrieval."""
    settings = SimpleNamespace(**locals())
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
                "pipeline": "donor_level_pseudobulk_deseq2", "output_dir": str(output_dir),
                "comparison_names": COMPARISONS, "identity_columns": ["source", "dataset_id", "donor_id"],
                "normalization": "DESeq2 median-of-ratios; no normalization before donor aggregation",
                "settings": {"disease": disease, "cell_type": cell_type, "organ": organ, "tissue": tissue,
                             "alpha": alpha, "min_count": min_count, "min_gene_donors": min_gene_donors,
                             "min_group_donors": min_group_donors, "chunk_size": chunk_size,
                             "exclude_donor_keys": str(exclude_donor_keys) if exclude_donor_keys else None,
                             "skip_enrichment": skip_enrichment, "skip_plots": skip_plots,
                             "prepare_only": prepare_only}}
    manifest_path = output_dir / "run_manifest.json"
    _write_json(manifest_path, manifest)
    run = RawPipelineRun(output_dir, settings, manifest)
    try:
        if not 0 < alpha < 1 or min_count < 1 or min_gene_donors < 1 or min_group_donors < 3:
            raise ValueError("Invalid significance or donor/count filtering parameters")
        data_directory = Path(data_root).resolve()
        inputs_dir = output_dir / "inputs"
        inputs_dir.mkdir()
        hgnc_reference_path = Path(hgnc_reference).resolve()
        metadata_path = data_directory / "cell_metadata_with_mappings.parquet"
        manifest["input_files"] = {
            "metadata": {"path": str(metadata_path), "sha256": file_sha256(metadata_path)},
            "hgnc": {"path": str(hgnc_reference_path), "sha256": file_sha256(hgnc_reference_path)},
            "bmg_mapping": {"path": str(data_directory / "bmg_gene_index.csv"),
                            "sha256": file_sha256(data_directory / "bmg_gene_index.csv")},
        }
        shutil.copyfile(hgnc_reference_path, inputs_dir / "hgnc_reference.tsv")
        return run
    except Exception as error:
        run.fail(error)
        raise


def select_donor_cohort(run: RawPipelineRun) -> DonorCohort:
    """Select exact case/control matches and keep annotation conflicts explicit.

    The output assignments follow metacell row order. Individuals are identified
    by (source, dataset_id, donor_id), never by donor_id alone.
    """
    settings = run.settings
    disease, cell_type = settings.disease, settings.cell_type
    organ, tissue = settings.organ, settings.tissue
    exclude_donor_keys, audit_session = settings.exclude_donor_keys, settings.audit_session
    output_dir, inputs_dir = run.output_dir, run.inputs_dir
    metadata_path = Path(settings.data_root).resolve() / "cell_metadata_with_mappings.parquet"
    manifest = run.manifest
    manifest_path = output_dir / "run_manifest.json"
    full_metadata = pd.read_parquet(metadata_path)
    _save_vocabularies(full_metadata, output_dir, disease, cell_type)
    selected_metacells, selection_audit = select_cohort(full_metadata, disease, cell_type, organ, tissue)
    del full_metadata
    _write_json(inputs_dir / "cohort_selection.json", selection_audit)
    selected_metacells.to_csv(inputs_dir / "queried_metacell_metadata.csv", index=False)
    if exclude_donor_keys:
        candidate_donors, _, candidate_issues = build_donor_metadata(selected_metacells, disease_name=disease)
        candidate_issues.to_csv(inputs_dir / "queried_donor_metadata_issues.csv", index=False)
        candidate_donors.to_csv(inputs_dir / "queried_donor_metadata.csv", index=False)
        exclusion_path = Path(exclude_donor_keys).resolve()
        exclusion_table = pd.read_csv(exclusion_path, dtype=str, keep_default_na=False)
        selected_metacells, removed, exclusion_audit = apply_donor_exclusions(selected_metacells, exclusion_table)
        removed.to_csv(inputs_dir / "excluded_metacell_metadata.csv", index=False)
        shutil.copyfile(exclusion_path, inputs_dir / "explicit_donor_exclusions.csv")
        exclusion_audit.update(path=str(exclusion_path), sha256=file_sha256(exclusion_path))
        _write_json(inputs_dir / "explicit_donor_exclusion_audit.json", exclusion_audit)
        manifest["explicit_donor_exclusions"] = exclusion_audit
        print(f"[Exclusions] Explicit list removed {exclusion_audit['excluded_donor_keys']} donor keys / "
              f"{exclusion_audit['excluded_metacells']} metacells", flush=True)
    selected_metacells.to_csv(inputs_dir / "retrieved_metacell_metadata.csv", index=False)
    donor_metadata, assignments, issues = build_donor_metadata(selected_metacells, disease_name=disease)
    assignments.to_csv(inputs_dir / "metacell_to_donor.csv", index=False)
    issues.to_csv(inputs_dir / "donor_metadata_issues.csv", index=False)
    manifest["cohort_selection"] = selection_audit
    manifest["donor_metadata_audit"] = {
        "donors": len(donor_metadata),
        "groups": donor_metadata.groupby("disease").size().to_dict(),
        "conflicts": {field: int(donor_metadata[f"{field}_conflict"].sum()) for field in ["sex", "age", "disease"]},
        "missing_or_conflicted": {field: int(donor_metadata[field].isna().sum()) for field in ["sex", "age", "disease"]},
    }
    manifest["previous_session_audit"] = _audit_previous_session(audit_session, inputs_dir)
    print(f"[Cohort] {len(selected_metacells):,} metacells, {len(donor_metadata):,} donor keys", flush=True)
    print(f"[Metadata] Conflicts: {manifest['donor_metadata_audit']['conflicts']}; unresolved values remain missing", flush=True)
    _write_json(manifest_path, manifest)
    return DonorCohort(selected_metacells, donor_metadata, assignments)


def retrieve_and_aggregate(run: RawPipelineRun, cohort: DonorCohort) -> dict:
    """Retrieve metacells × genes and export genes × donors raw integer counts.

    Representative BMG columns are chosen across the entire selected cohort.
    Donor summation precedes normalization and must conserve every gene total.
    """
    settings = run.settings
    data_directory, hgnc_reference_path = Path(settings.data_root).resolve(), Path(settings.hgnc_reference).resolve()
    output_dir, inputs_dir = run.output_dir, run.inputs_dir
    celltosg_root, chunk_size = settings.celltosg_root, settings.chunk_size
    selected_metacells, donor_metadata, assignments = cohort.metacells, cohort.donors, cohort.assignments
    manifest = run.manifest
    raw_metacell_counts, gene_symbols, _, retrieval_audit = load_raw_gene_counts(
        selected_metacells, data_directory, celltosg_root, hgnc_reference_path, output_dir / "retrieval", chunk_size=chunk_size
    )
    _write_json(inputs_dir / "retrieval_audit.json", retrieval_audit)
    pseudobulk_counts, conservation_audit = aggregate_raw_counts(raw_metacell_counts, gene_symbols, assignments, donor_metadata)
    del raw_metacell_counts
    donor_metadata["library_size"] = pseudobulk_counts.sum(axis=0).reindex(donor_metadata.sample_id).to_numpy()
    pseudobulk_counts.to_csv(inputs_dir / "pseudobulk_counts.csv")
    donor_metadata.to_csv(inputs_dir / "donor_metadata.csv", index=False)
    conservation_audit["scope"] = "HGNC-filtered CellTOSG representative gene axis"
    _write_json(inputs_dir / "count_conservation.json", conservation_audit)
    manifest["count_conservation"] = conservation_audit
    del pseudobulk_counts
    print(f"[Pseudobulk] Saved {len(gene_symbols):,} genes × {len(donor_metadata):,} donors; raw count totals conserved", flush=True)
    run.save()
    return conservation_audit


def fit_initial_comparisons(run: RawPipelineRun) -> dict:
    """Fit the three original donor contrasts; optionally plot/enrich each.

    Initial DESeq2 results retain adaptive independent filtering. The later
    sensitivity stage deliberately uses its separate fixed-family BH policy.
    """
    settings = run.settings
    output_dir, inputs_dir = run.output_dir, run.inputs_dir
    manifest, manifest_path = run.manifest, run.output_dir / "run_manifest.json"
    rscript, alpha = settings.rscript, settings.alpha
    min_count, min_gene_donors = settings.min_count, settings.min_gene_donors
    min_group_donors = settings.min_group_donors
    skip_plots, skip_enrichment = settings.skip_plots, settings.skip_enrichment
    deseq2_result = _run_deseq2(rscript, inputs_dir / "pseudobulk_counts.csv", inputs_dir / "donor_metadata.csv",
                    output_dir / "comparisons", output_dir / "deseq2.log", alpha, min_count,
                    min_gene_donors, min_group_donors)
    manifest["deseq2"] = deseq2_result
    manifest["downstream"] = {}
    _write_json(manifest_path, manifest)
    for name in COMPARISONS:
        status = deseq2_result["comparisons"][name]
        comparison = output_dir / "comparisons" / name
        print(f"[Comparison] {name}: {status.get('status')}", flush=True)
        downstream = {}
        if status.get("status") == "success":
            if not skip_plots:
                from .plots import plot_comparison
                downstream["plots"] = plot_comparison(comparison)
            if not skip_enrichment:
                from .enrichment import run_enrichment
                downstream["enrichment"] = run_enrichment(comparison, alpha=alpha)
        else:
            downstream["status"] = "skipped_due_to_deseq2_status"
        manifest["downstream"][name] = downstream
        _write_json(manifest_path, manifest)
    all_de_success = all(item.get("status") == "success" for item in deseq2_result["comparisons"].values())
    enrichment_ok = all(item.get("enrichment", {}).get("status", "success") in {"success", "empty"}
                        for item in manifest["downstream"].values())
    manifest["status"] = "success" if all_de_success and enrichment_ok else "partial"
    return run.finish(manifest["status"])


def run_pipeline(*, output_dir, data_root, celltosg_root, hgnc_reference,
                 disease="Alzheimer's Disease", cell_type="astrocyte", organ="brain", tissue=None,
                 rscript=None, chunk_size=64, alpha=0.05, min_count=10,
                 min_gene_donors=3, min_group_donors=3, prepare_only=False,
                 skip_enrichment=False, skip_plots=False, audit_session=None,
                 exclude_donor_keys=None):
    """Compatibility entry point for omic_pseudobulk_workflow.py.

    The unified omic_workflow.py calls these same stages explicitly so readers
    can follow each scientific step without a second hidden orchestration loop.
    """
    run = create_raw_run(**locals())
    try:
        # STEP 1: exact cohort and donor annotation audit.
        cohort = select_donor_cohort(run)
        # STEP 2: raw retrieval, HGNC selection, donor summation and exports.
        retrieve_and_aggregate(run, cohort)
        del cohort
        if prepare_only:
            return run.finish("prepared")
        # STEP 3: initial donor DESeq2 contrasts and optional saved-result plots.
        return fit_initial_comparisons(run)
    except Exception as error:
        run.fail(error)
        raise

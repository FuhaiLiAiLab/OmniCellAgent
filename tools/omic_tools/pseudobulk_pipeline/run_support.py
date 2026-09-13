"""Validation, session paths and run bookkeeping; no scientific computation.

CLI option definitions stay in omic_workflow.py. This module keeps filesystem
and audit-manifest details out of its numbered analysis sequence.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys


@dataclass(frozen=True)
class RunPaths:
    output_dir: Path
    raw_pipeline_dir: Path
    rscript: str | None

    @property
    def sensitivity_dir(self):
        return self.output_dir / "models"

    def comparison_dir(self, model_directory):
        return self.sensitivity_dir / "models" / model_directory


def validate_and_resolve(args, repository_root) -> RunPaths:
    """Reject invalid input before creating output; resolve defaults once."""
    # ==================== Validate before creating any output ================
    if args.session_id and not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', args.session_id):
        raise ValueError('--session-id must start with a letter/digit and contain only letters, digits, underscores, dots, or hyphens')
    if args.session_id and args.output_dir:
        raise ValueError('Use either --session-id or --output-dir')
    if args.disease.lower() != "alzheimer's disease":
        raise ValueError('The current donor pipeline supports Alzheimer\'s Disease only')
    if not math.isfinite(args.alpha) or not 0 < args.alpha < 1:
        raise ValueError('--alpha must be finite and between 0 and 1')
    if min(args.chunk_size, args.min_count, args.min_gene_donors, args.top_kegg, args.top_per_category) < 1 or args.min_group_donors < 3:
        raise ValueError('Count/chunk/plot limits must be positive; --min-group-donors must be at least 3')
    if not math.isfinite(args.log2fc_min) or args.log2fc_min < 0:
        raise ValueError('--log2fc-min must be finite and nonnegative')
    if not math.isfinite(args.enrichment_timeout) or args.enrichment_timeout <= 0 or args.enrichment_retries < 0:
        raise ValueError('Enrichment timeout must be positive and finite; retries must be nonnegative')
    if args.prepare_only and args.source_run:
        raise ValueError('--prepare-only cannot be combined with --source-run')
    rscript = shutil.which(args.rscript)
    if not args.prepare_only and not rscript:
        raise FileNotFoundError(f'Rscript not found: {args.rscript}')
    sessions_root = (repository_root / 'webapp' / 'sessions').resolve()
    default_output = sessions_root / (args.session_id or f'ad_complete_rerun_{datetime.now():%Y%m%d_%H%M%S}')
    output = Path(args.output_dir or default_output).resolve()
    if output == sessions_root or not output.is_relative_to(sessions_root):
        raise ValueError('--output-dir must be a new directory under webapp/sessions; use --session-id for a session name')
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite existing folder: {output}')
    source = Path(args.source_run).resolve() if args.source_run else output/'raw_pipeline'
    if args.source_run:
        if not source.is_dir():
            raise FileNotFoundError(f'Source run not found: {source}')
        if output.is_relative_to(source):
            raise ValueError('Output must be outside the source run')
    else:
        for value in (args.hgnc_reference, args.exclude_donor_keys):
            if not Path(value).is_file():
                raise FileNotFoundError(f'Input file not found: {value}')
        if not args.data_root or not args.celltosg_root:
            from utils.path_config import get_path
            args.data_root = args.data_root or get_path('external.omnicell_data_root', absolute=True)
            args.celltosg_root = args.celltosg_root or get_path('external.omnicell_root', absolute=True)

    return RunPaths(output, source, rscript)


def shared_de_settings(args):
    """Use identical count thresholds in initial and sensitivity fits."""
    return {
        "alpha": args.alpha,
        "min_count": args.min_count,
        "min_gene_donors": args.min_gene_donors,
        "min_group_donors": args.min_group_donors,
    }


def raw_stage_arguments(args, paths):
    """Translate CLI options to the existing raw-stage interface explicitly."""
    return {
        "output_dir": str(paths.raw_pipeline_dir),
        "data_root": args.data_root,
        "celltosg_root": args.celltosg_root,
        "hgnc_reference": args.hgnc_reference,
        "disease": args.disease,
        "organ": args.organ,
        "cell_type": args.cell_type,
        "tissue": args.tissue,
        "rscript": paths.rscript,
        "chunk_size": args.chunk_size,
        "prepare_only": args.prepare_only or not args.full_sensitivity,
        "skip_enrichment": True,
        "skip_plots": not args.raw_plots,
        "audit_session": args.audit_session,
        "exclude_donor_keys": args.exclude_donor_keys,
        **shared_de_settings(args),
    }


class RunReport:
    """Persist stage statuses and failures in the new session only."""

    def __init__(self, paths, args):
        self.paths = paths
        paths.output_dir.mkdir(parents=True, exist_ok=False)
        self.manifest = {
            "status": "running",
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "output_dir": str(paths.output_dir),
            "source_run": str(paths.raw_pipeline_dir),
            "settings": vars(args).copy(),
            "stages": {},
        }
        self.save()

    def save(self):
        path = self.paths.output_dir / "workflow_manifest.json"
        path.write_text(json.dumps(self.manifest, indent=2, default=str, allow_nan=False) + "\n")

    def record(self, name, result, allowed=("success",)):
        self.manifest["stages"][name] = result
        self.save()
        if result.get("status") not in allowed:
            raise RuntimeError(f"{name} stage returned {result.get('status')}; inspect {self.paths.output_dir}")

    def finish(self, status="success", error=None):
        self.manifest.update(status=status, finished_utc=datetime.now(timezone.utc).isoformat())
        if error is not None:
            self.manifest["error"] = str(error)
        self.save()
        return self.manifest


def configure_environment(repository_root):
    """Use persistent project caches; callers can override paths explicitly."""
    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["PYTHONPATH"] = str(repository_root)
    cache_root = repository_root / "dataset_outputs" / "pseudobulk_runtime_cache"
    for variable, subdirectory in (
        ("MPLCONFIGDIR", "matplotlib"),
        ("NUMBA_CACHE_DIR", "numba"),
        ("TMPDIR", "working"),
    ):
        location = Path(os.environ.setdefault(variable, str(cache_root / subdirectory)))
        location.mkdir(parents=True, exist_ok=True)

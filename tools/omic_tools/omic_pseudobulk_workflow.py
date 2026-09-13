"""New standalone CellTOSG donor-level DESeq2 pipeline; prior workflows are untouched."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
import uuid

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tools.omic_tools.pseudobulk_pipeline.pipeline_stages import run_pipeline


def main(argv=None):
    from tools.omic_tools.pseudobulk_pipeline.run_support import configure_environment
    configure_environment(REPO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--disease", default="Alzheimer's Disease", help="Exact disease_BMG_name for the AD comparison")
    parser.add_argument("--cell-type", default="astrocyte")
    parser.add_argument("--organ", default="brain")
    parser.add_argument("--tissue")
    parser.add_argument("--data-root")
    parser.add_argument("--celltosg-root")
    parser.add_argument("--hgnc-reference", default=str(Path(__file__).with_name("reference") / "hgnc_protein_coding.tsv"))
    parser.add_argument("--output-dir", help="New directory under webapp/sessions; existing paths are rejected")
    parser.add_argument("--rscript", help="Rscript in an environment containing DESeq2 and jsonlite")
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--min-count", type=int, default=10)
    parser.add_argument("--min-gene-donors", type=int, default=3)
    parser.add_argument("--min-group-donors", type=int, default=3)
    parser.add_argument("--audit-session", default=str(REPO / "webapp/sessions/alzheimer_test_910"))
    parser.add_argument("--exclude-donor-keys", help="Explicit reviewed CSV with source,dataset_id,donor_id,reason; excludes whole donor keys")
    parser.add_argument("--prepare-only", action="store_true", help="Export raw donor pseudobulk and audits without DE")
    parser.add_argument("--skip-enrichment", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args(argv)
    if args.disease.lower() != "alzheimer's disease":
        parser.error("This pipeline implements the three AD comparisons; use the exact Alzheimer's Disease label")
    from utils.path_config import get_path
    args.data_root = args.data_root or get_path("external.omnicell_data_root", absolute=True)
    args.celltosg_root = args.celltosg_root or get_path("external.omnicell_root", absolute=True)
    if not args.output_dir:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        args.output_dir = str(REPO / "webapp/sessions" /
                              f"pseudobulk_ad_{stamp}_{uuid.uuid4().hex[:8]}")
    sessions_root = (REPO / "webapp/sessions").resolve()
    output = Path(args.output_dir).resolve()
    if output == sessions_root or not output.is_relative_to(sessions_root):
        parser.error("--output-dir must be a new directory under webapp/sessions")
    try:
        manifest = run_pipeline(**vars(args))
    except Exception as exc:
        print(f"Pipeline failed: {exc}", file=sys.stderr)
        return 1
    print(f"Pipeline status: {manifest['status']}\nOutputs: {manifest['output_dir']}")
    return 0 if manifest["status"] in {"success", "prepared"} else 2


if __name__ == "__main__":
    raise SystemExit(main())

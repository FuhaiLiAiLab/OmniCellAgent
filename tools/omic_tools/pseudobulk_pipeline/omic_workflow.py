#!/usr/bin/env python3
"""Unified raw CellTOSG → donor DESeq2 → sensitivity → Enrichr workflow.

Run this file with the omnicellagent Python environment. All user options are
defined below, grouped by component. By default, only the selected model is
fitted. Use --full-sensitivity to run the original eleven-fit audit sequence.
"""

import argparse
import json
import os
from pathlib import Path
import re
import sys

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from tools.omic_tools.pseudobulk_pipeline.run_support import (
    RunReport, configure_environment, raw_stage_arguments,
    shared_de_settings, validate_and_resolve,
)

MODEL_NAMES = {
    'disease-only': 'full_A',
    'study-disease': 'full_B',
    'fully-adjusted': 'full_C',
    'shared-studies-adjusted': 'shared_C',
}


def model_name(value):
    """Translate readable CLI names to the existing saved-model directory names."""
    if value in MODEL_NAMES:
        return MODEL_NAMES[value]
    if value in MODEL_NAMES.values() or re.fullmatch(r'leave_one_out_[0-9]{2,}', value):
        return value
    raise argparse.ArgumentTypeError('Choose ' + ', '.join(MODEL_NAMES) +
                                     ' (legacy model directory names also accepted)')


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # ==================== 1. Run paths and stage controls ====================
    run = parser.add_argument_group('1. Run paths and stage controls')
    destination = run.add_mutually_exclusive_group()
    destination.add_argument('--session-id', help='New session name; saves under webapp/sessions/SESSION_ID')
    destination.add_argument('--output-dir', help='New folder under webapp/sessions; without either option: webapp/sessions/ad_complete_rerun_TIMESTAMP')
    run.add_argument('--rscript', default=os.environ.get('OMIC_RSCRIPT', os.environ.get(
        'PSEUDOBULK_RSCRIPT', '/storage1/fs1/fuhai.li/Active/di.huang/cache/R/omic-deseq2/bin/Rscript')),
        help='Rscript with DESeq2/jsonlite, used by initial and sensitivity models')
    run.add_argument('--source-run', help='Reuse a raw_pipeline folder; selected-only mode needs only its raw count/metadata inputs')
    run.add_argument('--full-sensitivity', action='store_true',
                     help='Run the initial three contrasts and all sensitivity models; default fits only the selected model')
    run.add_argument('--prepare-only', action='store_true', help='Retrieve and aggregate only; skip all DE and downstream stages')
    run.add_argument('--skip-enrichment', action='store_true', help='Skip final Enrichr submission and its plots')

    # ==================== 2. Raw retrieval and donor aggregation =============
    raw = parser.add_argument_group('2. Raw retrieval and donor aggregation')
    raw.add_argument('--disease', default="Alzheimer's Disease", help='Exact disease label; current models support AD only')
    raw.add_argument('--organ', default='brain', help='Exact tissue_general match')
    raw.add_argument('--cell-type', default='astrocyte', help='Exact CMT_name match')
    raw.add_argument('--tissue', help='Optional exact tissue match')
    raw.add_argument('--data-root', help='CellTOSG dataset directory; default from repository path config')
    raw.add_argument('--celltosg-root', help='CellTOSG code directory; default from repository path config')
    raw.add_argument('--hgnc-reference', default=str(REPO/'tools/omic_tools/reference/hgnc_protein_coding.tsv'),
                     help='Approved HGNC protein-coding reference')
    raw.add_argument('--exclude-donor-keys', default=str(REPO/'dataset_outputs/pseudobulk_source_qc/approved_donor_exclusions.csv'),
                     help='Reviewed donor exclusion CSV; default is the approved 11-donor exclusion list')
    raw.add_argument('--chunk-size', type=int, default=64, help='Raw metacells per read chunk')
    raw.add_argument('--audit-session', default=str(REPO/'webapp/sessions/alzheimer_test_910'),
                     help='Old session for a read-only donor audit; never used as expression input')

    # ==================== 3. DESeq2: initial AND sensitivity models ==========
    de = parser.add_argument_group('3. Shared DESeq2 settings (initial + sensitivity models)')
    de.add_argument('--alpha', type=float, default=.05,
                    help='DE FDR, enrichment foreground selection, and enrichment plot adjusted-P threshold')
    de.add_argument('--min-count', type=int, default=10, help='Minimum raw gene count per qualifying donor')
    de.add_argument('--min-gene-donors', type=int, default=3, help='Donors required to meet min-count for gene retention')
    de.add_argument('--min-group-donors', type=int, default=3, help='Minimum donors in each comparison group (at least 3)')

    # ==================== 4. Initial donor expression plots ==================
    plots = parser.add_argument_group('4. Initial donor expression plots')
    plots.add_argument('--raw-plots', action='store_true', help='Enable selected-model PCA/volcano/violin plots (initial models in full-sensitivity mode); default is off')
    plots.add_argument('--skip-plots', dest='raw_plots', action='store_false',
                       help='Explicitly disable initial expression plots; sensitivity/enrichment plots still run')
    parser.set_defaults(raw_plots=False)

    # ==================== 5. Final enrichment and enrichment plots ===========
    enrich = parser.add_argument_group('5. Final Enrichr analysis and plots')
    enrich.add_argument('--enrichment-model', '--enrichement-model', dest='enrichment_model',
                        default='disease-only', type=model_name, metavar='MODEL',
                        help='disease-only; study-disease; fully-adjusted (study+age+sex+disease); shared-studies-adjusted (same model in shared studies). Both flag spellings are accepted')
    enrich.add_argument('--log2fc-min', type=float, default=0, help='Minimum absolute DE log2FC for enrichment; all/up/down analyzed separately, no gene cap')
    enrich.add_argument('--enrichment-timeout', type=float, default=30, help='Seconds per Enrichr HTTP request')
    enrich.add_argument('--enrichment-retries', type=int, default=2, help='Retries after each initial HTTP request')
    enrich.add_argument('--top-kegg', type=int, default=20, help='Maximum significant KEGG terms displayed; tables retain all terms')
    enrich.add_argument('--top-per-category', type=int, default=5, help='Maximum terms per category in combined enrichment plots')
    return parser


# Component imports are lazy so --help does not load scientific packages.
def load_pipeline_stages():
    from tools.omic_tools.pseudobulk_pipeline import pipeline_stages
    return pipeline_stages


def run_models(**kwargs):
    from tools.omic_tools.donor_sensitivity.run import execute as sensitivity
    return sensitivity(**kwargs)


def run_selected_model(**kwargs):
    from tools.omic_tools.pseudobulk_pipeline.selected_model import run_selected_model as selected
    return selected(**kwargs)


def run_enrichment(path, **kwargs):
    from tools.omic_tools.pseudobulk_pipeline.enrichment import run_enrichment as enrich
    return enrich(path, **kwargs)


def regenerate_plots(path, **kwargs):
    from tools.omic_tools.pseudobulk_pipeline.enrichment import regenerate_enrichment_plots
    result = regenerate_enrichment_plots(path, **kwargs)
    # Keep this new run's metadata consistent when custom display limits are used.
    metadata_path = Path(path)/'enrichment/metadata.json'
    metadata = json.loads(metadata_path.read_text())
    metadata['plots'] = result
    metadata_path.write_text(json.dumps(metadata, indent=2, allow_nan=False)+'\n')
    return result


def execute(args):
    """Run the scientific stages in order; every output belongs to a new session."""
    # STEP 1: Validate settings and create a fresh session.
    paths = validate_and_resolve(args, REPO)
    report = RunReport(paths, args)
    raw_run = None
    try:
        if not args.source_run:
            stages = load_pipeline_stages()
            raw_run = stages.create_raw_run(**raw_stage_arguments(args, paths))

            # STEP 2: Select exact disease/control matches and resolve donor metadata.
            print("[2/7] Select known-donor metacells and audit annotation conflicts.", flush=True)
            cohort = stages.select_donor_cohort(raw_run)

            # STEP 3: Retrieve raw metacells × genes; sum into genes × donors.
            print("[3/7] Retrieve HGNC-filtered raw counts and aggregate donors.", flush=True)
            stages.retrieve_and_aggregate(raw_run, cohort)
            del cohort  # The count/metadata exports are the next stages' inputs.
            if args.prepare_only:
                report.record("raw_pipeline", raw_run.finish("prepared"), allowed=("prepared",))
                return report.finish("prepared")

            # STEP 4: Optional full audit; a single-model run needs no preliminary DE.
            if args.full_sensitivity:
                print("[4/7] Fit initial donor-level DESeq2 comparisons.", flush=True)
                report.record("raw_pipeline", stages.fit_initial_comparisons(raw_run))
            else:
                print("[4/7] Raw donor counts ready; no preliminary models fitted.", flush=True)
                report.record("raw_pipeline", raw_run.finish("prepared"), allowed=("prepared",))
        else:
            print(f"[2–4/7] Reuse completed raw run: {paths.raw_pipeline_dir}", flush=True)

        # STEP 5: Fit only the selection unless the full audit was requested.
        # Both paths retain the same donor eligibility and fixed-family BH policy.
        model_arguments = dict(source_run=str(paths.raw_pipeline_dir),
                               output_dir=str(paths.sensitivity_dir),
                               rscript=paths.rscript, **shared_de_settings(args))
        if args.full_sensitivity:
            print("[5/7] Fit all donor sensitivity models.", flush=True)
            report.record("sensitivity", run_models(**model_arguments))
        else:
            print(f"[5/7] Fit selected model only: {args.enrichment_model}.", flush=True)
            report.record("selected_model", run_selected_model(
                **model_arguments, model_name=args.enrichment_model,
                export_expression=args.raw_plots,
            ))

        if not args.skip_enrichment:
            # STEP 6: Enrich all/up/down genes from the selected saved model.
            comparison_dir = paths.comparison_dir(args.enrichment_model)
            if not comparison_dir.is_dir():
                raise FileNotFoundError(f"Selected model was not generated: {comparison_dir}")
            readable_model = next((name for name, folder in MODEL_NAMES.items()
                                   if folder == args.enrichment_model), args.enrichment_model)
            print(f"[6/7] Enrich significant genes: {readable_model}.", flush=True)
            report.record("enrichment", run_enrichment(
                str(comparison_dir), alpha=args.alpha, log2fc_min=args.log2fc_min,
                timeout=args.enrichment_timeout, retries=args.enrichment_retries,
            ), allowed=("success", "empty"))

            # STEP 7: Render saved enrichment tables; never refit DE while plotting.
            print("[7/7] Generate enrichment figures and save the run summary.", flush=True)
            report.record("enrichment_plots", regenerate_plots(
                str(comparison_dir), alpha=args.alpha,
                top_kegg=args.top_kegg, top_per_category=args.top_per_category,
            ), allowed=("success", "empty"))
            report.manifest["stages"]["enrichment"]["plots"] = report.manifest["stages"]["enrichment_plots"]
            report.manifest["enrichment_dir"] = str(comparison_dir / "enrichment")
        return report.finish()
    except Exception as error:
        if raw_run is not None and raw_run.manifest["status"] == "running":
            raw_run.fail(error)
        report.finish("error", error=error)
        raise


def main(argv=None):
    args = build_parser().parse_args(argv)
    configure_environment(REPO)
    try:
        result = execute(args)
    except Exception as exc:
        print(f'Workflow failed: {exc}', file=sys.stderr)
        return 1
    print(f"Workflow status: {result['status']}\nRun folder: {result['output_dir']}")
    if 'enrichment_dir' in result:
        print(f"Enrichment results: {result['enrichment_dir']}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

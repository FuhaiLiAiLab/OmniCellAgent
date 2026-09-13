"""Fit one donor model directly from raw pseudobulk exports, without pilot fits."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


def _hash(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def run_selected_model(source_run, output_dir, rscript, model_name,
                       alpha=.05, min_count=10, min_gene_donors=3,
                       min_group_donors=3, export_expression=False):
    """Prepare the established donor cohort and fit exactly one requested model.

    No initial comparison results are needed. Raw donor counts and annotations
    determine eligibility, count filtering and median-of-ratios normalization.
    Optional expression plots use VST from this same fit; they never refit DE.
    """
    source = Path(source_run).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite existing folder: {output}')
    if output.is_relative_to(source):
        raise ValueError('Output must be separate from the source run')
    executable = shutil.which(str(rscript))
    if not executable:
        raise FileNotFoundError(f'Rscript not found: {rscript}')
    inputs = [source/'inputs/pseudobulk_counts.csv', source/'inputs/donor_metadata.csv']
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(f'Raw donor input is missing: {path}')
    before = {str(path): _hash(path) for path in inputs}

    # STEP 1: configure the existing persistent R environment.
    env = os.environ.copy()
    executable = Path(executable).resolve()
    library = executable.parent.parent/'lib/R/library'
    if library.is_dir():
        for variable in ('R_LIBS', 'R_LIBS_USER', 'R_LIBS_SITE'):
            env[variable] = str(library)
    env.pop('R_HOME', None)
    for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        env[variable] = '1'
    command = [str(executable), '--vanilla', str(Path(__file__).with_suffix('.R')),
               str(source), str(output), model_name, str(alpha), str(min_count),
               str(min_gene_donors), str(min_group_donors), str(bool(export_expression)).lower()]
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output/'invocation.json', {
        'command': command, 'input_sha256': before,
        'started_utc': datetime.now(timezone.utc).isoformat(),
        'mode': 'selected_model_only', 'requested_model': model_name,
    })

    try:
        # STEP 2: R prepares normalization without DE, then calls one model fit.
        with (output/'model_run.log').open('w') as log:
            completed = subprocess.run(command, cwd=output, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, timeout=7200, check=False)
        manifest_path = output/'manifest.json'
        if not manifest_path.is_file():
            raise RuntimeError(f'Selected-model runner produced no manifest; inspect {output / "model_run.log"}')
        manifest = json.loads(manifest_path.read_text())
        if completed.returncode != 0 or manifest.get('status') != 'success':
            raise RuntimeError(f'Selected model failed: {manifest.get("reason", manifest.get("status"))}; inspect {manifest_path}')
        if manifest.get('fitted_model_count') != 1 or manifest.get('model_order') != [model_name]:
            raise RuntimeError('Selected-only runner must fit exactly the requested model')

        # STEP 3: summarize saved DE results and optionally plot donor expression.
        import pandas as pd
        comparison = output/'models'/model_name
        frame = pd.read_csv(comparison/'results.csv')
        significant = frame.tested & frame.padj.lt(alpha)
        status = json.loads((comparison/'status.json').read_text())
        summary = output/'summary'
        summary.mkdir()
        counts = {
            'model': model_name, 'formula': status['formula'],
            'AD_donors': status['group_counts']['AD'],
            'control_donors': status['group_counts']['control'],
            'significant_genes': int(significant.sum()),
            'up': int((significant & frame.log2FoldChange.gt(0)).sum()),
            'down': int((significant & frame.log2FoldChange.lt(0)).sum()),
        }
        pd.DataFrame([counts]).to_csv(summary/'model_summary.csv', index=False)
        result = {'status': 'success', 'fitted_model_count': 1, **counts,
                  'comparison_dir': str(comparison)}
        if export_expression:
            from .plots import plot_comparison
            result['plots'] = plot_comparison(comparison)
            if result['plots']['status'] != 'success':
                raise RuntimeError(f'Selected-model expression plotting failed: {result["plots"]}')
        (summary/'report.md').write_text(
            f'# Selected donor model: {model_name}\n\n'
            f'Formula: `{status["formula"]}`. One DESeq2 model fitted.\n\n'
            f'{counts["AD_donors"]} AD and {counts["control_donors"]} control donors; '
            f'{counts["significant_genes"]} genes at fixed-family FDR < {alpha:g} '
            f'({counts["up"]} higher and {counts["down"]} lower in AD).\n\n'
            'Donor eligibility matches the earlier full audit, including complete '
            'study/age/sex annotations even for disease-only. Normalization was '
            'prepared from raw counts without fitting any preliminary DE models.\n'
        )
        _write_json(output/'run_status.json', result)
        return result
    finally:
        changed = [path for path in inputs if _hash(path) != before[str(path)]]
        _write_json(output/'source_preservation.json', {
            'source_files_checked': len(inputs),
            'changed_files': [str(path) for path in changed],
            'source_unchanged': not changed,
        })
        if changed:
            raise RuntimeError('Raw source files changed during the selected-model run')

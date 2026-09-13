"""Run the approved fixed-donor/shared-study DESeq2 sensitivity analysis."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))


def _file_states(directory):
    return {str(path.relative_to(directory)): [path.stat().st_size, path.stat().st_mtime_ns]
            for path in directory.rglob('*') if path.is_file()}


def execute(source_run, output_dir, rscript, alpha=.05, min_count=10,
            min_gene_donors=3, min_group_donors=3):
    from tools.omic_tools.donor_sensitivity.reporting import build_report
    source = Path(source_run).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing folder: {output}")
    if output.is_relative_to(source):
        raise ValueError("Output must be separate from the source run")
    executable = shutil.which(str(rscript))
    if not executable:
        raise FileNotFoundError(f"Rscript not found: {rscript}")
    if not source.is_dir():
        raise FileNotFoundError(f"Source run is missing: {source}")
    before = _file_states(source)
    required = [source/'inputs/pseudobulk_counts.csv',
                source/'comparisons/ad_vs_control/donor_metadata.csv',
                source/'comparisons/ad_vs_control/size_factors.csv',
                source/'comparisons/ad_vs_control/results.csv',
                source/'comparisons/ad_vs_control/normalized_counts.csv']
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    output.mkdir(parents=True)
    env = os.environ.copy()
    prefix = Path(executable).resolve().parent.parent
    library = prefix/'lib/R/library'
    if library.is_dir():
        for name in ['R_LIBS', 'R_LIBS_USER', 'R_LIBS_SITE']:
            env[name] = str(library)
    env.pop('R_HOME', None)
    for name in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
        env[name] = '1'
    command = [str(Path(executable).resolve()), '--vanilla', str(Path(__file__).with_name('models.R')),
               str(source), str(output), str(alpha), str(min_count), str(min_gene_donors), str(min_group_donors)]
    invocation = {'source_run': str(source), 'output_dir': str(output), 'command': command,
                  'started_utc': datetime.now(timezone.utc).isoformat(),
                  'input_sha256': {str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
                                   for path in required}}
    (output/'invocation.json').write_text(json.dumps(invocation, indent=2)+'\n')
    print(f"Running sensitivity fits; output: {output}", flush=True)
    try:
        with (output/'run.log').open('w') as log:
            completed = subprocess.run(command, cwd=output, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, timeout=7200, check=False)
        if not (output/'manifest.json').exists():
            raise RuntimeError(f"R analysis did not produce a manifest; inspect {output/'run.log'}")
        report = build_report(source, output)
        model_manifest = json.loads((output/'manifest.json').read_text())
        statuses = {key: value['status'] for key, value in model_manifest['models'].items()}
        result = {'status': 'success' if completed.returncode == 0 and all(v == 'success' for v in statuses.values()) else 'partial',
                  'models': statuses, 'report': str(output/'summary/report.md'),
                  'de_recomputed_by_reporting': report['de_recomputed_by_reporting']}
        (output/'run_status.json').write_text(json.dumps(result, indent=2)+'\n')
        return result
    finally:
        after = _file_states(source)
        changed = [name for name in before if before[name] != after.get(name)]
        (output/'source_preservation.json').write_text(json.dumps(
            {'source_files_checked': len(before), 'changed_files': changed,
             'source_unchanged': not changed}, indent=2)+'\n')
        if changed:
            raise RuntimeError(f"Source files changed during sensitivity analysis: {changed}")


def main(argv=None):
    from tools.omic_tools.pseudobulk_pipeline.run_support import configure_environment
    configure_environment(REPO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', default=str(REPO/'webapp/sessions/pseudobulk_ad_astrocyte_deseq2_v2'))
    parser.add_argument('--output-dir', default=str(REPO/'webapp/sessions'/f'ad_donor_sensitivity_{datetime.now():%Y%m%d_%H%M%S}'),
                        help='New directory under webapp/sessions')
    parser.add_argument('--rscript', default=os.environ.get('PSEUDOBULK_RSCRIPT', '/storage1/fs1/fuhai.li/Active/di.huang/cache/R/omic-deseq2/bin/Rscript'))
    parser.add_argument('--alpha', type=float, default=.05)
    parser.add_argument('--min-count', type=int, default=10)
    parser.add_argument('--min-gene-donors', type=int, default=3)
    parser.add_argument('--min-group-donors', type=int, default=3)
    args = parser.parse_args(argv)
    sessions_root = (REPO/'webapp/sessions').resolve()
    output = Path(args.output_dir).resolve()
    if output == sessions_root or not output.is_relative_to(sessions_root):
        parser.error('--output-dir must be a new directory under webapp/sessions')
    try:
        result = execute(**vars(args))
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0 if result['status'] == 'success' else 2


if __name__ == '__main__':
    raise SystemExit(main())

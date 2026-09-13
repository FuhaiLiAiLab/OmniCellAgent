"""Check the unified entry point without fetching data or submitting gene lists."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.omic_tools.pseudobulk_pipeline import omic_workflow as entry


def setup_run(tmp_path, monkeypatch, extra=()):
    monkeypatch.setattr(entry, 'REPO', tmp_path)
    reference = tmp_path / 'genes.tsv'
    reference.write_text('symbol\n')
    exclusions = tmp_path / 'exclusions.csv'
    exclusions.write_text('source,dataset_id,donor_id,reason\n')
    args = entry.build_parser().parse_args([
        '--output-dir', str(tmp_path / 'webapp/sessions/new output'), '--data-root', str(tmp_path),
        '--celltosg-root', str(tmp_path), '--hgnc-reference', str(reference),
        '--exclude-donor-keys', str(exclusions), '--rscript', '/bin/true', *extra,
    ])
    calls = []

    def raw(**kwargs):
        calls.append(('raw', kwargs))
        Path(kwargs['output_dir']).mkdir()
        manifest = {'status': 'running'}
        def finish(status):
            manifest['status'] = status
            return manifest
        return SimpleNamespace(manifest=manifest, finish=finish,
                               fail=lambda error: finish('error'))

    def models(**kwargs):
        calls.append(('models', kwargs))
        (Path(kwargs['output_dir']) / 'models' / args.enrichment_model).mkdir(parents=True)
        return {'status': 'success'}

    def enrich(path, **kwargs):
        calls.append(('enrichment', {'path': path, **kwargs}))
        return {'status': 'success'}

    monkeypatch.setattr(entry, 'load_pipeline_stages', lambda: SimpleNamespace(
        create_raw_run=raw,
        select_donor_cohort=lambda run: object(),
        retrieve_and_aggregate=lambda run, cohort: {},
        fit_initial_comparisons=lambda run: run.finish('success'),
    ))
    monkeypatch.setattr(entry, 'run_models', models)
    def selected_model(**kwargs):
        calls.append(('selected_model', kwargs))
        (Path(kwargs['output_dir']) / 'models' / kwargs['model_name']).mkdir(parents=True)
        return {'status': 'success', 'fitted_model_count': 1}
    monkeypatch.setattr(entry, 'run_selected_model', selected_model)
    monkeypatch.setattr(entry, 'run_enrichment', enrich)
    monkeypatch.setattr(entry, 'regenerate_plots', lambda path, **kwargs: calls.append(('plots', kwargs)) or {'status': 'success'})
    return args, calls


def test_arguments_reach_correct_components(tmp_path, monkeypatch):
    args, calls = setup_run(tmp_path, monkeypatch, [
        '--alpha', '.02', '--min-count', '12', '--chunk-size', '17',
        '--enrichment-model', 'shared_C', '--log2fc-min', '.5',
        '--enrichment-timeout', '45', '--enrichment-retries', '4',
        '--top-kegg', '8', '--raw-plots', '--full-sensitivity',
    ])
    assert entry.execute(args)['status'] == 'success'
    assert [name for name, _ in calls] == ['raw', 'models', 'enrichment', 'plots']
    raw, models, enrich, plots = [value for _, value in calls]
    assert raw['skip_enrichment'] is True and raw['skip_plots'] is False
    assert raw['chunk_size'] == 17 and raw['min_count'] == models['min_count'] == 12
    assert raw['alpha'] == models['alpha'] == enrich['alpha'] == plots['alpha'] == .02
    assert Path(models['source_run']) == Path(raw['output_dir'])
    assert Path(enrich['path']).name == 'shared_C'
    assert enrich['log2fc_min'] == .5 and enrich['timeout'] == 45 and enrich['retries'] == 4
    assert plots['top_kegg'] == 8


def test_prepare_only_and_existing_output(tmp_path, monkeypatch):
    args, calls = setup_run(tmp_path, monkeypatch, ['--prepare-only'])
    assert entry.execute(args)['status'] == 'prepared'
    assert [name for name, _ in calls] == ['raw']
    with pytest.raises(FileExistsError):
        entry.execute(args)
    assert len(calls) == 1


def test_failed_stage_stops_and_records_error(tmp_path, monkeypatch):
    args, calls = setup_run(tmp_path, monkeypatch, ['--full-sensitivity'])
    monkeypatch.setattr(entry, 'run_models', lambda **kwargs: {'status': 'partial'})
    with pytest.raises(RuntimeError, match='sensitivity'):
        entry.execute(args)
    assert [name for name, _ in calls] == ['raw']
    manifest = json.loads((Path(args.output_dir) / 'workflow_manifest.json').read_text())
    assert manifest['status'] == 'error'


def test_source_run_skips_retrieval_and_skip_enrichment(tmp_path, monkeypatch):
    saved = tmp_path / 'saved'
    saved.mkdir()
    args, calls = setup_run(tmp_path, monkeypatch, ['--source-run', str(saved), '--skip-enrichment', '--full-sensitivity'])
    assert entry.execute(args)['status'] == 'success'
    assert [name for name, _ in calls] == ['models']
    assert Path(calls[0][1]['source_run']) == saved


@pytest.mark.parametrize('extra', [['--alpha', 'nan'], ['--min-group-donors', '2'], ['--top-kegg', '0'], ['--log2fc-min', '-1']])
def test_invalid_options_fail_before_any_stage(tmp_path, monkeypatch, extra):
    args, calls = setup_run(tmp_path, monkeypatch, extra)
    with pytest.raises(ValueError):
        entry.execute(args)
    assert not calls and not Path(args.output_dir).exists()


def test_requested_command_and_session_output(tmp_path, monkeypatch):
    parsed = entry.build_parser().parse_args([
        '--disease', "Alzheimer's Disease", '--organ', 'brain', '--cell-type', 'astrocyte',
        '--enrichement-model', 'disease-only', '--session-id', 'alzheimer_test',
    ])
    assert parsed.enrichment_model == 'full_A'
    args, calls = setup_run(tmp_path, monkeypatch)
    args.output_dir = None
    args.session_id = parsed.session_id
    monkeypatch.setattr(entry, 'REPO', tmp_path)
    result = entry.execute(args)
    session = tmp_path/'webapp/sessions/alzheimer_test'
    assert Path(result['output_dir']) == session
    assert Path(calls[2][1]['path']) == session/'models/models/full_A'


@pytest.mark.parametrize('name,expected', [
    ('disease-only', 'full_A'), ('study-disease', 'full_B'),
    ('fully-adjusted', 'full_C'), ('shared-studies-adjusted', 'shared_C'),
    ('full_B', 'full_B'),
])
def test_readable_model_names(name, expected):
    args = entry.build_parser().parse_args(['--enrichment-model', name])
    assert args.enrichment_model == expected


def test_session_rejects_traversal_and_conflicting_output(tmp_path, monkeypatch):
    args, calls = setup_run(tmp_path, monkeypatch)
    args.output_dir = None
    args.session_id = '../outside'
    with pytest.raises(ValueError, match='session-id'):
        entry.execute(args)
    assert not calls
    with pytest.raises(SystemExit):
        entry.build_parser().parse_args(['--session-id', 'test', '--output-dir', str(tmp_path/'test')])


def test_scientific_stages_run_in_order(tmp_path, monkeypatch):
    args, calls = setup_run(tmp_path, monkeypatch, ['--full-sensitivity'])
    stages = entry.load_pipeline_stages()
    order = []
    stages.select_donor_cohort = lambda run: order.append('select') or 'cohort'
    def aggregate(run, cohort):
        assert cohort == 'cohort'
        order.append('aggregate')
    stages.retrieve_and_aggregate = aggregate
    stages.fit_initial_comparisons = lambda run: order.append('DESeq2') or run.finish('success')
    monkeypatch.setattr(entry, 'load_pipeline_stages', lambda: stages)
    assert entry.execute(args)['status'] == 'success'
    assert order == ['select', 'aggregate', 'DESeq2']


def test_persistent_default_caches(tmp_path, monkeypatch):
    from tools.omic_tools.pseudobulk_pipeline.run_support import configure_environment
    import os
    for variable in ('MPLCONFIGDIR', 'NUMBA_CACHE_DIR', 'TMPDIR'):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv('PYTHONPATH', '.')
    monkeypatch.setenv('PYTHONDONTWRITEBYTECODE', '1')
    configure_environment(tmp_path)
    for variable, folder in [('MPLCONFIGDIR', 'matplotlib'), ('NUMBA_CACHE_DIR', 'numba'), ('TMPDIR', 'working')]:
        expected = tmp_path/'dataset_outputs/pseudobulk_runtime_cache'/folder
        assert Path(os.environ[variable]) == expected and expected.is_dir()


def test_default_output_is_a_webapp_session(tmp_path, monkeypatch):
    args, calls = setup_run(tmp_path, monkeypatch, ['--skip-enrichment'])
    args.output_dir = None
    result = entry.execute(args)
    assert Path(result['output_dir']).parent == tmp_path/'webapp/sessions'


def test_dataset_output_destination_is_rejected(tmp_path, monkeypatch):
    args, calls = setup_run(tmp_path, monkeypatch)
    args.output_dir = str(tmp_path/'dataset_outputs/unwanted_run')
    with pytest.raises(ValueError, match='under webapp/sessions'):
        entry.execute(args)
    assert not calls and not Path(args.output_dir).exists()


@pytest.mark.parametrize('model', ['disease-only', 'study-disease', 'fully-adjusted', 'shared-studies-adjusted'])
def test_only_selected_model_is_fitted_by_default(tmp_path, monkeypatch, model):
    args, calls = setup_run(tmp_path, monkeypatch, ['--enrichment-model', model, '--raw-plots'])
    stages = entry.load_pipeline_stages()
    def unexpected_fit(*args, **kwargs):
        pytest.fail('Selected-only mode must not fit initial or full sensitivity models')
    stages.fit_initial_comparisons = unexpected_fit
    monkeypatch.setattr(entry, 'load_pipeline_stages', lambda: stages)
    monkeypatch.setattr(entry, 'run_models', unexpected_fit)
    result = entry.execute(args)
    assert result['status'] == 'success'
    assert [name for name, _ in calls] == ['raw', 'selected_model', 'enrichment', 'plots']
    assert calls[1][1]['model_name'] == entry.MODEL_NAMES[model]
    assert calls[1][1]['export_expression'] is True
    assert result['stages']['raw_pipeline']['status'] == 'prepared'

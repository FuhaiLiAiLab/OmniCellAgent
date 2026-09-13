"""The Python wrapper must neither require pilot results nor accept extra fits."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.omic_tools.pseudobulk_pipeline import selected_model


@pytest.mark.parametrize('fit_count', [1, 2])
def test_raw_only_inputs_and_exactly_one_fit(tmp_path, monkeypatch, fit_count):
    source = tmp_path/'raw'
    (source/'inputs').mkdir(parents=True)
    for name in ('pseudobulk_counts.csv', 'donor_metadata.csv'):
        (source/'inputs'/name).write_text('fixture input\n')
    output = tmp_path/'selected'

    def run(command, **kwargs):
        assert command[-6:] == ['full_A', '0.05', '10', '3', '3', 'false']
        comparison = output/'models/full_A'
        comparison.mkdir(parents=True)
        (output/'manifest.json').write_text(json.dumps({
            'status': 'success', 'model_order': ['full_A'], 'fitted_model_count': fit_count,
        }))
        (comparison/'status.json').write_text(json.dumps({
            'status': 'success', 'formula': '~disease', 'group_counts': {'AD': 3, 'control': 3},
        }))
        (comparison/'results.csv').write_text('gene,tested,padj,log2FoldChange\nA,TRUE,0.01,1\nB,TRUE,0.5,-1\n')
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(selected_model.subprocess, 'run', run)
    if fit_count == 1:
        result = selected_model.run_selected_model(source, output, '/bin/true', 'full_A')
        assert result['fitted_model_count'] == result['significant_genes'] == 1
        assert not (source/'comparisons').exists()
    else:
        with pytest.raises(RuntimeError, match='exactly'):
            selected_model.run_selected_model(source, output, '/bin/true', 'full_A')
    assert json.loads((output/'source_preservation.json').read_text())['source_unchanged']

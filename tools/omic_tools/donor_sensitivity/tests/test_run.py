import importlib

import pytest


def test_run_refuses_to_reuse_a_folder(tmp_path):
    module = importlib.import_module('tools.omic_tools.donor_sensitivity.run')
    output = tmp_path / 'existing'
    output.mkdir()
    with pytest.raises(FileExistsError):
        module.execute(tmp_path/'source', output, 'Rscript')


def test_output_cannot_be_inside_original_run(tmp_path):
    module = importlib.import_module('tools.omic_tools.donor_sensitivity.run')
    source = tmp_path / 'source'
    source.mkdir()
    with pytest.raises(ValueError, match='source'):
        module.execute(source, source/'new_analysis', 'Rscript')

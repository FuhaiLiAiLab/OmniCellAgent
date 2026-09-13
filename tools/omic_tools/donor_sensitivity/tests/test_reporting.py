import importlib
import json
from pathlib import Path

import pandas as pd
import pytest


def fixture(tmp_path):
    source = tmp_path / "source"
    reference = source / "comparisons/ad_vs_control"
    reference.mkdir(parents=True)
    metadata = pd.DataFrame({"sample_id": ["A", "B", "C", "D"],
                             "disease": ["AD", "control", "AD", "control"],
                             "study": ["S1", "S1", "S2", "S2"]})
    metadata.to_csv(reference / "donor_metadata.csv", index=False)
    pd.DataFrame({"gene": ["RPLP0", "G1"], "A": [2, 8], "B": [4, 2],
                  "C": [4, 12], "D": [4, 3]}).to_csv(reference / "normalized_counts.csv", index=False)
    base = pd.DataFrame({"gene": ["RPLP0", "G1"], "baseMean": [4, 8],
                         "log2FoldChange": [-1.0, 2.0], "lfcSE": [.2, .3],
                         "pvalue": [.001, .0001], "padj": [.01, .001]})
    base.to_csv(reference / "results.csv", index=False)
    output = tmp_path / "output"
    statuses = {}
    for name, shift in [("full_A", 0), ("full_B", .5), ("full_C", .75), ("shared_C", .75)]:
        directory = output / "models" / name
        directory.mkdir(parents=True)
        results = base.copy()
        results["log2FoldChange"] += shift
        results["lfc_ci_low"] = results.log2FoldChange - 1.96 * results.lfcSE
        results["lfc_ci_high"] = results.log2FoldChange + 1.96 * results.lfcSE
        results["tested"] = True
        results.to_csv(directory / "results.csv", index=False)
        status = {"status": "success", "group_counts": {"AD": 2, "control": 2},
                  "formula": "~study + disease", "included_donors": metadata.sample_id.tolist(),
                  "filter": {"n_retained_genes": 2}, "omitted_study": None}
        (directory / "status.json").write_text(json.dumps(status))
        statuses[name] = status
    (output / "manifest.json").write_text(json.dumps({"model_order": list(statuses),
        "models": statuses, "shared_studies": ["S1", "S2"], "settings": {"alpha": .05}}))
    return source, output


def test_reports_saved_effects_and_within_study_donor_means(tmp_path):
    source, output = fixture(tmp_path)
    before = {p: p.read_bytes() for p in source.rglob('*') if p.is_file()}
    module = importlib.import_module('tools.omic_tools.donor_sensitivity.reporting')
    result = module.build_report(source, output)
    summary = pd.read_csv(output / 'summary/model_summary.csv')
    assert summary.model.tolist() == ['full_A', 'full_B', 'full_C', 'shared_C']
    effects = pd.read_csv(output / 'summary/gene_effects.csv')
    assert effects.loc[(effects.model == 'full_C') & (effects.gene == 'RPLP0'), 'log2FoldChange'].iloc[0] == -.25
    within = pd.read_csv(output / 'summary/within_study_effects.csv')
    assert within.loc[(within.study == 'S1') & (within.gene == 'RPLP0'), 'descriptive_log2fc'].iloc[0] == -1
    assert result['de_recomputed_by_reporting'] is False
    assert before == {p: p.read_bytes() for p in source.rglob('*') if p.is_file()}
    assert (output / 'summary/fold_change_concordance.png').exists()
    assert (output / 'summary/focused_effect_forest.png').exists()


def test_reporting_refuses_existing_summary(tmp_path):
    source, output = fixture(tmp_path)
    (output/'summary').mkdir()
    module = importlib.import_module('tools.omic_tools.donor_sensitivity.reporting')
    with pytest.raises(FileExistsError):
        module.build_report(source, output)

import importlib

import numpy as np
import pandas as pd
import pytest


def aggregate(counts):
    module = importlib.import_module("tools.omic_tools.pseudobulk_pipeline.aggregation")
    donors = pd.DataFrame({"sample_id": ["PB2", "PB1"]})
    assignments = pd.DataFrame({"sample_id": ["PB1", "PB2", "PB1"]})
    return module.aggregate_raw_counts(counts, ["G2", "G1"], assignments, donors)


def test_raw_counts_are_summed_and_gene_totals_conserved_in_donor_order():
    counts = np.array([[1, 3], [7, 9], [2, 5]], dtype=np.float32)
    result, report = aggregate(counts)
    assert result.index.tolist() == ["G2", "G1"]
    assert result.columns.tolist() == ["PB2", "PB1"]
    np.testing.assert_array_equal(result.to_numpy(), [[7, 3], [9, 8]])
    assert report["input_total_counts"] == report["pseudobulk_total_counts"] == 27
    assert report["per_gene_count_conservation"] is True


@pytest.mark.parametrize("invalid", [0.25, -1, np.nan, np.inf])
def test_invalid_counts_are_not_rounded_or_normalized(invalid):
    values = np.ones((3, 2))
    values[1, 0] = invalid
    with pytest.raises(ValueError, match="counts"):
        aggregate(values)


def test_donor_alignment_mismatch_fails():
    module = importlib.import_module("tools.omic_tools.pseudobulk_pipeline.aggregation")
    with pytest.raises(ValueError, match="donor"):
        module.aggregate_raw_counts(np.ones((2, 1)), ["G1"],
                                    pd.DataFrame({"sample_id": ["A", "B"]}),
                                    pd.DataFrame({"sample_id": ["A", "C"]}))

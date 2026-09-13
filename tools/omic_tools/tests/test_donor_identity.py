"""Diagnostics and sampling must count the same study-scoped donor keys."""

import pandas as pd

from tools.omic_tools.cohort_diagnostics import compute_cohort_diagnostics


def test_diagnostics_distinguish_reused_donor_labels_and_normalize_whitespace():
    metadata = pd.DataFrame({
        "source": ["A", "A", "A", "B", "A", "A", "A"],
        "dataset_id": ["one", "one", "two", "one", "one", "one", "one"],
        "donor_id": [" 1 ", "1", "1", "1", "unknown", "2", "2"],
    })
    is_ref = [True, True, True, True, True, False, False]
    is_alt = [not value for value in is_ref]
    result = compute_cohort_diagnostics(metadata, is_ref, is_alt)
    check = result["checks"]["donor_count"]
    assert check["counts"] == {"reference": 3, "alternate": 1}
    assert check["donor_key_columns"] == ["source", "dataset_id", "donor_id"]


def test_diagnostics_dominance_does_not_merge_same_label_across_studies():
    metadata = pd.DataFrame({
        "source": ["A"] * 6,
        "dataset_id": ["one", "two", "three"] * 2,
        "donor_id": ["1"] * 6,
    })
    result = compute_cohort_diagnostics(metadata, [True] * 3 + [False] * 3,
                                       [False] * 3 + [True] * 3)
    assert result["checks"]["donor_count"]["value"] == 3
    assert result["checks"]["donor_dominance"]["value"] == 1 / 3

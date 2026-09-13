"""Sample only known-donor metacells, with independent per-group caps."""

import importlib

import pandas as pd
import pytest


ALIASES = {"disease": "disease_BMG_name", "cell_type": "CMT_name", "sex": "sex_normalized"}
CONDITIONS = {"disease": "AD", "cell_type": "astrocyte", "tissue_general": "brain"}


def _pool():
    rows = []
    for disease, donors in [
        ("AD", ["A", "B", "unknown", " UNKNOWN ", None, "", "NaN", "none", "na"]),
        ("normal", ["C", "D", "E", "F", "G", "unknown"]),
    ]:
        for donor in donors:
            rows.append({"disease_BMG_name": disease, "donor_id": donor,
                         "CMT_name": "astrocyte", "tissue_general": "brain",
                         "matrix_file_path": "matrix.npy", "matrix_row_idx": len(rows),
                         "sex_normalized": "female", "suspension_type": "nucleus"})
    rows.append(dict(rows[0], matrix_row_idx=100, tissue_general="lung"))
    rows.append(dict(rows[0], matrix_row_idx=101, disease_BMG_name="other disease"))
    rows.append(dict(rows[0]))  # repeated source pointer must not be resampled
    return pd.DataFrame(rows)


def _sample(*args, **kwargs):
    module = importlib.import_module("tools.omic_tools.donor_sampling")
    return module.sample_known_donor_cohort(*args, **kwargs)


def test_known_donor_pools_are_sampled_independently_and_report_counts(capsys):
    original = _pool()
    selected, summary = _sample(original, CONDITIONS, ALIASES, sample_size=3)
    assert len(selected[selected.disease_BMG_name.eq("AD")]) == 2
    assert len(selected[selected.disease_BMG_name.eq("normal")]) == 3
    assert set(selected[selected.disease_BMG_name.eq("AD")].donor_id) == {"A", "B"}
    assert selected.donor_id.notna().all()
    assert not selected.duplicated(["matrix_file_path", "matrix_row_idx"]).any()
    assert summary["disease"]["known_donor_available"] == 2
    assert summary["disease"]["excluded_unknown_donor"] == 7
    assert summary["disease"]["duplicate_rows_removed"] == 1
    assert summary["control"]["known_donor_available"] == 5
    output = capsys.readouterr().out
    assert "disease" in output and "control" in output
    assert "known-donor available=2" in output and "selected=2" in output
    assert "using all available" in output
    pd.testing.assert_frame_equal(original, _pool())


def test_uses_all_of_both_groups_when_both_are_below_limit():
    selected, _ = _sample(_pool(), CONDITIONS, ALIASES, sample_size=1000)
    assert selected.disease_BMG_name.value_counts().to_dict() == {"normal": 5, "AD": 2}


def test_sampling_is_reproducible_without_replacement_and_respects_constraints():
    pool = _pool()
    pool.loc[pool.donor_id.eq("G"), "suspension_type"] = "cell"
    conditions = dict(CONDITIONS, sex="FEMALE", suspension_type="nucleus")
    first, _ = _sample(pool, conditions, ALIASES, sample_size=2, random_state=42)
    second, _ = _sample(pool, conditions, ALIASES, sample_size=2, random_state=42)
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 4
    assert not first.donor_id.eq("G").any()
    assert first.tissue_general.eq("brain").all()


@pytest.mark.parametrize("disease", ["AD", "normal"])
def test_empty_known_donor_group_fails_instead_of_backfilling_unknowns(disease, capsys):
    pool = _pool()
    pool.loc[pool.disease_BMG_name.eq(disease), "donor_id"] = "unknown"
    with pytest.raises(ValueError, match="No known-donor metacells"):
        _sample(pool, CONDITIONS, ALIASES, sample_size=3)
    assert "known-donor available=0" in capsys.readouterr().out


def test_no_explicit_disease_preserves_other_disease_labels():
    selected, _ = _sample(_pool(), {"tissue_general": "brain"}, ALIASES, sample_size=1000)
    assert set(selected.disease_BMG_name) == {"AD", "other disease", "normal"}


def test_missing_donor_column_is_an_error():
    with pytest.raises(ValueError, match="donor_id"):
        _sample(_pool().drop(columns="donor_id"), CONDITIONS, ALIASES)


@pytest.mark.parametrize("sample_size", [0, -1])
def test_sample_limit_must_be_positive(sample_size):
    with pytest.raises(ValueError, match="positive"):
        _sample(_pool(), CONDITIONS, ALIASES, sample_size=sample_size)


def _donor_pool(donors):
    template = _pool().iloc[0].to_dict()
    return pd.DataFrame([
        dict(template, source="source", dataset_id="study", donor_id=donor,
             disease_BMG_name=group, matrix_row_idx=i * 2 + j)
        for i, donor in enumerate(donors)
        for j, group in enumerate(["AD", "normal"])
    ])


def test_covers_every_donor_before_taking_more_from_abundant_donors(capsys):
    pool = _donor_pool(["abundant"] * 1000 + [f"rare_{i}" for i in range(10)])
    selected, summary = _sample(pool, CONDITIONS, ALIASES, sample_size=12)
    for group in ["AD", "normal"]:
        part = selected[selected.disease_BMG_name.eq(group)]
        assert len(part) == 12
        assert part.donor_id.nunique() == 11
        assert part.donor_id.value_counts()["abundant"] == 2
    assert summary["disease"]["donors_available"] == 11
    assert summary["disease"]["donors_selected"] == 11
    assert "donors selected=11/11" in capsys.readouterr().out


def test_uses_as_many_donors_as_slots_when_donors_exceed_sample_cap():
    pool = _donor_pool(["abundant"] * 1000 + [f"rare_{i}" for i in range(10)])
    selected, summary = _sample(pool, CONDITIONS, ALIASES, sample_size=5)
    for _, part in selected.groupby("disease_BMG_name"):
        assert part.donor_id.nunique() == 5
    assert summary["control"]["donors_selected"] == 5


def test_remaining_slots_are_distributed_evenly_when_donors_have_enough_rows():
    pool = _donor_pool(["A"] * 40 + ["B"] * 40 + ["C"] * 40)
    selected, _ = _sample(pool, CONDITIONS, ALIASES, sample_size=9)
    for _, part in selected.groupby("disease_BMG_name"):
        assert part.donor_id.value_counts().to_dict() == {"A": 3, "B": 3, "C": 3}


def test_same_donor_label_in_different_studies_gets_distinct_slots():
    pool = _donor_pool(["1"] * 100 + ["1", "2", "2"])
    pool.loc[pool.matrix_row_idx.ge(200), "dataset_id"] = "other_study"
    pool.loc[pool.matrix_row_idx.ge(204), "source"] = "other_source"
    selected, summary = _sample(pool, CONDITIONS, ALIASES, sample_size=4)
    for _, part in selected.groupby("disease_BMG_name"):
        assert len(part.drop_duplicates(["source", "dataset_id", "donor_id"])) == 4
    assert summary["donor_key_columns"] == ["source", "dataset_id", "donor_id"]
    assert summary["disease"]["donors_selected"] == 4


def test_donor_diversity_selection_is_reproducible_and_has_no_duplicate_rows():
    pool = _donor_pool(["A"] * 100 + [f"D{i}" for i in range(30)])
    first, _ = _sample(pool, CONDITIONS, ALIASES, sample_size=12, random_state=42)
    second, _ = _sample(pool, CONDITIONS, ALIASES, sample_size=12, random_state=42)
    pd.testing.assert_frame_equal(first, second)
    assert not first.duplicated(["matrix_file_path", "matrix_row_idx"]).any()

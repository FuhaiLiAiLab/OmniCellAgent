"""Sample only known-donor metacells, with independent per-group caps."""

import importlib
import inspect
import re

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
    # Most fixtures are smaller than the run's minimum, which has its own tests below
    kwargs.setdefault("min_metacells", 1)
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


def test_keep_unknown_donor_includes_unlabelled_metacells_without_counting_them_as_donors(capsys):
    selected, summary = _sample(_pool(), CONDITIONS, ALIASES, sample_size=1000,
                                keep_unknown_donor=True)
    assert selected.disease_BMG_name.value_counts().to_dict() == {"AD": 9, "normal": 6}
    assert not selected.duplicated(["matrix_file_path", "matrix_row_idx"]).any()
    assert summary["keep_unknown_donor"] is True
    assert summary["disease"]["included_unknown_donor"] == 7
    assert summary["disease"]["excluded_unknown_donor"] == 0
    assert summary["disease"]["known_donor_available"] == 2
    assert summary["disease"]["donors_available"] == 2
    assert summary["control"]["included_unknown_donor"] == 1
    assert "kept unknown/unusable donor=7" in capsys.readouterr().out


def test_default_excludes_unlabelled_metacells_and_reports_none_included():
    _, summary = _sample(_pool(), CONDITIONS, ALIASES, sample_size=1000)
    assert summary["keep_unknown_donor"] is False
    assert summary["disease"]["included_unknown_donor"] == 0


def test_keep_unknown_donor_rescues_a_group_with_no_known_donor():
    pool = _pool()
    pool.loc[pool.disease_BMG_name.eq("AD"), "donor_id"] = "unknown"
    selected, summary = _sample(pool, CONDITIONS, ALIASES, sample_size=1000,
                                keep_unknown_donor=True)
    assert len(selected[selected.disease_BMG_name.eq("AD")]) == 9
    assert summary["disease"]["donors_available"] == 0


def test_unlabelled_metacells_share_one_sampling_slot_per_study_when_capped():
    pool = _donor_pool(["unknown"] * 20 + ["NaN"] * 20 + [""] * 10 + ["A", "B"])
    selected, summary = _sample(pool, CONDITIONS, ALIASES, sample_size=3,
                                keep_unknown_donor=True)
    for _, part in selected.groupby("disease_BMG_name"):
        assert len(part) == 3
        assert {"A", "B"} <= set(part.donor_id)
    assert summary["disease"]["included_unknown_donor"] == 1
    assert summary["disease"]["donors_selected"] == 2


def _tumour_pool():
    """Tumour cells and their control are different cell types, as in LUAD and PDAC."""
    rows = []
    for disease, cmt, label, donors in [
        ("AD", "epithelial cell", "Malignant epithelial cell", ["A", "B", "C"]),
        ("AD", "epithelial cell", "Ductal epithelial cell", ["D"]),
        ("AD", "ductal cell", "Ductal", ["E"]),
        ("normal", "ductal cell", "Ductal", ["F", "G"]),
        ("normal", "epithelial cell", "Ductal epithelial cell", ["H"]),
    ]:
        for donor in donors:
            rows.append({"disease_BMG_name": disease, "donor_id": donor, "CMT_name": cmt,
                         "cell_type": label, "tissue_general": "brain",
                         "matrix_file_path": "matrix.npy", "matrix_row_idx": len(rows)})
    return pd.DataFrame(rows)


def test_control_cell_type_selects_the_control_group_by_its_own_name():
    conditions = {"disease": "AD", "cell_type": "epithelial cell", "tissue_general": "brain"}
    selected, summary = _sample(_tumour_pool(), conditions, ALIASES, sample_size=1000,
                                control_cell_type="ductal cell")
    disease = selected[selected.disease_BMG_name.eq("AD")]
    control = selected[selected.disease_BMG_name.eq("normal")]
    assert set(disease.CMT_name) == {"epithelial cell"} and len(disease) == 4
    assert set(control.CMT_name) == {"ductal cell"} and set(control.donor_id) == {"F", "G"}
    assert summary["control_cell_type"] == "ductal cell"


def test_disease_cell_label_also_filters_disease_cells_by_their_original_label():
    conditions = {"disease": "AD", "cell_type": "epithelial cell", "tissue_general": "brain"}
    selected, summary = _sample(_tumour_pool(), conditions, ALIASES, sample_size=1000,
                                control_cell_type="ductal cell",
                                disease_cell_label="malignant EPITHELIAL cell")
    disease = selected[selected.disease_BMG_name.eq("AD")]
    assert set(disease.cell_type) == {"Malignant epithelial cell"} and len(disease) == 3
    assert set(selected[selected.disease_BMG_name.eq("normal")].donor_id) == {"F", "G"}
    assert summary["disease_cell_label"] == "malignant EPITHELIAL cell"


def test_without_the_new_options_both_groups_share_one_cell_type():
    conditions = {"disease": "AD", "cell_type": "epithelial cell", "tissue_general": "brain"}
    selected, summary = _sample(_tumour_pool(), conditions, ALIASES, sample_size=1000)
    assert set(selected.CMT_name) == {"epithelial cell"}
    assert selected.disease_BMG_name.value_counts().to_dict() == {"AD": 4, "normal": 1}
    assert summary["control_cell_type"] is None and summary["disease_cell_label"] is None


def _study_pool(counts):
    """{(dataset_id, disease): metacells}, each metacell from its own donor."""
    rows = [{"disease_BMG_name": disease, "dataset_id": dataset, "donor_id": f"{dataset}-{disease}-{i}",
             "CMT_name": "astrocyte", "tissue_general": "brain", "matrix_file_path": "matrix.npy"}
            for (dataset, disease), n in counts.items() for i in range(n)]
    return pd.DataFrame(rows).assign(matrix_row_idx=range(len(rows)))


def test_default_minimum_is_ten_metacells_per_group():
    module = importlib.import_module("tools.omic_tools.donor_sampling")
    assert module.MIN_METACELLS_PER_GROUP == 10
    parameters = inspect.signature(module.sample_known_donor_cohort).parameters
    assert parameters["min_metacells"].default == module.MIN_METACELLS_PER_GROUP


@pytest.mark.parametrize("n_disease, n_normal", [(9, 10), (10, 9)])
def test_a_group_below_the_minimum_fails(n_disease, n_normal, capsys):
    pool = _study_pool({("s1", "AD"): n_disease, ("s1", "normal"): n_normal})
    short = "disease (9)" if n_disease == 9 else "control (9)"
    with pytest.raises(ValueError, match=rf"Fewer than 10 known-donor metacells available for: {re.escape(short)}"):
        _sample(pool, CONDITIONS, ALIASES, min_metacells=10)


def test_exactly_the_minimum_runs_and_is_recorded():
    pool = _study_pool({("s1", "AD"): 10, ("s1", "normal"): 10})
    selected, summary = _sample(pool, CONDITIONS, ALIASES, min_metacells=10)
    assert selected.disease_BMG_name.value_counts().to_dict() == {"AD": 10, "normal": 10}
    assert summary["min_metacells_per_group"] == 10


def test_minimum_counts_only_metacells_in_datasets_holding_both_groups():
    # 25 disease metacells in all, but only 6 in s1, the one dataset that also holds normal ones
    pool = _study_pool({("s1", "AD"): 6, ("s1", "normal"): 12, ("s2", "AD"): 19})
    with pytest.raises(ValueError, match=r"Fewer than 10 known-donor metacells available for: disease \(6\)"):
        _sample(pool, CONDITIONS, ALIASES, min_metacells=10, shared_datasets_only=True)
    selected, summary = _sample(pool, CONDITIONS, ALIASES, min_metacells=10)
    assert selected.disease_BMG_name.value_counts().to_dict() == {"AD": 25, "normal": 12}
    pool = _study_pool({("s1", "AD"): 6, ("s1", "normal"): 12, ("s2", "AD"): 19, ("s2", "normal"): 1})
    selected, summary = _sample(pool, CONDITIONS, ALIASES, min_metacells=10, shared_datasets_only=True)
    assert summary["shared_datasets"] == ["s1", "s2"]
    assert selected.disease_BMG_name.value_counts().to_dict() == {"AD": 25, "normal": 13}


def test_minimum_counts_unlabelled_metacells_only_when_they_are_kept():
    pool = _study_pool({("s1", "AD"): 10, ("s1", "normal"): 10})
    pool.loc[pool.index[:2], "donor_id"] = "unknown"
    with pytest.raises(ValueError, match=r"disease \(8\)"):
        _sample(pool, CONDITIONS, ALIASES, min_metacells=10)
    selected, _ = _sample(pool, CONDITIONS, ALIASES, min_metacells=10, keep_unknown_donor=True)
    assert len(selected) == 20


@pytest.mark.parametrize("min_metacells", [0, -1])
def test_minimum_must_be_positive(min_metacells):
    with pytest.raises(ValueError, match="min_metacells must be a positive"):
        _sample(_pool(), CONDITIONS, ALIASES, min_metacells=min_metacells)


def test_donor_diversity_selection_is_reproducible_and_has_no_duplicate_rows():
    pool = _donor_pool(["A"] * 100 + [f"D{i}" for i in range(30)])
    first, _ = _sample(pool, CONDITIONS, ALIASES, sample_size=12, random_state=42)
    second, _ = _sample(pool, CONDITIONS, ALIASES, sample_size=12, random_state=42)
    pd.testing.assert_frame_equal(first, second)
    assert not first.duplicated(["matrix_file_path", "matrix_row_idx"]).any()

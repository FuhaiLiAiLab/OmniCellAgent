import importlib

import pandas as pd


def module():
    return importlib.import_module("tools.omic_tools.pseudobulk_pipeline.metadata")


def fixture():
    return pd.DataFrame({
        "source": ["atlas"] * 6,
        "dataset_id": ["one", "one", "two", "two", "one", "one"],
        "donor_id": ["1", "1", "1", "2", "unknown", "3"],
        "disease_BMG_name": ["Alzheimer's Disease"] * 3 + ["normal"] * 3,
        "CMT_name": ["astrocyte"] * 6,
        "tissue_general": ["brain"] * 5 + ["lung"],
        "sex_normalized": ["female", "male", "female", "male", "unknown", "male"],
        "development_stage_numeric_age": [70, 71, 70, 65, None, 65],
        "matrix_file_path": ["matrix.npy"] * 6,
        "matrix_row_idx": list(range(6)),
    })


def test_exact_query_preserves_all_known_donor_metacells():
    original = fixture()
    selected, audit = module().select_cohort(
        original, "alzheimer's disease", "ASTROCYTE", "brain"
    )
    assert selected.matrix_row_idx.tolist() == [0, 1, 2, 3]
    assert audit["disease"]["selected_metacells"] == 3
    assert audit["control"]["selected_metacells"] == 1
    assert audit["control"]["unknown_donor_excluded"] == 1
    pd.testing.assert_frame_equal(original, fixture())


def test_exact_miss_is_not_replaced_by_fuzzy_match():
    import pytest
    with pytest.raises(ValueError, match="disease"):
        module().select_cohort(fixture(), "Alzheimer disease", "astrocyte", "brain")


def test_conflicts_are_unresolved_and_repeated_labels_are_independent_by_study():
    selected, _ = module().select_cohort(fixture(), "Alzheimer's Disease", "astrocyte", "brain")
    donors, assignment, issues = module().build_donor_metadata(selected)
    assert len(donors) == 3
    assert len(assignment) == len(selected)
    assert assignment.sample_id.iloc[0] == assignment.sample_id.iloc[1]
    assert assignment.sample_id.iloc[0] != assignment.sample_id.iloc[2]
    conflict = donors[(donors.dataset_id == "one") & (donors.donor_id == "1")].iloc[0]
    assert pd.isna(conflict.sex) and pd.isna(conflict.age)
    assert conflict.sex_conflict and conflict.age_conflict
    assert set(issues.loc[issues.sample_id.eq(conflict.sample_id), "field"]) == {"sex", "age"}
    assert set(donors.disease) == {"AD", "control"}


def test_missing_covariates_are_recorded_not_invented():
    selected = fixture().iloc[[0, 2, 3]].drop(columns=["sex_normalized", "development_stage_numeric_age"])
    donors, _, issues = module().build_donor_metadata(selected)
    assert donors.sex.isna().all() and donors.age.isna().all()
    assert set(issues.resolution) == {"missing"}


def test_conflicting_disease_is_not_assigned_by_majority():
    data = fixture().iloc[[0, 1, 3]].copy()
    data.loc[data.index[-1], ["dataset_id", "donor_id"]] = ["one", "1"]
    donors, _, issues = module().build_donor_metadata(data)
    assert len(donors) == 1 and donors.disease_conflict.iloc[0]
    assert pd.isna(donors.disease.iloc[0])
    assert "disease" in set(issues.field)


def test_conflicting_annotations_on_duplicate_pointer_are_not_silently_dropped():
    import pytest
    data = fixture().iloc[[0, 3]].copy()
    duplicate = data.iloc[[0]].copy()
    duplicate["sex_normalized"] = "male"
    data = pd.concat([data, duplicate], ignore_index=True)
    with pytest.raises(ValueError, match="conflicting annotations"):
        module().select_cohort(data, "Alzheimer's Disease", "astrocyte", "brain")


def test_explicit_exclusion_removes_the_whole_donor_key_only():
    selected, _ = module().select_cohort(fixture(), "Alzheimer's Disease", "astrocyte", "brain")
    exclusions = pd.DataFrame([{"source": "atlas", "dataset_id": "one", "donor_id": "1",
                                "reason": "Source count values are not raw integer counts"}])
    kept, removed, report = module().apply_donor_exclusions(selected, exclusions)
    assert kept.matrix_row_idx.tolist() == [2, 3]
    assert removed.matrix_row_idx.tolist() == [0, 1]
    assert report["excluded_donor_keys"] == 1 and report["excluded_metacells"] == 2
    assert kept.loc[kept.dataset_id.eq("two"), "donor_id"].tolist() == ["1", "2"]


def test_explicit_exclusion_requires_reason_and_a_matching_key():
    import pytest
    selected, _ = module().select_cohort(fixture(), "Alzheimer's Disease", "astrocyte", "brain")
    bad = pd.DataFrame([{"source": "atlas", "dataset_id": "missing", "donor_id": "1", "reason": "invalid counts"}])
    with pytest.raises(ValueError, match="not present"):
        module().apply_donor_exclusions(selected, bad)
    bad["dataset_id"] = "one"
    bad["reason"] = ""
    with pytest.raises(ValueError, match="reason"):
        module().apply_donor_exclusions(selected, bad)

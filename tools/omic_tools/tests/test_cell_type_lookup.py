"""List the cell types with both disease and normal metacells, as the sampler selects them."""

import csv
import inspect
import os

import pandas as pd
import pytest

from tools.omic_tools.cell_type_lookup import (LINEAGE_FILE, MIN_METACELLS, OTHER_HEADING,
                                               available_cell_types, disease_candidates,
                                               format_cell_types, load_lineage)
from tools.omic_tools.donor_sampling import DEFAULT_SAMPLE_SIZE, sample_known_donor_cohort

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
ALIASES = {"disease": "disease_BMG_name", "cell_type": "CMT_name"}
METADATA = "/storage3/fs1/fuhai.li/Active/Shared/dataset/OmniCellTOSG_dataset/cell_metadata_with_mappings.parquet"


def _metadata():
    rows = []
    for cell_type, organ, disease, donors in [
        ("astrocyte", "brain", "AD", ["A", "B", "unknown"]),
        ("astrocyte", "brain", "normal", ["C", "D", "E", None]),
        ("astrocyte", "lung", "AD", ["F"]),
        ("microglial cell", "brain", "AD", ["A", "B"]),  # no normal metacells
        ("made-up cell", "brain", "AD", ["G"]),           # not in the lineage file
        ("made-up cell", "brain", "normal", ["H"]),
        ("oligodendrocyte", "brain", "unknown", ["I"]),   # control-like, not a disease
        ("oligodendrocyte", "brain", "normal", ["J"]),
        ("", "brain", "AD", ["K"]),                       # no cell type
        ("", "brain", "normal", ["L"]),
    ]:
        for donor in donors:
            rows.append({"CMT_name": cell_type, "tissue_general": organ, "disease_BMG_name": disease,
                         "donor_id": donor, "matrix_file_path": "matrix.npy",
                         "matrix_row_idx": len(rows)})
    rows.append(dict(rows[0]))  # a repeated matrix row counts once
    return pd.DataFrame(rows)


def _counts(result):
    return {row.cell_type: (row.disease_metacells, row.normal_metacells)
            for row in result["cell_types"].itertuples(index=False)}


def test_counts_usable_donor_metacells_with_both_groups():
    result = available_cell_types(_metadata(), "AD", "brain", min_metacells=1)
    assert result["disease_found"] and result["disease_name"] == "AD"
    # microglia (no normal), oligodendrocyte (no disease) and the blank name cannot be compared
    assert _counts(result) == {"astrocyte": (2, 3), "made-up cell": (1, 1)}
    assert result["organs_with_cell_types"] == ["brain"]  # lung has no normal astrocytes


def test_lists_only_cell_types_with_the_minimum_in_both_groups():
    assert MIN_METACELLS == 10
    assert inspect.signature(available_cell_types).parameters["min_metacells"].default == MIN_METACELLS
    metadata = _metadata()
    assert _counts(available_cell_types(metadata, "AD", "brain", min_metacells=2)) == {"astrocyte": (2, 3)}
    assert _counts(available_cell_types(metadata, "AD", "brain", min_metacells=3)) == {}
    assert _counts(available_cell_types(metadata, "AD", "brain")) == {}


def test_disease_and_organ_match_ignoring_case_and_whitespace():
    result = available_cell_types(_metadata(), " ad ", "BRAIN", min_metacells=1)
    assert _counts(result) == {"astrocyte": (2, 3), "made-up cell": (1, 1)}
    assert result["disease_name"] == "AD"


def test_no_organ_covers_every_organ():
    assert _counts(available_cell_types(_metadata(), "AD", min_metacells=1))["astrocyte"] == (3, 3)


def test_keep_unknown_donor_counts_metacells_without_a_usable_donor():
    metadata = _metadata()
    assert _counts(available_cell_types(metadata, "AD", "brain", keep_unknown_donor=True,
                                        min_metacells=1))["astrocyte"] == (3, 4)
    # A collection applies to the named cell types only, ignoring case
    named = _counts(available_cell_types(metadata, "AD", "brain", keep_unknown_donor={"ASTROCYTE"},
                                         min_metacells=1))
    assert named["astrocyte"] == (3, 4)
    other = _counts(available_cell_types(metadata, "AD", "brain", keep_unknown_donor={"made-up cell"},
                                         min_metacells=1))
    assert other["astrocyte"] == (2, 3)


@pytest.mark.parametrize("keep_unknown_donor", [False, True])
def test_counts_equal_what_the_sampler_can_select(keep_unknown_donor):
    metadata = _metadata()
    result = available_cell_types(metadata, "AD", "brain", keep_unknown_donor=keep_unknown_donor,
                                  min_metacells=1)
    for cell_type, (n_disease, n_normal) in _counts(result).items():
        _, summary = sample_known_donor_cohort(
            metadata, {"disease": "AD", "cell_type": cell_type, "tissue_general": "brain"}, ALIASES,
            sample_size=10_000, keep_unknown_donor=keep_unknown_donor)
        assert (summary["disease"]["selected"], summary["control"]["selected"]) == (n_disease, n_normal)


DISEASES = pd.Series(["COVID-19 Virus Disease", "long COVID-19", "Alzheimer's Disease", "Heart Diseases",
                      "Crohn Disease", "normal", "unknown", "Unclassified"])


def test_disease_candidates_rank_names_containing_the_query_first():
    assert disease_candidates(DISEASES, "COVID-19") == ["COVID-19 Virus Disease", "long COVID-19"]
    assert disease_candidates(DISEASES, "alzheimer") == ["Alzheimer's Disease"]
    # A difflib close match without containment
    assert disease_candidates(DISEASES, "Alzheimers disease")[0] == "Alzheimer's Disease"


def test_disease_candidates_exclude_control_names_and_unrelated_queries():
    assert disease_candidates(DISEASES, "xyzzy plugh") == []
    assert disease_candidates(DISEASES, "normal") == []
    assert disease_candidates(DISEASES, "  ") == []
    assert len(disease_candidates(pd.Series([f"disease {i}" for i in range(30)]), "disease")) == 10


@pytest.mark.skipif(not os.path.exists(METADATA), reason="OmniCellTOSG metadata not available")
def test_disease_candidates_on_the_database_names():
    diseases = pd.read_parquet(METADATA, columns=["disease_BMG_name"])["disease_BMG_name"]
    assert disease_candidates(diseases, "COVID-19")[0] == "COVID-19 Virus Disease"
    assert disease_candidates(diseases, "alzheimer")[0] == "Alzheimer's Disease"
    assert disease_candidates(diseases, "xyzzy plugh") == []


def test_unknown_disease_lists_the_closest_disease_names():
    result = available_cell_types(_metadata(), "Alzheimer disease", "brain")
    assert not result["disease_found"] and result["cell_types"].empty
    assert format_cell_types(result) == ('No disease named "Alzheimer disease" in OmniCellTOSG. '
                                         'No disease name is close to it.')
    metadata = _metadata().assign(disease_BMG_name=lambda df: df.disease_BMG_name.replace("AD", "Alzheimer's Disease"))
    result = available_cell_types(metadata, "alzheimer", "brain")
    assert result["disease_candidates"] == ["Alzheimer's Disease"]
    assert format_cell_types(result) == ('No disease named "alzheimer" in OmniCellTOSG. '
                                         'Closest disease names: "Alzheimer\'s Disease".')


def test_known_disease_without_cell_types_in_the_organ_names_the_organs_that_have_some():
    text = format_cell_types(available_cell_types(_metadata(), "AD", "heart", min_metacells=1))
    assert text == ('No cell type has at least 1 disease and 1 normal metacells available for disease "AD" '
                    'in organ "heart". Organs where some cell type has: "brain".')
    # Under the default minimum no organ qualifies, so none is named
    assert format_cell_types(available_cell_types(_metadata(), "AD", "heart")) == (
        'No cell type has at least 10 disease and 10 normal metacells available for disease "AD" '
        'in organ "heart".')


def test_text_lists_names_under_lineage_headings_with_other_last():
    text = format_cell_types(available_cell_types(_metadata(), "AD", "brain", min_metacells=1))
    lines = text.splitlines()
    assert lines[0] == ('2 cell types have at least 1 disease and 1 normal metacells available before '
                        'sampling for disease "AD" in organ "brain", listed under Cell Ontology lineage '
                        'headings:')
    glial, other = lines.index("## glial cell"), lines.index(f"## {OTHER_HEADING}")
    assert lines[glial + 1] == '- "astrocyte" (disease 2, normal 3 metacells available before sampling)'
    assert lines[other + 1] == '- "made-up cell" (disease 1, normal 1 metacells available before sampling)'
    assert glial < other
    assert lines[-1] == (f"The analysis uses up to {DEFAULT_SAMPLE_SIZE:,} metacells per group (disease, "
                         f"normal); a larger group is sampled down to {DEFAULT_SAMPLE_SIZE:,}.")
    assert "donor" not in text.lower()


def test_lineage_file_has_one_heading_for_every_vocabulary_name():
    vocabulary = os.path.join(REPO_ROOT, "tools/omic_tools/tests/name_extraction/vocabulary/cell_types.tsv")
    with open(vocabulary, newline="", encoding="utf-8") as handle:
        names = [row["db_name"] for row in csv.DictReader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)]
    with open(LINEAGE_FILE, newline="", encoding="utf-8") as handle:
        mapped = [row["cell_type"] for row in csv.DictReader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)]
    assert sorted(mapped) == sorted(names) and len(set(mapped)) == len(mapped)
    lineage = load_lineage()
    assert lineage["microglial cell"] == "glial cell"
    assert lineage["macrophage;macrophage:monocyte-derived;macrophage:1;macrophage:2;macrophage:3"] == "macrophage"


def test_lookup_is_no_agent_tool_and_its_text_has_no_donor_wording():
    """The lookup only builds omic_analysis_tool's refusal message, which agents read.
    Reads the agent's source text, as test_cohort_diagnostics_never_reach_agents_or_the_report does."""
    with open(os.path.join(REPO_ROOT, "agent", "langgraph_agent.py")) as handle:
        source = handle.read()
    assert "cell_type_lookup_tool" not in source and "tools=[omic_analysis_tool]" in source
    texts = [format_cell_types(available_cell_types(_metadata(), disease, "brain", keep_unknown_donor=keep,
                                                    min_metacells=minimum), lineage={})
             for disease in ("AD", "A D", "nothing like it") for keep in (False, True) for minimum in (1, 10)]
    assert any('- "astrocyte"' in text for text in texts)
    for term in ("donor", "unknown", "diagnos"):
        assert not any(term in text.lower() for text in texts)

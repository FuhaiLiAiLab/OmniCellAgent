"""Offline tests for the name-extraction benchmark builders (no downloads)."""

import pandas as pd
import pytest

from tools.omic_tools.tests.name_extraction.build_vocabulary import (
    build_cell_type_table, build_disease_table)
from tools.omic_tools.tests.name_extraction.corpora import CorpusMention, Lrabr
from tools.omic_tools.tests.name_extraction.drafting import FieldIndex, Relation
from tools.omic_tools.tests.name_extraction.generate_candidates import (
    Pool, apply_caps, context_candidates, finalize, mark_loose_variants, merge_by_mention,
    names_concept, near_miss_candidates)
from tools.omic_tools.tests.name_extraction.obo import Ontology
from tools.omic_tools.tests.name_extraction.text import (
    abbreviation_core, norm_basic, norm_loose, sentence_around)

CL_OBO = r'''format-version: 1.2
data-version: releases/2025-01-01

[Term]
id: CL:0000000
name: cell

[Term]
id: CL:0000008
name: lymphocyte
is_a: CL:0000000 ! cell

[Term]
id: CL:0000002
name: T cell
synonym: "T lymphocyte" EXACT []
synonym: "T-cell" EXACT OMO:0003000 [PMID:1]
is_a: CL:0000008 ! lymphocyte

[Term]
id: CL:0000003
name: regulatory T cell
synonym: "Treg" EXACT []
synonym: "suppressor \"T\" cell" RELATED []
is_a: CL:0000002 {source="x"} ! T cell

[Term]
id: CL:0000005
name: CD4-positive T cell
is_a: CL:0000002 ! T cell

[Term]
id: CL:0000006
name: CD8-positive T cell
is_a: CL:0000002 ! T cell

[Term]
id: CL:0000007
name: B cell
is_a: CL:0000008 ! lymphocyte

[Term]
id: CL:0000009
name: obsolete old T cell
is_obsolete: true
replaced_by: CL:0000002

[Term]
id: UBERON:0000001
name: imported anatomy

[Typedef]
id: part_of
name: part of
'''

MONDO_OBO = r'''format-version: 1.2
data-version: releases/2026-01-01

[Term]
id: MONDO:0000001
name: disease

[Term]
id: MONDO:0000010
name: dementia
is_a: MONDO:0000001

[Term]
id: MONDO:0004975
name: Alzheimer disease
synonym: "AD" EXACT ABBREVIATION []
xref: UMLS:C0002395 {source="MONDO:equivalentTo", source="MONDO:MEDGEN"}
is_a: MONDO:0000010

[Term]
id: MONDO:0000011
name: frontotemporal dementia
xref: UMLS:C0338451 {source="MONDO:relatedTo"}
is_a: MONDO:0000010

[Term]
id: MONDO:0004980
name: atopic eczema
synonym: "atopic dermatitis" EXACT []
xref: UMLS:C0011615 {source="MONDO:equivalentTo"}
is_a: MONDO:0000001
'''


@pytest.fixture(scope="module")
def cl():
    return Ontology.parse(CL_OBO, prefix="CL:")


@pytest.fixture(scope="module")
def mondo():
    return Ontology.parse(MONDO_OBO, prefix="MONDO:")


# ------------------------------------------------------------------- obo
def test_obo_parser_reads_fields_and_drops_imported_terms(cl):
    assert cl.data_version == "releases/2025-01-01"
    assert "UBERON:0000001" not in cl.terms
    treg = cl.terms["CL:0000003"]
    assert [(s.text, s.scope, s.type) for s in treg.synonyms] == [
        ("Treg", "EXACT", None), ('suppressor "T" cell', "RELATED", None)]
    assert cl.terms["CL:0000002"].synonyms[1].type == "OMO:0003000"
    assert treg.is_a == ["CL:0000002"]
    assert cl.current_id("CL:0000009") == "CL:0000002"
    assert cl.label_index()["t cell"] == {"CL:0000002"}


def test_obo_hierarchy_distances(cl):
    assert cl.ancestors("CL:0000003") == {"CL:0000002": 1, "CL:0000008": 2, "CL:0000000": 3}
    assert cl.ancestors("CL:0000003", max_depth=1) == {"CL:0000002": 1}
    assert set(cl.descendants("CL:0000002")) == {"CL:0000003", "CL:0000005", "CL:0000006"}


def test_xref_index_can_require_equivalence(mondo):
    assert mondo.xref_index("UMLS:", require_source="MONDO:equivalentTo") == {
        "C0002395": {"MONDO:0004975"}, "C0011615": {"MONDO:0004980"}}
    assert "C0338451" in mondo.xref_index("UMLS:")


# ------------------------------------------------------------------- text
@pytest.mark.parametrize("left, right", [
    ("Alzheimer’s Disease", "Alzheimer disease"),
    ("B cells", "B cell"),
    ("type II pneumocyte", "type 2 pneumocyte"),
    ("CD8+ T cells", "CD8-positive T cell"),
    ("CD8 + T cells", "CD8+ T cell"),
    ("NK CD16−", "NK CD16 negative"),
    ("CD14- positive monocytes", "CD14-positive monocyte"),
    ("B-cells", "B cell"),
    ("alpha-beta T cell", "αβ T cell"),
])
def test_loose_normalization_merges_surface_variants(left, right):
    assert norm_loose(left) == norm_loose(right)


@pytest.mark.parametrize("left, right", [
    ("ALS", "AL"),
    ("NK CD16 +", "NK CD16-"),
    ("CD4-CD8- T cell", "CD4/CD8 T cells"),
])
def test_loose_normalization_keeps_different_names_apart(left, right):
    assert norm_loose(left) != norm_loose(right)


def test_basic_normalization_only_unifies_case_space_and_unicode():
    assert norm_basic("  Alzheimer’s   Disease ") == "alzheimer's disease"


@pytest.mark.parametrize("mention, known, expected", [
    ("NK cells", (), "NK"),
    ("Tregs", ("Treg",), "Treg"),
    ("Tregs", (), None),
    ("ILC3s", (), "ILC3"),
    ("ALS", (), "ALS"),
    ("COPD disease", (), "COPD"),
    ("pDC", (), "pDC"),
    ("theca", (), None),
    ("plasma", ("plasma",), None),
    ("ASTHMA", (), None),
    ("CD8+ T cells", (), None),
])
def test_abbreviation_core(mention, known, expected):
    assert abbreviation_core(mention, frozenset(known)) == expected


def test_sentence_around_returns_the_containing_sentence_and_trims():
    text = "Background here. AD brains show gliosis. Other text follows."
    start = text.index("AD")
    assert sentence_around(text, start, start + 2) == "AD brains show gliosis."
    long_text = "Intro. " + "word " * 200 + "TARGET " + "word " * 200 + ". End."
    start = long_text.index("TARGET")
    trimmed = sentence_around(long_text, start, start + 6, max_chars=80)
    assert "TARGET" in trimmed and len(trimmed) <= 82


# ------------------------------------------------------------- vocabulary
def _metadata():
    return pd.DataFrame({
        "source": ["s"] * 7,
        "dataset_id": ["d1", "d1", "d2", "d2", "d2", "d1", "d1"],
        "donor_id": ["a", "b", "a", "unknown", "c", "a", "b"],
        "CMT_id": ["CMT1", "CMT1", "CMT2", "CMT3", "", "CMT1", "CMT2"],
        "CMT_name": ["regulatory T cell;regulatory T cell:activated"] * 2
                    + ["T lymphocyte", "mystery cell", "", "regulatory T cell;regulatory T cell:activated",
                       "T lymphocyte"],
        "cell_type": ["Treg", "Tregs", "T cells", "odd", "", "Treg", "T cells"],
        "disease_BMG_id": ["BMG_1", "BMG_1", "", "BMG_9", "", "BMG_1", ""],
        "disease_BMG_name": ["Alzheimer's Disease"] * 2 + ["normal", "atopic dermatitis", "normal",
                                                           "Alzheimer's Disease", "normal"],
        "disease": ["AD"] * 2 + ["healthy", "eczema", "healthy", "AD", "healthy"],
        "tissue_general": ["brain"] * 7,
    })


def test_cell_type_table_links_component_heads_and_counts_usable_donors(cl):
    table, stats = build_cell_type_table(_metadata(), cl)
    rows = table.set_index("db_name")
    treg = rows.loc["regulatory T cell;regulatory T cell:activated"]
    assert treg["heads"] == "regulatory T cell" and treg["cl_ids"] == "CL:0000003"
    assert treg["cl_link"] == "label" and treg["is_composite"]
    assert (treg["n_rows"], treg["n_donors"], treg["n_datasets"]) == (3, 2, 1)
    assert rows.loc["T lymphocyte", "cl_link"] == "exact_synonym"
    assert rows.loc["mystery cell", "cl_link"] == "none"
    assert rows.loc["mystery cell", "n_donors"] == 0
    assert stats["rows_without_cell_type"] == 1


def test_disease_table_crosswalk_and_mondo_fallback(mondo):
    bmg = pd.DataFrame({"BioMedGraphica_ID": ["BMG_1"], "UMLS_ID": ["C0002395"], "MONDO_ID": [None],
                        "MeSH_ID": ["D000544"], "DO_ID": [None], "OMIM_ID": [None]})
    table, stats = build_disease_table(_metadata(), bmg, mondo)
    rows = table.set_index("db_name")
    alzheimer = rows.loc["Alzheimer's Disease"]
    assert alzheimer["mondo_ids"] == "MONDO:0004975"
    assert alzheimer["crosswalk"] == "biomedgraphica | mondo_umls_xref"
    eczema = rows.loc["atopic dermatitis"]
    assert eczema["mondo_ids"] == "MONDO:0004980" and eczema["crosswalk"] == "mondo_exact_synonym"
    assert rows.loc["normal", "is_control_like"] and rows.loc["normal", "mondo_ids"] == ""
    assert stats["bmg_ids_missing_from_biomedgraphica"] == ["BMG_9"]


# ---------------------------------------------------------------- drafting
@pytest.fixture()
def cell_index(cl):
    table = pd.DataFrame({
        "db_name": ["regulatory T cell;regulatory T cell:activated", "CD4-positive T cell",
                    "CD8-positive T cell"],
        "n_rows": ["30", "20", "10"],
        "cl_ids": ["CL:0000003", "CL:0000005", "CL:0000006"],
    })
    return FieldIndex("cell_type", table, cl, "cl_ids")


def test_drafts_follow_the_hierarchy(cell_index):
    resolve = cell_index.draft(cell_index.relate(["CL:0000003"]))
    assert resolve["expected_action"] == "resolve"
    assert resolve["gold_preferred"] == "regulatory T cell;regulatory T cell:activated"

    choose = cell_index.draft(cell_index.relate(["CL:0000002"]))
    assert choose["expected_action"] == "choose" and choose["n_narrower_total"] == 3
    # Same depth, so the most populated database term is listed first.
    assert [item["name"] for item in choose["gold_narrower"]] == [
        "regulatory T cell;regulatory T cell:activated", "CD4-positive T cell", "CD8-positive T cell"]

    decline = cell_index.draft(cell_index.relate(["CL:0000007"]))
    assert decline["expected_action"] == "decline"

    unplaced = cell_index.draft(cell_index.relate(["CL:9999999"]))
    assert unplaced["expected_action"] == "review"

    forced = cell_index.draft(cell_index.relate(["CL:0000003"]), "RELATED synonym")
    assert forced["expected_action"] == "review" and forced["basis"].startswith("RELATED synonym")


def _chain_index(n_db_children=0):
    """X:0 (in DB) <- X:1 <- X:2 <- X:3 <- X:4, plus X:10 with n DB children."""
    stanzas = ["[Term]\nid: X:0\nname: root disease"]
    stanzas += [f"[Term]\nid: X:{i}\nname: level {i}\nis_a: X:{i - 1}" for i in range(1, 5)]
    stanzas.append("[Term]\nid: X:10\nname: grouping class")
    stanzas += [f"[Term]\nid: X:1{i:02d}\nname: child {i}\nis_a: X:10" for i in range(n_db_children)]
    ontology = Ontology.parse("\n\n".join(stanzas) + "\n", prefix="X:")
    names = ["root disease"] + [f"child {i}" for i in range(n_db_children)]
    ids = ["X:0"] + [f"X:1{i:02d}" for i in range(n_db_children)]
    table = pd.DataFrame({"db_name": names, "n_rows": ["1"] * len(names), "ids": ids})
    return FieldIndex("disease", table, ontology, "ids")


def test_broader_terms_within_three_steps_are_review_beyond_are_decline():
    index = _chain_index()
    three_up = index.draft(index.relate(["X:3"]))
    assert three_up["expected_action"] == "review" and three_up["nearest_broader_depth"] == 3
    four_up = index.draft(index.relate(["X:4"]))
    assert four_up["expected_action"] == "decline" and four_up["nearest_broader_depth"] == 4
    assert four_up["gold_broader"] == [{"name": "root disease", "depth": 4}]


def test_grouping_concepts_with_many_db_children_are_review_not_choose():
    assert _chain_index(15).draft(_chain_index(15).relate(["X:10"]))["expected_action"] == "choose"
    index = _chain_index(16)
    draft = index.draft(index.relate(["X:10"]))
    assert draft["expected_action"] == "review" and draft["basis"].startswith("grouping concept")


def _merge_record(field, mention, concept, action, preferred, sources, count, names=False):
    return {"id": f"{field}-{concept}", "category": "abbreviation", "field": field,
            "mention": mention, "intended_concept": {"id": concept}, "corpus_count": count,
            "sources": sources, "ambiguity": {"lrabr_senses": 9}, "surface": {"names_concept": names},
            "draft": {"expected_action": action, "gold_preferred": preferred,
                      "n_narrower_total": 0, "basis": "b"}}


def test_merge_by_mention_keeps_one_item_per_string_with_alternatives():
    record = _merge_record
    records = [
        record("cell_type", "OPC", "CL:2", "resolve", "preosteoblast",
               [{"source": "celllink"}], 40),
        record("cell_type", "opc", "CL:1", "resolve", "oligodendrocyte precursor cell",
               [{"source": "lrabr"}, {"source": "celllink"}], 5),
        record("cell_type", "OPC", "CL:3", "decline", None, [{"source": "celllink"}], 2),
        record("disease", "OPC", "UMLS:C1", "review", None, [{"source": "medmentions"}], 3),
    ]
    merged = {r["field"]: r for r in merge_by_mention(records)}
    cell = merged["cell_type"]
    assert cell["intended_concept"]["id"] == "CL:1"          # LRABR evidence beats frequency
    assert [a["concept"]["id"] for a in cell["alternatives"]] == ["CL:2", "CL:3"]
    assert cell["ambiguity"] == {"lrabr_senses": 9, "n_concepts": 3, "db_senses": 2,
                                 "concept_disagreement": True, "other_field_concepts": 1}
    assert cell["corpus_count"] == 47 and cell["id"].startswith("abb-")
    assert merged["disease"]["ambiguity"]["other_field_concepts"] == 3


def test_a_concepts_own_name_outranks_corpus_majority():
    # CellLink links 'germ cell' to spermatogonium in testis papers; out of
    # context the string names the germ cell concept.
    records = [
        _merge_record("cell_type", "germ cell", "CL:0000020", "resolve", "spermatogonium",
                      [{"source": "celllink"}], 3),
        _merge_record("cell_type", "germ cell", "CL:0000586", "choose", None,
                      [{"source": "celllink"}], 2, names=True),
    ]
    [merged] = merge_by_mention(records)
    assert merged["intended_concept"]["id"] == "CL:0000586"
    assert [a["concept"]["id"] for a in merged["alternatives"]] == ["CL:0000020"]


def test_names_concept_allows_plurals_but_not_related_synonyms(cl):
    assert names_concept("T lymphocytes", "CL:0000002", cl)       # EXACT synonym, plural
    assert names_concept("regulatory T-cells", "CL:0000003", cl)  # label, plural and hyphen
    assert not names_concept('suppressor "T" cell', "CL:0000003", cl)  # RELATED synonym
    assert not names_concept("T cell", "CL:0000003", cl)
    assert not names_concept("obsolete old T cell", "CL:0000009", cl)  # obsolete term
    assert not names_concept("T cell", None, cl)


def test_categories_and_triviality(cell_index):
    known = frozenset({"Treg"})
    resolve = cell_index.draft(cell_index.relate(["CL:0000003"]))
    assert cell_index.categorize("Tregs", resolve, known)[0] == "abbreviation"
    category, surface = cell_index.categorize("regulatory T cells", resolve, known)
    assert category == "surface_variant" and not surface["draft_conflict"]
    choose = cell_index.draft(cell_index.relate(["CL:0000002"]))
    assert cell_index.categorize("T lymphocytes", choose, known)[0] == "granularity"
    assert cell_index.is_trivial("Regulatory T cell;Regulatory T cell:activated")
    assert not cell_index.is_trivial("regulatory T cell")


# ---------------------------------------------------------------- generator
def test_pool_merges_sources_and_skips_trivial(cell_index):
    pool = Pool()
    relation = cell_index.relate(["CL:0000003"])
    source = {"source": "cell_ontology", "review_reason": None}
    pool.add(cell_index, "Treg", "CL:0000003", "regulatory T cell", "CL", relation, source)
    pool.add(cell_index, "Treg", "CL:0000003", "regulatory T cell", "CL", relation, dict(source))
    pool.add(cell_index, "treg", "CL:0000003", "regulatory T cell", "CL", relation,
             {"source": "celllink", "review_reason": None}, count=4, casing={"treg": 4})
    pool.add(cell_index, "CD4-positive T cell", "CL:0000005", "x", "CL", relation, source)
    assert len(pool.items) == 1
    candidate = next(iter(pool.items.values()))
    assert candidate.mention == "treg" and candidate.corpus_count == 4
    assert [s["source"] for s in candidate.sources] == ["cell_ontology", "celllink"]
    assert pool.trivial[("cell_type", "cell_ontology")] == 1


def test_finalize_only_forces_review_when_every_source_asks(cell_index):
    pool = Pool()
    relation = cell_index.relate(["CL:0000003"])
    related = {"source": "celllink", "review_reason": "CellLink related (not exact) link"}
    pool.add(cell_index, "suppressor T cells", "CL:0000003", "regulatory T cell", "CL", relation,
             related)
    candidate = next(iter(pool.items.values()))
    lrabr = Lrabr({}, {})
    record = finalize(candidate, cell_index, lrabr, {}, {})
    assert record["draft"]["expected_action"] == "review"
    assert not record["surface"]["names_concept"]
    pool.add(cell_index, "suppressor T cells", "CL:0000003", "regulatory T cell", "CL", relation,
             {"source": "celllink", "link": "exact", "review_reason": None})
    record = finalize(candidate, cell_index, lrabr, {}, {})
    assert record["draft"]["expected_action"] == "resolve"


def test_finalize_does_not_review_a_string_that_names_its_concept(cell_index):
    # 'Treg' is an EXACT synonym of regulatory T cell, so a related-only
    # CellLink link does not make its meaning uncertain.
    pool = Pool()
    pool.add(cell_index, "Treg", "CL:0000003", "regulatory T cell", "CL",
             cell_index.relate(["CL:0000003"]),
             {"source": "celllink", "review_reason": "CellLink related (not exact) link"})
    lrabr = Lrabr({"Treg": {"E1": ["regulatory T cell"]}}, {})
    record = finalize(next(iter(pool.items.values())), cell_index, lrabr, {}, {})
    assert record["draft"]["expected_action"] == "resolve"
    assert record["surface"]["names_concept"] and record["surface"]["loose_variants"] == []
    assert record["category"] == "abbreviation" and record["ambiguity"]["lrabr_senses"] == 1
    assert record["id"].startswith("abb-") and record["labels"]["adjudicated"] is None


def test_loose_variants_are_linked_and_conflicting_drafts_flagged():
    def record(mention, action, preferred, narrower=(), context=None):
        return {"field": "cell_type", "mention": mention, "context": context,
                "surface": {"normalized": norm_loose(mention), "loose_variants": []},
                "ambiguity": {"loose_variant_conflict": False},
                "draft": {"expected_action": action, "gold_preferred": preferred,
                          "gold_narrower": [{"name": n} for n in narrower]}}
    singular = record("germ cell", "resolve", "spermatogonium")
    plural = record("germ cells", "choose", None, ["oocyte", "spermatogonium"])
    hyphen = record("germ-cell", "resolve", "spermatogonium")
    other = record("B cell", "resolve", "B cell")
    in_context = record("germ cells", "decline", None, context="Germ cells of the fly.")
    mark_loose_variants([singular, plural, hyphen, other, in_context])
    assert singular["surface"]["loose_variants"] == ["germ cells", "germ-cell"]
    assert all(r["ambiguity"]["loose_variant_conflict"] for r in (singular, plural, hyphen))
    assert other["surface"]["loose_variants"] == [] and not other["ambiguity"]["loose_variant_conflict"]
    assert in_context["surface"]["loose_variants"] == []
    agreeing = [record("NK cell", "resolve", "NK"), record("NK cells", "resolve", "NK")]
    mark_loose_variants(agreeing)
    assert not any(r["ambiguity"]["loose_variant_conflict"] for r in agreeing)


def test_near_miss_keeps_unrelated_siblings_only(cell_index):
    items = near_miss_candidates(cell_index, "CL", per_concept=2, min_score=0.3, cap=10)
    assert {c.concept for c in items} == set()  # every sibling here is in the DB or related
    b_index = FieldIndex("cell_type", pd.DataFrame({"db_name": ["T cell"], "n_rows": ["5"],
                                                    "cl_ids": ["CL:0000002"]}),
                         cell_index.ontology, "cl_ids")
    items = near_miss_candidates(b_index, "CL", per_concept=2, min_score=0.3, cap=10)
    assert [c.concept for c in items] == ["CL:0000007"]
    assert items[0].sources[0]["near_miss_of"] == "T cell"


def test_context_items_contrast_db_sense_with_non_disease_sense(mondo, cell_index):
    disease_table = pd.DataFrame({"db_name": ["Alzheimer's Disease"], "n_rows": ["9"],
                                  "mondo_ids": ["MONDO:0004975"], "umls_ids": ["C0002395"]})
    diseases = FieldIndex("disease", disease_table, mondo, "mondo_ids",
                          umls_to_nodes=mondo.xref_index("UMLS:", "MONDO:equivalentTo"))
    lrabr = Lrabr({"AD": {"E1": ["Alzheimer disease"]}}, {})
    mm = [
        CorpusMention("medmentions", "PMID:1", "train", "AD", "C0002395", "annotated", "T047",
                      "Microglia in AD brains."),
        CorpusMention("medmentions", "PMID:2", "train", "AD", "C0443147", "annotated", "T045",
                      "Inheritance was AD in this family."),
        CorpusMention("medmentions", "PMID:3", "dev", "AD", "C0011615", "annotated", "T047",
                      "AD skin lesions were biopsied."),
        CorpusMention("medmentions", "PMID:4", "dev", "HD", "C9", "annotated", "T047", "HD only."),
    ]
    items, summary = context_candidates({"disease": diseases, "cell_type": cell_index}, [], mm, lrabr,
                                        per_sense=3, seed=1, label_of=lambda f, c: None)
    assert set(summary) == {"AD"}
    records = {c.concept: finalize(c, diseases, lrabr, {}, {}) for c in items}
    assert records["C0002395"]["draft"]["expected_action"] == "resolve"
    assert records["C0443147"]["draft"]["expected_action"] == "decline"
    assert "not a disease" in records["C0443147"]["draft"]["basis"]
    assert records["C0011615"]["draft"]["expected_action"] == "decline"
    assert {r["category"] for r in records.values()} == {"context_abbreviation"}
    assert {r["pair_id"] for r in records.values()} == {"AD"}
    assert not any(r["surface"]["context_is_table_like"] for r in records.values())


def test_caps_keep_positive_drafts_and_prioritise_traps():
    def record(i, action, trap=False, count=0):
        return {"id": f"x{i}", "category": "synonym", "field": "disease", "lexical_trap": trap,
                "corpus_count": count, "draft": {"expected_action": action}}
    records = [record(i, "resolve") for i in range(5)] + [record(10, "review", trap=True)] \
        + [record(i, "review", count=i) for i in range(11, 16)]
    kept, stats = apply_caps(records, cap=2, seed=0)
    kept_ids = {r["id"] for r in kept}
    assert {f"x{i}" for i in range(5)} <= kept_ids
    assert "x10" in kept_ids and "x15" in kept_ids and len(kept_ids) == 7
    assert {(s["expected_action"], s["generated"], s["written"]) for s in stats} == {
        ("resolve", 5, 5), ("review", 6, 2)}


def test_capped_sample_is_stable_when_one_candidate_leaves():
    def record(i):
        return {"id": f"x{i:03d}", "category": "synonym", "field": "disease", "lexical_trap": False,
                "corpus_count": 1, "draft": {"expected_action": "review"}}
    records = [record(i) for i in range(200)]
    kept, _ = apply_caps(records, cap=50, seed=13)
    before = {r["id"] for r in kept}
    dropped = sorted(before)[0]
    kept, _ = apply_caps([r for r in records if r["id"] != dropped], cap=50, seed=13)
    after = {r["id"] for r in kept}
    assert len(before - after) == 1 and len(after - before) == 1  # only the freed slot changes


def test_relation_defaults_are_empty():
    relation = Relation()
    assert not (relation.nodes or relation.equal or relation.narrower or relation.broader)

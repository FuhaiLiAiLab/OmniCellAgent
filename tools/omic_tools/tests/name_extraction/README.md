# Name-extraction benchmark: mapping biologists' terms to the OmniCellTOSG vocabulary

This benchmark tests whether a disease or cell-type term, as a biologist would type
it, is mapped to the right database value, offered as a choice, or declined. It
responds to a reviewer's request to show how the system handles non-standard
queries and abbreviations.

The omic tool filters on exact database strings (`disease_BMG_name`, `CMT_name`,
`tissue_general`) with case-insensitive whole-string equality, so every test
item names the exact database value(s) it should resolve to.

**Status: steps 1–2 of 6 are done.** The vocabulary is frozen, and candidates
carry machine-drafted labels. **Nothing here is a gold label yet.**

| Step | What | Where |
|---|---|---|
| 1 | Freeze the query vocabulary with counts and ontology links | `vocabulary/` |
| 2 | Generate candidate test strings with draft labels | `candidates/` |
| 3 | Sample per category; rewrite into user questions | to do |
| 4 | Two annotators label; adjudicate; report kappa | to do |
| 5 | Split dev/test by concept | to do |
| 6 | Harness: exact+difflib (current) vs resolver vs LLM+resolver | to do |

## Layout

```
name_extraction/
├── README.md
├── sources.py               pinned external sources, SHA-256-verified download cache
├── obo.py                   minimal OBO reader (Cell Ontology, MONDO) with is_a walks
├── text.py                  normalisation, abbreviation detection, sentence extraction
├── corpora.py               LRABR, CellLink, and MedMentions loaders
├── drafting.py              concept -> database relation and draft labels
├── build_vocabulary.py      step 1
├── generate_candidates.py   step 2
├── test_name_extraction.py  offline unit tests (no downloads)
├── vocabulary/              step-1 outputs
└── candidates/              step-2 outputs
```

## Regenerate

From the project root, in an environment with pandas, pyarrow, pyyaml, and
pytest (tested with Python 3.10):

```bash
python -m tools.omic_tools.tests.name_extraction.build_vocabulary
python -m tools.omic_tools.tests.name_extraction.generate_candidates
python -m pytest tools/omic_tools/tests/name_extraction
```

The first run downloads about 160 MB of pinned sources into
`~/.cache/omnicellagent/name_extraction`; set `OMNICELL_NAME_BENCH_CACHE` to use
another directory. Every file is checked against the SHA-256 in `sources.py`, so
a changed upstream file stops the run instead of silently changing the benchmark.
Step 1 takes about 15 s and step 2 about 2 min. Both write a `manifest.json`
with input checksums, ontology versions, parameters, and the git commit.
Step 2 refuses to run if the ontology versions differ from step 1's manifest.

## Step 1: `vocabulary/`

| File | Rows | Content |
|---|---|---|
| `cell_types.tsv` | 762 | `CMT_name` values, Cell Ontology links, counts |
| `diseases.tsv` | 144 | `disease_BMG_name` values, UMLS/MONDO/MeSH/DOID/OMIM links, counts |
| `organs.tsv` | 65 | `tissue_general` values, counts |
| `manifest.json` | – | inputs, checksums, link statistics |

Every table has `n_rows` (metadata rows, i.e. SEACells metacells),
`n_donors` (usable study-scoped donors: `(source, dataset_id, donor_id)`, the same
rule as `cohort_diagnostics.py`), and `n_datasets`. List-valued columns are
joined with ` | `. Load a table with `build_vocabulary.read_tsv`.

**Cell types.** 77 names are composite (`"B cell;B cell:activated"`); `heads`
holds each component's part before `:`. Every head is linked to the Cell
Ontology (the 2025-01-07 release CellLink uses). It is matched by exact label
(747 names), else by EXACT synonym (14, e.g. `type II pneumocyte` →
`pulmonary alveolar type 2 cell`), else by the label of an obsolete term with a
replacement (1). 760 of 762 names are linked. 25,775 metadata rows have no cell
type.

**Diseases.** IDs come from the BioMedGraphica disease table, then MONDO
equivalence cross-references to UMLS, then an exact MONDO label or EXACT-synonym
match on the name (column `crosswalk`). 131 of the 141 non-control names have a
MONDO ID and 116 have a UMLS CUI. Five have no link: Acterol, Cell stress,
Crohn's disease of ileum, Disorder following viral disease, and Trisomy 18
Syndrome. Twenty BioMedGraphica IDs used by the database are missing from the
BioMedGraphica table (listed in the manifest). `normal`, `unknown`, and
`Unclassified` are flagged `is_control_like` and are never gold targets.

`raw_labels_unaudited` lists the source-study labels mapped to each name. That
mapping has known errors (raw "malignant pancreatic neoplasm" →
`parametrium malignant neoplasm`), so do not use these as gold synonyms without
review.

## Step 2: `candidates/`

### Sources

| Source | Contributes |
|---|---|
| Cell Ontology / MONDO | labels and synonyms (all scopes) of concepts linked to database terms |
| LRABR (NLM SPECIALIST Lexicon 2025) | abbreviations whose expansion names a database concept |
| CellLink train+val (NLM, 2026) | cell mentions from recent papers, linked to Cell Ontology IDs |
| MedMentions full (CZI, 2019) | disease mentions from PubMed abstracts, linked to UMLS CUIs (joined to MONDO) |
| Ontology siblings | absent concepts next to a database concept (near-miss traps) |

A candidate string equal to a full database name (case-insensitive) is
excluded, because today's exact matcher already finds it. The manifest counts
these exclusions.

### Files

| File | Category | What it tests |
|---|---|---|
| `synonym.jsonl` | synonym | different name, same concept (`microglia` → `microglial cell`) |
| `abbreviation.jsonl` | abbreviation | abbreviation without context (`SLE`, `Tregs`, `NK cells`, `OPC`) |
| `surface_variant.jsonl` | surface_variant | plural/punctuation/composite variants (`B cells` → `B cell;B cell:activated`) |
| `granularity.jsonl` | granularity | concept present only as 2–15 narrower terms (`lung cancer` → 5 subtypes) |
| `absent_concept.jsonl` | absent_concept | concept not in the database (`atopic dermatitis`) |
| `near_miss.jsonl` | near_miss | absent ontology sibling that looks like a database term |
| `context_abbreviation.jsonl` | context_abbreviation | one abbreviation, real sentences, different meanings (`MS` = multiple sclerosis vs mass spectrometry) |
| `sense_inventory.jsonl` | – | every LRABR sense of each abbreviation, placed against the database; for writing contrastive contexts in step 3 |
| `summary.tsv` | – | generated vs written count per (category, field, draft action) |

### Record schema

The real `OPC` record from `abbreviation.jsonl`, with long lists shortened:

```json
{
  "id": "abb-e67a9eee4d",
  "category": "abbreviation",
  "field": "cell_type",
  "mention": "OPC",
  "query": null,
  "context": null,
  "pair_id": null,
  "intended_concept": {"id": "CL:0002453", "label": "oligodendrocyte precursor cell", "ontology": "CL"},
  "draft": {
    "expected_action": "resolve",
    "gold_preferred": "oligodendrocyte precursor cell",
    "gold_equal": ["oligodendrocyte precursor cell"],
    "gold_narrower": [],
    "n_narrower_total": 0,
    "gold_broader": [{"name": "progenitor cell", "depth": 1}, "..."],
    "nearest_broader_depth": 1,
    "basis": "a database term carries the concept's id"
  },
  "surface": {"normalized": "opc", "abbreviation_core": "OPC", "variant_of": [],
              "has_marker_token": false, "draft_conflict": false,
              "names_concept": false, "loose_variants": []},
  "ambiguity": {"lrabr_senses": 7, "ontology_concepts": 0, "corpus_concepts": 3, "n_concepts": 3,
                "db_senses": 2, "concept_disagreement": true, "other_field_concepts": 1,
                "loose_variant_conflict": false},
  "alternatives": [
    {"concept": {"id": "CL:0002164", "label": "external pillar cell of cochlea", "ontology": "CL"},
     "expected_action": "decline", "gold_preferred": null, "n_narrower_total": 0,
     "basis": "...", "sources": ["celllink"], "corpus_count": 1},
    "... preosteoblast (resolve) ..."
  ],
  "lexical_neighbors": [],
  "lexical_trap": false,
  "corpus_count": 22,
  "sources": [{"source": "lrabr", "expansion": "oligodendrocyte precursor cell", "sense": "E0698774",
               "n_senses": 7}, "..."],
  "examples": [{"doc": "36384442_30", "split": "train", "text": "...Oligodendrocyte precursor cells (OPC)..."}],
  "labels": {"annotator_a": null, "annotator_b": null, "adjudicated": null, "notes": null}
}
```

- **One item per string.** Items without context are unique per field and
  string, ignoring case and whitespace.
  - If the sources link a string to several concepts, the best-supported one
    is primary and the others appear in `alternatives`. The ranking is:
    1. a concept the string names — its ontology label or an EXACT synonym,
       ignoring case, plurals and punctuation (`surface.names_concept`);
    2. LRABR;
    3. corpus frequency.
  - Whether a concept is in the database plays no part. So `HD` drafts to
    Huntington disease (`decline`), with `Heart Diseases` as a `resolve`
    alternative, and `germ cell` drafts to germ cell (`choose` among 4
    subtypes), although CellLink's testis papers mostly link it to
    spermatogonium.
  - `db_senses >= 2` means the string names two different database terms.
    CellLink uses `OPC` for oligodendrocyte precursor cells and for
    preosteoblasts.
- **Loose variants.** `surface.loose_variants` lists the other written items
  that differ only in ways `surface.normalized` ignores:
  - case, plural, hyphens and Greek letters;
  - how a marker is written (`CD8+`, `CD8 +` and `CD8 positive` match;
    `CD16+` and `CD16-` do not).

  `ambiguity.loose_variant_conflict` flags groups whose drafts disagree: 23
  groups, 49 items. In most of them, CellLink papers linked the same string to
  different concepts (`tuft cell` → brush cell vs `tuft cells` → choose).
  Label each group together, and keep it in one split in step 5.
- **Context items** are one per sentence, grouped by `pair_id` (the
  abbreviation).
  - `surface.context_contains_label` marks glossary sentences that spell out
    the answer ("DC, dendritic cell; ...").
  - `surface.context_is_table_like` marks table rows. Both kinds are poor
    query contexts and should usually be dropped in step 3.
- **`lexical_trap`** means the closest database name by edit similarity
  (score ≥ 0.6) is not an acceptable answer. The current `difflib` fallback
  would suggest that trap.
- **`query`** is filled in step 3; **`labels`** in step 4.

### Draft rules (`drafting.py`)

| Relation of the concept to database terms | Draft |
|---|---|
| a database term carries the same ontology ID | `resolve` |
| no equal term; 2–15 database terms are is_a descendants | `choose` |
| no equal term; more than 15 descendants (grouping class such as `disease`) | `review` |
| no equal term; exactly 1 descendant | `review` |
| nothing equal or narrower; nearest database ancestor ≤ 3 is_a steps up | `review` (`colon carcinoma` → `colorectal neoplasm`, 3 steps) |
| concept in the ontology, otherwise unrelated | `decline` (nearest ancestor still listed) |
| concept cannot be placed in the ontology | `review` |
| a non-EXACT ontology synonym or a CellLink "related" link is the only evidence, and the string is not the concept's own name | `review` |
| context sense whose UMLS semantic types are not disease-like | `decline` |

`gold_narrower` and `gold_broader` list the nearest terms with their is_a
distance, so annotators can apply a strict policy (equal only) or a lenient
one (equal plus narrower or broader). **Whether a broader term is acceptable is
the benchmark owner's decision.** The drafts do not make it.

### Sampling

Every `resolve` and `choose` draft, every context item, and every near miss is
written. Other buckets (category × field × draft action) keep at most `--cap`
items (default 300).

- **Order.** Lexical traps are kept first, then frequent corpus strings, then a
  seeded draw.
- **Stability.** The draw hashes each item's id, so a rerun after a small
  generator change keeps the rest of the sample, and labels keyed by `id`
  stay attached.
- **Counts.** `summary.tsv` shows generated vs written counts per bucket.

Current output:

| File | Items |
|---|---|
| synonym | 3,725 |
| abbreviation | 1,742 |
| surface_variant | 422 |
| granularity | 316 |
| absent_concept | 370 |
| context_abbreviation | 1,321 |
| near_miss | 128 |
| sense_inventory | 817 |

The 6,703 items without context draft to resolve 4,394, review 1,202,
decline 624 and choose 355.

## Known limitations

- **Drafts inherit corpus annotation noise.** One MedMentions abstract links
  "lung cancer" to NSCLC; `alternatives` and `concept_disagreement` expose most
  of these cases. Corpus strings also include truncations ("reast cancer") and
  coordinations ("Th1 and Th17 cell") that annotators should drop.
- **Formulaic Cell Ontology synonyms dominate the synonym file.** Examples are
  "CD8-positive, alpha-beta T-lymphocyte" and "... T-cell". Step 3 should
  stratify by `sources` and favour CellLink/MedMentions strings, which reflect
  how authors write.
- **Paper mentions are not typed queries.** Step 3 rewrites them as user
  questions. Abbreviations defined in a paper may be ambiguous without that paper.
- **Duplicate database concepts are not linked by ID.** `Parkinson Disease`
  and `Idiopathic Parkinsonism or Parkinson's disease` are an example, so
  `gold_equal` may list only one of them. Annotators must add the other.
- **Some senses cannot be placed.** MedMentions uses UMLS 2017AA; a CUI without
  a MONDO equivalence stays `review`.
- **Abbreviation detection is heuristic:** string shape plus LRABR membership.
  See `text.abbreviation_core`.
- **Items are merged ignoring case.** `PiD` (Pick disease) and `PID` (pelvic
  inflammatory disease) are one item; its primary sense is PID and Pick
  disease is an alternative. This matches the case-insensitive database
  filter, but annotators should check abbreviations whose meaning depends on
  case.
- **Organs are covered only in the vocabulary;** there are no organ candidates.
- **CellLink's test split is released without annotations,** so only train+val
  are used.

## Sources, licenses, citation

| Source | Version | License |
|---|---|---|
| Cell Ontology (Diehl et al. 2016) | 2025-01-07, copy shipped with CellLink | CC BY 4.0 |
| MONDO (Vasilevsky et al. 2022) | v2026-09-01 | CC BY 4.0 |
| NLM CellLink (Rotenberg et al. 2026; doi:10.5281/zenodo.18090009) | Zenodo 2025-12-29 | public domain (US Government work) |
| MedMentions (Mohan & Li, AKBC 2019) | full corpus, commit 14578232 | CC0 1.0 |
| SPECIALIST Lexicon LRABR (NLM) | 2025 release | NLM SPECIALIST terms; check before redistributing the raw table |
| BioMedGraphica disease table | local copy, checksum in manifest | internal |

Exact URLs and checksums are in `sources.py` and in both manifests.

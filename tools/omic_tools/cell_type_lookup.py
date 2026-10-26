"""List the cell types a disease-vs-normal comparison can use, under lineage headings."""

import csv
import json
import re
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd

from tools.omic_tools.cohort_diagnostics import _usable_donor_mask
from tools.omic_tools.donor_sampling import (DEFAULT_SAMPLE_SIZE, MIN_METACELLS_PER_GROUP,
                                             _disease_mask, _normal_mask, _scope)

# CMT_name -> Cell Ontology lineage heading; regenerate with scripts/build_cell_type_lineage.py
LINEAGE_FILE = Path(__file__).with_name("cell_type_lineage.tsv")
OTHER_HEADING = "Other"
POINTER_COLUMNS = ["matrix_file_path", "matrix_row_idx"]
LOOKUP_COLUMNS = ["CMT_name", "disease_BMG_name", "tissue_general", "dataset_id", "donor_id",
                  *POINTER_COLUMNS]
# Fewest disease and fewest normal metacells for a cell type to be listed: the minimum
# the analysis itself enforces
MIN_METACELLS = MIN_METACELLS_PER_GROUP
# difflib.get_close_matches' default cutoff
CLOSE_MATCH_CUTOFF = 0.6
MAX_DISEASE_CANDIDATES = 10


def read_lookup_metadata(path):
    """The metadata columns available_cell_types needs, from cell_metadata_with_mappings.parquet."""
    return pd.read_parquet(path, columns=LOOKUP_COLUMNS)


def _words(text):
    """Lower-cased words of letters and digits, space-joined ("COVID-19" -> "covid 19")."""
    return " ".join(re.sub(r"[^0-9a-z]+", " ", str(text).lower()).split())


def disease_candidates(values, query, limit=MAX_DISEASE_CANDIDATES):
    """Database disease names closest to a query that matches none exactly, best first.

    Names whose words contain the query's words in order (those starting with them
    first), or are contained in them, come before difflib close matches; ties are
    ranked by difflib ratio. Control-like names (normal, unknown, ...) are never
    candidates.
    """
    target = _words(query)
    if not target:
        return []
    names = sorted(set(values[_disease_mask(values)].astype(str).str.strip()))
    scores = {}
    for name in names:
        words = _words(name)
        ratio = SequenceMatcher(None, target, words).ratio()
        if f" {words} ".startswith(f" {target} "):
            scores[name] = (0, -ratio)
        elif f" {target} " in f" {words} " or f" {words} " in f" {target} ":
            scores[name] = (1, -ratio)
        elif ratio >= CLOSE_MATCH_CUTOFF:
            scores[name] = (2, -ratio)
    return sorted(scores, key=lambda name: (scores[name], name.lower()))[:limit]


def _pool_counts(eligible, keys, disease, cell_type_column, disease_column):
    """Disease and normal metacells per key, counted only in the datasets that hold both
    groups for that key, as the workflow selects them (shared_datasets_only); a key that no
    dataset holds both groups of is dropped."""
    keys = [keys] if isinstance(keys, str) else list(keys)
    pools = {
        "disease_metacells": eligible.loc[_disease_mask(eligible[disease_column], disease)],
        "normal_metacells": eligible.loc[_normal_mask(eligible[disease_column])],
    }
    per_dataset = pd.DataFrame({
        name: pool.drop_duplicates([cell_type_column, *POINTER_COLUMNS]).groupby([*keys, "dataset_id"]).size()
        for name, pool in pools.items()
    }).dropna()  # a dataset missing either group is not compared
    return per_dataset.groupby(level=keys).sum().astype(int)


def available_cell_types(metadata, disease, organ=None, keep_unknown_donor=False,
                         min_metacells=MIN_METACELLS, cell_type_column="CMT_name",
                         disease_column="disease_BMG_name", organ_column="tissue_general"):
    """Cell types with at least min_metacells disease and min_metacells normal metacells
    the sampler can select.

    Follows sample_known_donor_cohort as the workflow calls it (shared_datasets_only):
    organ matches ignoring case, disease matches ignoring case and surrounding
    whitespace, duplicate matrix rows count once, only datasets that hold both groups
    of the cell type count, and only metacells with a usable donor ID count, except
    for the cell types in keep_unknown_donor (True: every cell type; a collection of
    names: those, ignoring case). organ=None covers all organs. Counts are before
    sampling, which caps each group at the workflow's sample size.

    Return a dict: `disease_found` (False if no metacell carries the disease),
    `disease_name` (its database spelling), `disease_candidates` (closest database
    names when it is not found), `cell_types` (DataFrame of cell_type,
    disease_metacells and normal_metacells, sorted by name) and
    `organs_with_cell_types` (organs where some cell type meets the minimum).
    """
    required = {cell_type_column, disease_column, organ_column, "dataset_id", "donor_id", *POINTER_COLUMNS}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"Cell-type lookup requires metadata columns: {', '.join(sorted(missing))}")

    result = {
        "disease": disease, "organ": organ, "min_metacells": min_metacells,
        "disease_found": False, "disease_name": None, "disease_candidates": [],
        "organs_with_cell_types": [],
        "cell_types": pd.DataFrame(columns=["cell_type", "disease_metacells", "normal_metacells"]),
    }
    found = _disease_mask(metadata[disease_column], disease)
    if not found.any():
        result["disease_candidates"] = disease_candidates(metadata[disease_column], disease)
        return result
    result["disease_found"] = True
    result["disease_name"] = metadata.loc[found, disease_column].astype(str).str.strip().value_counts().index[0]

    if keep_unknown_donor is True:
        eligible = np.ones(len(metadata), dtype=bool)
    else:
        keep = {str(name).lower() for name in (keep_unknown_donor or ())}
        names = metadata[cell_type_column].astype("string").str.lower()
        eligible = _usable_donor_mask(metadata["donor_id"]) | names.isin(keep).fillna(False).to_numpy(bool)
    named = metadata[cell_type_column].astype("string").str.strip().fillna("").ne("")
    eligible = metadata.loc[eligible & named.to_numpy(bool)]

    def listed(counts):
        return counts[(counts >= min_metacells).all(axis=1)]

    per_organ = listed(_pool_counts(
        eligible.assign(_organ=eligible[organ_column].astype("string").str.lower()),
        ["_organ", cell_type_column], disease, cell_type_column, disease_column))
    result["organs_with_cell_types"] = sorted({key[0] for key in per_organ.index})
    if organ is not None:
        eligible = _scope(eligible, {organ_column: organ}, disease_column)
    counts = listed(_pool_counts(eligible, cell_type_column, disease, cell_type_column, disease_column))
    counts = counts.rename_axis("cell_type").reset_index()
    result["cell_types"] = counts.sort_values("cell_type", key=lambda names: names.str.lower(),
                                              ignore_index=True)
    return result


def load_lineage(path=LINEAGE_FILE):
    """Lower-cased CMT_name -> lineage heading."""
    with open(path, newline="", encoding="utf-8") as handle:
        rows = csv.DictReader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)
        return {row["cell_type"].lower(): row["heading"] for row in rows}


def _quoted(names):
    return ", ".join(json.dumps(name, ensure_ascii=False) for name in names)


def format_cell_types(result, lineage=None, sample_size=DEFAULT_SAMPLE_SIZE):
    """The lookup as text for the omic agent: one quoted name per line, under lineage headings."""
    disease, organ, minimum = result["disease"], result["organ"], result["min_metacells"]
    where = f'organ "{organ}"' if organ else "any organ"
    if not result["disease_found"]:
        text = f'No disease named "{disease}" in OmniCellTOSG.'
        if result["disease_candidates"]:
            return text + " Closest disease names: " + _quoted(result["disease_candidates"]) + "."
        return text + " No disease name is close to it."
    disease = result["disease_name"]
    table = result["cell_types"]
    if table.empty:
        text = (f"No cell type has at least {minimum} disease and {minimum} normal metacells "
                f'from datasets that hold both groups for disease "{disease}" in {where}.')
        if result["organs_with_cell_types"]:
            text += " Organs where some cell type has: " + _quoted(result["organs_with_cell_types"]) + "."
        return text

    lineage = load_lineage() if lineage is None else lineage
    headings = table.cell_type.map(lambda name: lineage.get(name.lower(), OTHER_HEADING))
    lines = [f"{len(table)} cell types have at least {minimum} disease and {minimum} normal metacells "
             f'from datasets that hold both groups for disease "{disease}" in {where}, listed under '
             f"Cell Ontology lineage headings:"]
    for heading in sorted(set(headings), key=lambda name: (name == OTHER_HEADING, name.lower())):
        lines.append(f"\n## {heading}")
        for row in table[headings == heading].itertuples(index=False):
            lines.append(f"- {json.dumps(row.cell_type, ensure_ascii=False)} "
                         f"(disease {row.disease_metacells:,}, normal {row.normal_metacells:,} "
                         f"metacells available before sampling)")
    lines.append(f"\nThe analysis uses up to {sample_size:,} metacells per group (disease, normal); "
                 f"a larger group is sampled down to {sample_size:,}.")
    return "\n".join(lines)

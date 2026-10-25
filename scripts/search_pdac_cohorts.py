#!/usr/bin/env python
"""Brute-force search for the cell-type labels behind the PDAC handoff cohorts.

The handoff overview (webapp/sessions/case_study_handoff_20260923/00_overview.csv) gives
each PDAC cohort's disease / normal metacell counts but only a high-level cell-type name.
Every PDAC pool is below the 1,000-per-group cap, so retrieval involves no sampling and
the right labels must reproduce the counts. This script tries every label related to
each cohort and reports which ones reproduce them. It only counts metacells: no
differential expression, no enrichment.

Search space, per cohort and per group. Disease and normal are searched independently,
because a cohort's control may be a different cell type:
  - label column: CMT_name (what the agent matches) or cell_type (the source study's label)
  - labels: one label, or a union of up to --max-union labels, drawn from the labels whose
    name contains one of the cohort's keywords; matching ignores case, like the sampler
  - organ filter: tissue_general == "pancreas", or none
  - donor rule: "known_donor" drops metacells without a usable donor ID, as the agent's
    sampler does (tools/omic_tools/donor_sampling.py); "keep_unknown" keeps them

A count match can be a coincidence, so every match is compared with the metacells the
handoff actually used (metadata/labels.csv, matched on matrix_file_path + matrix_row_idx).

Usage:
    python scripts/search_pdac_cohorts.py [--max-union 2] [--out results.csv]
"""

import argparse
import itertools
import os
import sys

import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)

from utils.path_config import get_path
from tools.omic_tools.cohort_diagnostics import _usable_donor_mask

DISEASE = "pancreatic ductal adenocarcinoma"
POINTER = ["matrix_file_path", "matrix_row_idx"]
COLUMNS = ("CMT_name", "cell_type")
ORGANS = ("pancreas", None)
DONOR_RULES = ("known_donor", "keep_unknown")
# Words related to each cohort's high-level name in the overview (keys = its cell_type column)
KEYWORDS = {
    "cancer_cell": ["epithel", "duct", "malignant", "tumor", "tumour", "cancer", "carcinoma", "neoplas"],
    "CAF_PENDING_REVIEW": ["fibro", "caf", "stellate", "mesenchym", "pericyte"],
    "macrophage": ["macrophage", "monocyte", "myeloid", "histiocyte", "kupffer"],
}


def load_metadata():
    root = get_path("external.omnicell_data_root", absolute=True)
    columns = [*COLUMNS, "tissue_general", "disease_BMG_name", "donor_id", *POINTER]
    meta = pd.read_parquet(os.path.join(root, "cell_metadata_with_mappings.parquet"), columns=columns)
    meta["pointer"] = meta.matrix_file_path.astype(str) + "#" + meta.matrix_row_idx.astype(str)
    return meta


def retrieval_frames(meta):
    """Return {(group, organ, donor_rule): metacells} for every retrieval setting."""
    disease = meta.disease_BMG_name.astype("string").str.strip().str.lower()
    tissue = meta.tissue_general.astype("string").str.lower()
    frames = {}
    for group, value in (("disease", DISEASE), ("normal", "normal")):
        in_group = disease.eq(value).fillna(False)
        for organ in ORGANS:
            mask = in_group & tissue.eq(organ).fillna(False) if organ else in_group
            frame = meta[mask]
            for rule in DONOR_RULES:
                kept = frame[_usable_donor_mask(frame.donor_id)] if rule == "known_donor" else frame
                frames[(group, organ, rule)] = kept.drop_duplicates(POINTER)
    return frames


def label_sets(frame, column, keywords):
    """Map each keyword-matching label (lower case) to its metacells and its spellings."""
    lower = frame[column].astype("string").str.lower()
    hit = lower.str.contains("|".join(keywords), regex=True).fillna(False)
    sets, spellings = {}, {}
    for label, sub in frame[hit].groupby(lower[hit]):
        sets[label] = frozenset(sub.pointer)
        spellings[label] = " | ".join(sorted(sub[column].astype(str).unique()))
    return sets, spellings


def candidates(sets, target, max_union):
    """Yield (labels, metacells): every single label, plus unions that can hit the target."""
    labels = sorted(sets)
    for label in labels:
        yield (label,), sets[label]
    for k in range(2, max_union + 1):
        for combo in itertools.combinations(labels, k):
            sizes = [len(sets[label]) for label in combo]
            # A union is at least its largest member and at most the sum of its members.
            if max(sizes) > target or sum(sizes) < target:
                continue
            yield combo, frozenset().union(*(sets[label] for label in combo))


def handoff_cells(handoff, folder):
    labels = pd.read_csv(os.path.join(handoff, folder, "metadata/labels.csv"), encoding="utf-8-sig")
    pointer = labels.matrix_file_path.astype(str) + "#" + labels.matrix_row_idx.astype(str)
    is_normal = labels.disease_BMG_name.astype(str).str.strip().str.lower().eq("normal")
    return {"disease": frozenset(pointer[~is_normal]), "normal": frozenset(pointer[is_normal])}


def search(overview, frames, handoff, max_union):
    rows = []
    for _, cohort in overview.iterrows():
        targets = {"disease": int(cohort.disease_metacells), "normal": int(cohort.normal_metacells)}
        truth = handoff_cells(handoff, cohort.result_folder)
        for (group, organ, rule), frame in frames.items():
            for column in COLUMNS:
                sets, spellings = label_sets(frame, column, KEYWORDS[cohort.cell_type])
                for combo, cells in candidates(sets, targets[group], max_union):
                    count_match = len(cells) == targets[group]
                    if len(combo) > 1 and not count_match:
                        continue
                    rows.append({
                        "cohort": cohort.cell_type, "group": group, "target": targets[group],
                        "column": column, "organ": organ or "any", "donor_rule": rule,
                        "n_labels": len(combo), "labels": " + ".join(spellings[l] for l in combo),
                        "count": len(cells), "count_match": count_match,
                        "same_cells_as_handoff": cells == truth[group],
                        "agent_reachable": column == "CMT_name" and len(combo) == 1 and rule == "known_donor",
                    })
    return pd.DataFrame(rows)


def report(results, overview):
    for _, cohort in overview.iterrows():
        name = cohort.cell_type
        res = results[results.cohort == name]
        print(f"\n{'=' * 78}\nPDAC {name}: target disease {cohort.disease_metacells} / "
              f"normal {cohort.normal_metacells}\n  overview: {cohort.actual_comparison}")
        for group in ("disease", "normal"):
            hits = res[(res.group == group) & res.count_match]
            exact = hits[hits.same_cells_as_handoff]
            print(f"  [{group}] {len(hits)} settings match the count; "
                  f"{len(exact)} select the handoff's exact metacells; "
                  f"{len(hits) - len(exact)} are coincidences")
            for _, hit in exact.iterrows():
                print(f"      EXACT  {hit.column:9} organ={hit.organ:8} donor={hit.donor_rule:12} "
                      f"{hit.labels}")
        # The agent applies one CMT_name label to both groups, keeping known donors only.
        agent = res[res.agent_reachable]
        both = (agent.groupby(["organ", "labels"])
                .filter(lambda g: set(g[g.count_match].group) == {"disease", "normal"}))
        verdict = ", ".join(sorted(set(both.labels))) or "none"
        print(f"  Agent-reachable (one CMT_name label, known donors only) matching both counts: {verdict}")
        for label in sorted(set(res[res.same_cells_as_handoff & (res.column == "CMT_name")].labels)):
            under_agent = agent[(agent.labels == label) & (agent.organ == "pancreas")]
            counts = {r.group: r["count"] for _, r in under_agent.iterrows()}
            print(f"      under the agent's donor rule, {label!r} gives disease "
                  f"{counts.get('disease', 0)} / normal {counts.get('normal', 0)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--handoff", default=os.path.join(
        get_path("sessions.base", absolute=True), "case_study_handoff_20260923"))
    parser.add_argument("--max-union", type=int, default=2,
                        help="largest number of labels pooled into one group (default 2)")
    parser.add_argument("--out", default=os.path.join(
        get_path("sessions.base", absolute=True), "pdac_cohort_search", "results.csv"))
    args = parser.parse_args()

    overview = pd.read_csv(os.path.join(args.handoff, "00_overview.csv"), encoding="utf-8-sig")
    overview = overview[overview.disease == "PDAC"]
    results = search(overview, retrieval_frames(load_metadata()), args.handoff, args.max_union)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    results.to_csv(args.out, index=False)
    report(results, overview)
    print(f"\nAll {len(results):,} evaluated settings: {args.out}")


if __name__ == "__main__":
    main()

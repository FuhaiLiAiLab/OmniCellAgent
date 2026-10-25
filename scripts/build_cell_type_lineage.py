"""Write tools/omic_tools/cell_type_lineage.tsv: one Cell Ontology lineage heading per CMT_name.

Each cell type goes under the nearest of a fixed list of broad Cell Ontology classes
(fewest is_a steps from any of its linked terms); on a tie the class listed first wins.
Names with no linked term, or none of the classes among their ancestors, go under
"Other". The headings only organise the list the omic agent sees
(tools/omic_tools/cell_type_lookup.py); they do not restrict its choice.

Inputs: the name-extraction benchmark vocabulary (Cell Ontology links per CMT_name)
and the pinned Cell Ontology release it was built with, both checked as the benchmark
does. Run from the project root:

    python scripts/build_cell_type_lineage.py [--output PATH] [--ties]
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from tools.omic_tools.tests.name_extraction.build_vocabulary import read_tsv, write_tsv
from tools.omic_tools.tests.name_extraction.obo import Ontology
from tools.omic_tools.tests.name_extraction.sources import fetch
from tools.omic_tools.tests.name_extraction.text import norm_basic

VOCABULARY = ROOT / "tools/omic_tools/tests/name_extraction/vocabulary/cell_types.tsv"
OUTPUT = ROOT / "tools/omic_tools/cell_type_lineage.tsv"
OTHER = "Other"
# Broad Cell Ontology classes, more specific first: on a tie the earlier class wins.
ANCHORS = [
    "T cell", "B cell", "natural killer cell", "innate lymphoid cell", "macrophage",
    "monocyte", "dendritic cell", "mast cell", "granulocyte", "megakaryocyte",
    "erythroid lineage cell", "hematopoietic precursor cell", "endothelial cell",
    # Before "neuron", so (neuro)endocrine cells, tied between the two, are epithelial
    "epithelial cell", "neuron", "glial cell", "fibroblast", "pericyte", "muscle cell",
    "adipocyte", "pigment cell", "germ cell", "stem cell", "stromal cell",
    "connective tissue cell", "neural cell", "leukocyte",
]


def anchor_ids(cl):
    labels = cl.label_index()
    ids = {}
    for name in ANCHORS:
        matches = labels.get(norm_basic(name), set())
        if len(matches) != 1:
            raise ValueError(f"Anchor {name!r} matches {len(matches)} Cell Ontology terms")
        ids[next(iter(matches))] = name
    return ids


def lineage_table(vocabulary, cl):
    """One row per CMT_name: heading, and the anchors tied for nearest (for review)."""
    anchors = anchor_ids(cl)
    rank = {name: i for i, name in enumerate(ANCHORS)}
    rows = []
    for record in vocabulary.itertuples(index=False):
        steps = {}
        for term in filter(None, record.cl_ids.split(" | ")):
            distances = cl.ancestors(term)
            distances[term] = 0
            for anchor, name in anchors.items():
                if anchor in distances:
                    steps[name] = min(steps.get(name, distances[anchor]), distances[anchor])
        if not steps:
            rows.append({"cell_type": record.db_name, "heading": OTHER, "tied": ""})
            continue
        nearest = min(steps.values())
        tied = sorted((name for name, d in steps.items() if d == nearest), key=rank.get)
        rows.append({"cell_type": record.db_name, "heading": tied[0],
                     "tied": " | ".join(tied) if len(tied) > 1 else ""})
    return pd.DataFrame(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--ties", action="store_true", help="print names with tied nearest classes")
    args = parser.parse_args(argv)

    vocabulary = read_tsv(VOCABULARY)
    cl = Ontology.load(fetch("cell_ontology"), prefix="CL:")
    table = lineage_table(vocabulary, cl)
    if table.cell_type.duplicated().any():
        raise ValueError("Duplicate cell type in the vocabulary")
    write_tsv(table[["cell_type", "heading"]], args.output)
    print(f"Wrote {len(table)} cell types under {table.heading.nunique()} headings to {args.output}")
    print(table.heading.value_counts().to_string())
    if args.ties:
        tied = table[table.tied != ""]
        print(f"\n{len(tied)} names with tied nearest classes (heading = first listed):")
        for record in tied.itertuples(index=False):
            print(f"  {record.cell_type!r}: {record.tied}")


if __name__ == "__main__":
    main()

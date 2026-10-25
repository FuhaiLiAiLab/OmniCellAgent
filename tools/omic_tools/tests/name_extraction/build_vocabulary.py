"""Step 1: freeze the OmniCellTOSG query vocabulary.

Writes one table per query field (cell type, disease, organ) with the exact
database strings the omic tool filters on, how much data sits behind each,
and links to standard ontology identifiers:

* cell types -> Cell Ontology, by exact label of each component head
  ("B cell;B cell:activated" has head "B cell"), then by EXACT synonym, then
  by the label of an obsolete term that names a replacement;
* diseases -> UMLS / MONDO / MeSH / DOID / OMIM through the BioMedGraphica
  disease table, then MONDO equivalence xrefs to UMLS, then an exact MONDO
  label or EXACT-synonym match on the database name.

Usage (from the project root):
    python -m tools.omic_tools.tests.name_extraction.build_vocabulary
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from tools.omic_tools.cohort_diagnostics import _usable_donor_mask, donor_keys  # noqa: E402
from tools.omic_tools.tests.name_extraction import sources  # noqa: E402
from tools.omic_tools.tests.name_extraction.obo import Ontology  # noqa: E402
from tools.omic_tools.tests.name_extraction.text import norm_basic  # noqa: E402

DEFAULT_OUT = HERE / "vocabulary"
LIST_SEP = " | "
CONTROL_LIKE = {"normal", "unknown", "unclassified", "unannotated", "unannoted"}
METADATA_COLUMNS = ["source", "dataset_id", "donor_id", "CMT_id", "CMT_name", "cell_type",
                    "disease_BMG_id", "disease_BMG_name", "disease", "tissue_general"]
BMG_ID_COLUMNS = {"umls_ids": "UMLS_ID", "mondo_ids": "MONDO_ID", "mesh_ids": "MeSH_ID",
                  "doid_ids": "DO_ID", "omim_ids": "OMIM_ID"}
_QUALIFIER_SUFFIX_RE = re.compile(r"\s*\((?:disorder|disease|finding)\)\s*$", re.I)


def _join(values) -> str:
    return LIST_SEP.join(str(value) for value in values)


def _unique(values) -> list:
    return list(dict.fromkeys(values))


def _split_ids(value) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    return [part.strip() for part in str(value).split(";") if part.strip()]


def value_counts(metadata: pd.DataFrame, column: str) -> pd.DataFrame:
    """Rows, usable study-scoped donors, and datasets behind each value."""
    usable = pd.Series(_usable_donor_mask(metadata["donor_id"]), index=metadata.index)
    donors = donor_keys(metadata).where(usable)
    datasets = metadata["source"].astype(str) + "/" + metadata["dataset_id"].astype(str)
    frame = pd.DataFrame({"value": metadata[column].fillna("").astype(str),
                          "donor": donors, "dataset": datasets})
    grouped = frame.groupby("value", sort=False)
    return pd.DataFrame({"n_rows": grouped.size(),
                         "n_donors": grouped["donor"].nunique(),
                         "n_datasets": grouped["dataset"].nunique()})


def raw_label_summary(metadata: pd.DataFrame, name_column: str, raw_column: str,
                      top: int = 8) -> dict[str, str]:
    """Most frequent source-study labels mapped to each database name."""
    if raw_column not in metadata.columns:
        return {}
    counts = (metadata.groupby([name_column, raw_column], sort=False).size()
              .reset_index(name="n").sort_values(["n", raw_column], ascending=[False, True]))
    summary = {}
    for name, group in counts.groupby(name_column, sort=False):
        summary[name] = _join(f"{raw} ({n})" for raw, n in zip(group[raw_column].head(top),
                                                              group["n"].head(top)))
    return summary


def _obsolete_label_index(ontology: Ontology) -> dict[str, set[str]]:
    index: dict[str, set[str]] = defaultdict(set)
    for term in ontology.terms.values():
        if term.is_obsolete and term.name:
            live = ontology.current_id(term.id)
            if live:
                index[norm_basic(re.sub(r"^obsolete\s+", "", term.name, flags=re.I))].add(live)
    return index


def build_cell_type_table(metadata: pd.DataFrame, cl: Ontology) -> tuple[pd.DataFrame, dict]:
    names = metadata["CMT_name"].fillna("").astype(str)
    named = metadata.loc[names.str.strip().ne("")]
    counts = value_counts(named, "CMT_name")
    cmt_ids = named.groupby("CMT_name")["CMT_id"].agg(
        lambda s: sorted({str(v).strip() for v in s.dropna() if str(v).strip()}))
    raw = raw_label_summary(named, "CMT_name", "cell_type")
    labels, exact = cl.label_index(), cl.synonym_index(("EXACT",))
    obsolete = _obsolete_label_index(cl)

    rows = []
    for name in sorted(counts.index, key=lambda value: (value.casefold(), value)):
        components = [part.strip() for part in name.split(";") if part.strip()]
        heads = _unique(part.split(":", 1)[0].strip() for part in components)
        cl_ids, links = [], []
        for head in heads:
            key = norm_basic(head)
            for index, how in ((labels, "label"), (exact, "exact_synonym"),
                               (obsolete, "obsolete_label")):
                if key in index:
                    matched = sorted(index[key])
                    cl_ids.extend(matched)
                    links.append(how if len(matched) == 1 else f"{how}_ambiguous")
                    break
            else:
                links.append("none")
        cl_ids = _unique(cl_ids)
        rows.append({
            "db_name": name,
            "cmt_id": _join(cmt_ids.get(name, [])),
            "n_rows": int(counts.at[name, "n_rows"]),
            "n_donors": int(counts.at[name, "n_donors"]),
            "n_datasets": int(counts.at[name, "n_datasets"]),
            "is_composite": ";" in name or ":" in name,
            "components": _join(components),
            "heads": _join(heads),
            "cl_ids": _join(cl_ids),
            "cl_labels": _join(cl.label(term_id) for term_id in cl_ids),
            "cl_link": _join(links),
            "raw_labels_unaudited": raw.get(name, ""),
        })
    table = pd.DataFrame(rows)
    stats = {
        "cell_types": len(table),
        "rows_without_cell_type": int(len(metadata) - len(named)),
        "composite_names": int(table["is_composite"].sum()),
        "linked_to_cl": int(table["cl_ids"].ne("").sum()),
        "cl_link_methods": dict(pd.Series(
            [how for value in table["cl_link"] for how in value.split(LIST_SEP)]).value_counts()),
    }
    return table, stats


def build_disease_table(metadata: pd.DataFrame, bmg: pd.DataFrame,
                        mondo: Ontology) -> tuple[pd.DataFrame, dict]:
    names = metadata["disease_BMG_name"].fillna("").astype(str)
    named = metadata.loc[names.str.strip().ne("")]
    counts = value_counts(named, "disease_BMG_name")
    bmg_ids = named.groupby("disease_BMG_name")["disease_BMG_id"].agg(
        lambda s: sorted({str(v).strip() for v in s.dropna() if str(v).strip()}))
    raw = raw_label_summary(named, "disease_BMG_name", "disease")
    bmg_rows = {row["BioMedGraphica_ID"]: row for row in bmg.to_dict("records")}
    mondo_by_umls = mondo.xref_index("UMLS:", require_source="MONDO:equivalentTo")
    labels, exact = mondo.label_index(), mondo.synonym_index(("EXACT",))

    rows = []
    for name in sorted(counts.index, key=lambda value: (value.casefold(), value)):
        control_like = norm_basic(name) in CONTROL_LIKE
        ids = bmg_ids.get(name, [])
        linked = {column: [] for column in BMG_ID_COLUMNS}
        crosswalk = []
        for bmg_id in ids:
            row = bmg_rows.get(bmg_id)
            if row is None:
                continue
            crosswalk.append("biomedgraphica")
            for column, source_column in BMG_ID_COLUMNS.items():
                linked[column].extend(_split_ids(row.get(source_column)))
        linked = {column: _unique(values) for column, values in linked.items()}

        mondo_live, mondo_dropped = [], []
        for term_id in linked["mondo_ids"]:
            live = mondo.current_id(term_id)
            if live:
                mondo_live.append(live)
            else:
                mondo_dropped.append(term_id)
        if not control_like:
            via_umls = sorted({m for cui in linked["umls_ids"] for m in mondo_by_umls.get(cui, ())}
                              - set(mondo_live))
            if via_umls:
                mondo_live.extend(via_umls)
                crosswalk.append("mondo_umls_xref")
            if not mondo_live:
                key = norm_basic(_QUALIFIER_SUFFIX_RE.sub("", name))
                for index, how in ((labels, "mondo_label"), (exact, "mondo_exact_synonym")):
                    if key in index:
                        mondo_live.extend(sorted(index[key]))
                        crosswalk.append(how)
                        break
        mondo_live = _unique(mondo_live)
        rows.append({
            "db_name": name,
            "bmg_ids": _join(ids),
            "n_rows": int(counts.at[name, "n_rows"]),
            "n_donors": int(counts.at[name, "n_donors"]),
            "n_datasets": int(counts.at[name, "n_datasets"]),
            "is_control_like": control_like,
            "umls_ids": _join(linked["umls_ids"]),
            "mondo_ids": _join(mondo_live),
            "mondo_labels": _join(mondo.label(term_id) for term_id in mondo_live),
            "mondo_ids_unresolved": _join(mondo_dropped),
            "mesh_ids": _join(linked["mesh_ids"]),
            "doid_ids": _join(linked["doid_ids"]),
            "omim_ids": _join(linked["omim_ids"]),
            "crosswalk": _join(_unique(crosswalk)) or "none",
            "raw_labels_unaudited": raw.get(name, ""),
        })
    table = pd.DataFrame(rows)
    disease_rows = table.loc[~table["is_control_like"]]
    stats = {
        "diseases": len(table),
        "control_like": int(table["is_control_like"].sum()),
        "rows_without_disease": int(len(metadata) - len(named)),
        "bmg_ids_missing_from_biomedgraphica": sorted(
            {bmg_id for ids in bmg_ids for bmg_id in ids if bmg_id not in bmg_rows}),
        "with_umls": int(disease_rows["umls_ids"].ne("").sum()),
        "with_mondo": int(disease_rows["mondo_ids"].ne("").sum()),
        "with_mesh": int(disease_rows["mesh_ids"].ne("").sum()),
        "without_any_ontology_link": sorted(disease_rows.loc[
            disease_rows["umls_ids"].eq("") & disease_rows["mondo_ids"].eq(""), "db_name"]),
    }
    return table, stats


def build_organ_table(metadata: pd.DataFrame) -> pd.DataFrame:
    counts = value_counts(metadata, "tissue_general")
    counts = counts.loc[[value for value in counts.index if value.strip()]]
    table = counts.reset_index().rename(columns={"value": "db_name"})
    return table.sort_values("db_name", key=lambda s: s.str.casefold()).reset_index(drop=True)


def write_tsv(table: pd.DataFrame, path: Path) -> None:
    """Plain TSV: no quoting, so every value must be free of tabs and newlines."""
    lines = ["\t".join(table.columns)]
    for record in table.itertuples(index=False):
        cells = ["" if value is None else str(value) for value in record]
        bad = [cell for cell in cells if any(ch in cell for ch in "\t\r\n")]
        if bad:
            raise ValueError(f"Value with tab/newline cannot be written to {path.name}: {bad[0]!r}")
        lines.append("\t".join(cells))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_tsv(path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=3)


def git_state(root: Path) -> dict:
    def run(*args):
        result = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    return {"commit": run("rev-parse", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD")}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--metadata", type=Path, default=None,
                        help="cell_metadata_with_mappings.parquet (default: from configs/paths.yaml)")
    parser.add_argument("--biomedgraphica-disease", type=Path, default=None)
    args = parser.parse_args(argv)

    from utils.path_config import get_path
    metadata_path = args.metadata or Path(get_path("external.omnicell_data_root", absolute=True)) \
        / "cell_metadata_with_mappings.parquet"
    bmg_path = args.biomedgraphica_disease or Path(get_path("external.biomedgraphica_dir", absolute=True)) \
        / "Entity" / "Disease" / "BioMedGraphica_Conn_Disease.csv"

    metadata = pd.read_parquet(metadata_path, columns=METADATA_COLUMNS)
    bmg = pd.read_csv(bmg_path, dtype=str)
    cl = Ontology.load(sources.fetch("cell_ontology", args.cache_dir), prefix="CL:")
    mondo = Ontology.load(sources.fetch("mondo", args.cache_dir), prefix="MONDO:")

    cell_types, cell_stats = build_cell_type_table(metadata, cl)
    diseases, disease_stats = build_disease_table(metadata, bmg, mondo)
    organs = build_organ_table(metadata)

    args.out.mkdir(parents=True, exist_ok=True)
    write_tsv(cell_types, args.out / "cell_types.tsv")
    write_tsv(diseases, args.out / "diseases.tsv")
    write_tsv(organs, args.out / "organs.tsv")
    manifest = {
        "step": "1 - frozen query vocabulary",
        "created_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "generator": "tools/omic_tools/tests/name_extraction/build_vocabulary.py",
        "git": git_state(PROJECT_ROOT),
        "inputs": {
            "metadata": {"path": str(metadata_path), "sha256": sources.sha256_file(metadata_path),
                         "rows": int(len(metadata))},
            "biomedgraphica_disease": {"path": str(bmg_path), "sha256": sources.sha256_file(bmg_path)},
            "ontologies": {"cell_ontology": cl.data_version, "mondo": mondo.data_version},
            "sources": sources.describe(["cell_ontology", "mondo"]),
        },
        "outputs": {name: sources.sha256_file(args.out / name)
                    for name in ("cell_types.tsv", "diseases.tsv", "organs.tsv")},
        "counts": {"cell_type": cell_stats, "disease": disease_stats, "organ": {"organs": len(organs)}},
        "notes": [
            "Counts are metadata rows (SEACells metacells), usable study-scoped donors "
            "(source, dataset_id, donor_id), and source/dataset pairs.",
            "raw_labels_unaudited lists source-study labels mapped to each name; the mapping "
            "contains known errors and must not be used as gold synonyms without review.",
        ],
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=int) + "\n",
                                            encoding="utf-8")
    print(json.dumps(manifest["counts"], indent=2, default=int))


if __name__ == "__main__":
    main()

"""Runtime isolation and HGNC protein-coding selection for CellTOSG queries."""

import csv
from contextlib import contextmanager
from os import PathLike
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterator


HGNC_PROTEIN_CODING_REFERENCE = (
    Path(__file__).resolve().parent / "reference" / "hgnc_protein_coding.tsv"
)


CELLTOSG_READ_INPUTS = (
    "cell_metadata_with_mappings.parquet",
    "bmg_gene_index.csv",
    "expression_matrix",
    "edge_index.npy",
    "internal_edge_index.npy",
    "ppi_edge_index.npy",
    "x_name_emb.npy",
    "x_desc_emb.npy",
    "x_bio_emb.npy",
)


def _filter_bmg_gene_index(source: Path, destination: Path, reference: Path) -> None:
    """Copy approved protein-coding rows, preserving BMG order and indices.

    Match HGNC's current symbol exactly; aliases and previous symbols can be
    ambiguous and are not substituted. The reference is a local HGNC snapshot.
    """
    with reference.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not {"symbol", "status", "locus_group"} <= set(reader.fieldnames or []):
            raise ValueError(
                f"HGNC reference {reference} must have columns: symbol, status, locus_group"
            )
        protein_coding = {
            row["symbol"] for row in reader
            if row["status"] == "Approved"
            and row["locus_group"] == "protein-coding gene"
            and row["symbol"]
        }
    if not protein_coding:
        raise ValueError(f"HGNC reference {reference} has no approved protein-coding symbols")

    with source.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"gene_name", "indices"} <= set(reader.fieldnames or []):
            raise ValueError(f"BMG mapping {source} must have columns: gene_name, indices")
        fieldnames = reader.fieldnames
        rows = list(reader)
    retained = [row for row in rows if row["gene_name"].strip() in protein_coding]
    if not retained:
        raise ValueError(f"No BMG genes match the approved protein-coding symbols in {reference}")

    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(retained)
    print(
        f"[HGNC] Protein-coding filter: kept {len(retained):,} / {len(rows):,} genes; "
        f"removed {len(rows) - len(retained):,} (exact approved-symbol match)."
    )


@contextmanager
def writable_celltosg_root(source_root: str | PathLike[str], *, metadata=None) -> Iterator[Path]:
    """Yield a temporary loader root with an HGNC-filtered gene mapping.

    All other inputs remain read-only by convention. Filtering the mapping
    before extraction keeps dataset.data, expression_gene.csv and the gene
    choice sidecar aligned, before CP10K normalization and downstream analysis.
    If metadata is supplied, write that preselected cohort into the temporary
    root instead of linking the full metadata table.
    """
    source_root = Path(source_root).expanduser().resolve()
    missing = [
        name for name in CELLTOSG_READ_INPUTS if not (source_root / name).exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"CellTOSG source root {source_root} is missing required inputs: "
            f"{', '.join(missing)}"
        )

    temporary_root = TemporaryDirectory(prefix="celltosg-shadow-")
    shadow_root = Path(temporary_root.name)
    try:
        for name in CELLTOSG_READ_INPUTS:
            source = source_root / name
            if name == "cell_metadata_with_mappings.parquet" and metadata is not None:
                metadata.to_parquet(shadow_root / name, index=False)
                continue
            if name == "bmg_gene_index.csv":
                _filter_bmg_gene_index(
                    source, shadow_root / name, HGNC_PROTEIN_CODING_REFERENCE
                )
                continue
            (shadow_root / name).symlink_to(
                source, target_is_directory=source.is_dir()
            )
        yield shadow_root
    finally:
        temporary_root.cleanup()

"""CellTOSG raw retrieval with cohort-wide representative-gene selection."""

import csv
import hashlib
import importlib
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from .aggregation import validate_raw_counts


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def filter_gene_mapping(mapping_path, reference_path):
    reference = pd.read_csv(reference_path, sep="\t", dtype=str, keep_default_na=False)
    if not {"symbol", "status", "locus_group"} <= set(reference.columns):
        raise ValueError("HGNC reference must contain symbol, status and locus_group")
    allowed = set(reference.loc[reference.status.eq("Approved") &
                                reference.locus_group.eq("protein-coding gene"), "symbol"])
    if not allowed:
        raise ValueError("HGNC reference has no approved protein-coding genes")
    with Path(mapping_path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"gene_name", "indices"} <= set(reader.fieldnames or []):
            raise ValueError("BMG mapping requires gene_name and indices")
        all_rows = list(reader)
    metacell_metadata = []
    for row in all_rows:
        name = row["gene_name"].strip()
        if name in allowed:
            indices = [int(value.strip()) for value in row["indices"].split(";") if value.strip()]
            if not indices or min(indices) < 0:
                raise ValueError(f"Invalid BMG candidate indices for {name}")
            metacell_metadata.append((name, indices))
    if not metacell_metadata or len({name for name, _ in metacell_metadata}) != len(metacell_metadata):
        raise ValueError("HGNC-filtered BMG mapping is empty or has duplicate gene names")
    return metacell_metadata, {"bmg_genes": len(all_rows), "reference_protein_coding_genes": len(allowed),
                      "protein_coding_genes": len(metacell_metadata), "excluded_bmg_genes": len(all_rows) - len(metacell_metadata),
                      "hgnc_sha256": file_sha256(reference_path), "bmg_mapping_sha256": file_sha256(mapping_path)}


def _celltosg_api(celltosg_root):
    data_directory = Path(celltosg_root).resolve()
    sys.path.insert(0, str(data_directory))
    api = importlib.import_module("CellTOSG_Loader.data_loader")
    if not Path(api.__file__).resolve().is_relative_to(data_directory):
        raise RuntimeError(f"CellTOSG imported from unexpected path: {api.__file__}")
    return api


def load_raw_gene_counts(metadata, data_root, celltosg_root, hgnc_reference,
                         output_dir, chunk_size=64):
    """Use CellTOSG readers in chunks and collapse genes once over all rows.

    Only candidate columns are retained between chunks. Original BMG indices
    are translated to compact coordinates for CellTOSG's collapse function and
    translated back in the exported choice table. No normalization is applied.
    """
    if chunk_size < 1 or metadata.empty:
        raise ValueError("Raw retrieval requires metacells and a positive chunk size")
    data_directory = Path(data_root).resolve()
    output_directory = Path(output_dir)
    output_directory.mkdir(parents=True, exist_ok=False)
    gene_mapping, audit = filter_gene_mapping(data_directory / "bmg_gene_index.csv", hgnc_reference)
    candidate_columns = sorted({index for _, indices in gene_mapping for index in indices})
    compact_column_index = {original: i for i, original in enumerate(candidate_columns)}
    for name, rows in [("bmg_gene_index_protein_coding.csv", gene_mapping),
                       ("compact_bmg_gene_index.csv", [(gene, [compact_column_index[i] for i in indices])
                                                       for gene, indices in gene_mapping])]:
        with (output_directory / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["gene_name", "indices"])
            writer.writerows((gene, ";".join(map(str, indices))) for gene, indices in rows)

    metacell_metadata = metadata.copy().reset_index(drop=True)
    metacell_metadata["sample_index"] = np.arange(len(metacell_metadata))
    row_indices = pd.to_numeric(metacell_metadata.matrix_row_idx, errors="raise")
    if row_indices.isna().any() or (row_indices < 0).any() or not np.equal(row_indices, np.floor(row_indices)).all():
        raise ValueError("Metacell matrix row indices must be nonnegative integers")
    metacell_metadata["matrix_row_idx"] = row_indices.astype(np.int64)
    matrix_root = (data_directory / "expression_matrix").resolve()
    shapes = {}
    for filename, group in metacell_metadata.groupby("matrix_file_path", sort=False):
        path = (matrix_root / filename).resolve()
        if not path.is_relative_to(matrix_root):
            raise ValueError(f"Matrix pointer escapes expression directory: {filename}")
        matrix = np.load(path, mmap_mode="r", allow_pickle=False)
        if matrix.ndim != 2 or matrix.shape[1] <= max(candidate_columns) or group.matrix_row_idx.max() >= matrix.shape[0]:
            raise ValueError(f"BMG or row indices exceed source matrix dimensions: {filename}")
        shapes[filename] = list(matrix.shape)
        del matrix
    if len({shape[1] for shape in shapes.values()}) != 1:
        raise ValueError("Source expression matrices have inconsistent feature axes")

    api = _celltosg_api(celltosg_root)
    candidate_counts = np.empty((len(metacell_metadata), len(candidate_columns)), dtype=np.int64)
    for start in range(0, len(metacell_metadata), chunk_size):
        end = min(start + chunk_size, len(metacell_metadata))
        block = api.load_expression_by_metadata(metacell_metadata.iloc[start:end], dataset_dir=str(matrix_root))
        values = block[:, candidate_columns]
        validate_raw_counts(values)
        candidate_counts[start:end] = values.astype(np.int64)
        del block, values
        print(f"[Raw retrieval] {end:,}/{len(metacell_metadata):,} metacells; no normalization", flush=True)

    gene_counts, gene_names, compact_choice = api.bmg_matrix_to_gene_matrix(
        candidate_counts, str(output_directory / "compact_bmg_gene_index.csv"), len(candidate_columns)
    )
    chosen_compact = compact_choice.chosen_bmg_index.to_numpy(dtype=int)
    original_candidates = dict(gene_mapping)
    choices = pd.DataFrame({
        "gene_name": gene_names,
        "chosen_bmg_index": [candidate_columns[index] for index in chosen_compact],
        "candidate_bmg_indices": [";".join(map(str, original_candidates[gene])) for gene in gene_names],
        "chosen_compact_column": chosen_compact,
        "choice_reason_compact_coordinates": compact_choice.choice_reason.tolist(),
    })
    choices.to_csv(output_directory / "bmg_to_gene_choice.csv", index=False)
    audit.update({"retrieved_metacells": len(metacell_metadata), "candidate_columns": len(candidate_columns),
                  "source_matrices": shapes, "collapse_scope": "entire_selected_cohort",
                  "normalization": "none", "count_dtype": str(gene_counts.dtype),
                  "celltosg_module": str(Path(api.__file__).resolve()),
                  "celltosg_module_sha256": file_sha256(api.__file__)})
    return gene_counts, gene_names, choices, audit

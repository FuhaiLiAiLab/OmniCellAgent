"""Integer-count validation and donor pseudobulk aggregation, before normalization."""

import numpy as np
import pandas as pd


def validate_raw_counts(counts):
    """Reject normalization, invalid values, and unsafe integer-sum ranges."""
    raw_count_values = np.asarray(counts)
    if raw_count_values.ndim != 2 or not np.issubdtype(raw_count_values.dtype, np.number):
        raise ValueError("Raw counts must be a numeric two-dimensional matrix")
    if not np.isfinite(raw_count_values).all() or np.any(raw_count_values < 0):
        raise ValueError("Raw counts must be finite and nonnegative")
    if not np.equal(raw_count_values, np.floor(raw_count_values)).all():
        raise ValueError("Raw integer counts are required; normalized or fractional inputs are not accepted")
    if raw_count_values.size and raw_count_values.max() > np.iinfo(np.int64).max // max(raw_count_values.size, 1):
        raise ValueError("Raw counts exceed the supported exact integer-sum range")


def aggregate_raw_counts(counts, gene_names, assignments, donor_metadata):
    """Sum metacells × genes into genes × donors, conserving each gene total.

    assignments rows must follow the input count rows. Output columns follow
    donor_metadata.sample_id; a column is an individual, not a metacell.
    """
    raw_count_values = np.asarray(counts)
    validate_raw_counts(raw_count_values)
    gene_symbols = list(gene_names)
    if len(gene_symbols) != raw_count_values.shape[1] or len(set(gene_symbols)) != len(gene_symbols):
        raise ValueError("Gene names must uniquely identify every count column")
    if len(assignments) != raw_count_values.shape[0]:
        raise ValueError("Metacell assignments do not align with count rows")
    donor_ids = donor_metadata.sample_id.tolist()
    if len(set(donor_ids)) != len(donor_ids) or set(assignments.sample_id) != set(donor_ids):
        raise ValueError("Assignment donor IDs do not match donor metadata")
    donor_count_sums = np.zeros((len(donor_ids), len(gene_symbols)), dtype=np.int64)
    for donor_index, donor in enumerate(donor_ids):
        indices = np.flatnonzero(assignments.sample_id.to_numpy() == donor)
        donor_count_sums[donor_index] = np.sum(raw_count_values[indices], axis=0, dtype=np.int64)
    input_gene_totals = np.sum(raw_count_values, axis=0, dtype=np.int64)
    donor_gene_totals = np.sum(donor_count_sums, axis=0, dtype=np.int64)
    if not np.array_equal(input_gene_totals, donor_gene_totals):
        raise AssertionError("Per-gene raw-count conservation failed")
    pseudobulk_counts = pd.DataFrame(donor_count_sums.T, index=pd.Index(gene_symbols, name="gene"), columns=donor_ids)
    report = {"metacells": len(assignments), "donors": len(donor_ids), "genes": len(gene_symbols),
              "input_total_counts": int(input_gene_totals.sum()), "pseudobulk_total_counts": int(donor_gene_totals.sum()),
              "per_gene_count_conservation": True, "normalization_before_aggregation": "none"}
    return pseudobulk_counts, report

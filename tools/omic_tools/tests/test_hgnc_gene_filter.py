"""HGNC filtering must preserve expression-column indices and symbol order."""

import csv

import pytest

from tools.omic_tools import celltosg_runtime_adapter as adapter


def test_filter_keeps_only_exact_approved_protein_coding_symbols(tmp_path):
    reference = tmp_path / "hgnc.txt"
    reference.write_text(
        "symbol\tstatus\tlocus_group\talias_symbol\n"
        "ARF5\tApproved\tprotein-coding gene\tOLD_ARF5\n"
        "M6PR\tApproved\tprotein-coding gene\t\n"
        "MALAT1\tApproved\tnon-coding RNA\t\n"
        "WITHDRAWN\tEntry Withdrawn\tprotein-coding gene\t\n"
        "IGHV1-2\tApproved\tother\t\n"
    )
    source = tmp_path / "source.csv"
    source.write_text(
        "gene_name,indices\nMALAT1,12\nM6PR,9;41\nARF5,6\n"
        "WITHDRAWN,8\nOLD_ARF5,3\nUNKNOWN,4\nIGHV1-2,2\narf5,1\n"
    )
    destination = tmp_path / "filtered.csv"

    adapter._filter_bmg_gene_index(source, destination, reference)

    with destination.open() as handle:
        assert list(csv.DictReader(handle)) == [
            {"gene_name": "M6PR", "indices": "9;41"},
            {"gene_name": "ARF5", "indices": "6"},
        ]
    assert "MALAT1" in source.read_text()


@pytest.mark.parametrize(
    "reference_text, expected_error",
    [
        ("symbol\nARF5\n", "columns"),
        ("symbol\tstatus\tlocus_group\nMALAT1\tApproved\tnon-coding RNA\n", "protein-coding"),
        ("symbol\tstatus\tlocus_group\nOTHER\tApproved\tprotein-coding gene\n", "No BMG"),
    ],
)
def test_invalid_or_nonmatching_reference_fails_closed(tmp_path, reference_text, expected_error):
    reference = tmp_path / "hgnc.txt"
    reference.write_text(reference_text)
    source = tmp_path / "source.csv"
    source.write_text("gene_name,indices\nARF5,6\n")
    destination = tmp_path / "filtered.csv"

    with pytest.raises(ValueError, match=expected_error):
        adapter._filter_bmg_gene_index(source, destination, reference)
    assert not destination.exists()


def test_bundled_reference_contains_only_the_supplied_protein_coding_set():
    with adapter.HGNC_PROTEIN_CODING_REFERENCE.open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert len(rows) == 19297
    assert len({row["symbol"] for row in rows}) == 19297
    assert all(row["status"] == "Approved" for row in rows)
    assert all(row["locus_group"] == "protein-coding gene" for row in rows)
    symbols = {row["symbol"] for row in rows}
    assert {"ARF5", "M6PR", "TP53"} <= symbols
    assert not {"MALAT1", "XIST", "IGHV1-2"} & symbols

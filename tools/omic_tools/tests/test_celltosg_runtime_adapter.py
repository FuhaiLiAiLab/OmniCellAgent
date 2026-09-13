from pathlib import Path

import pytest


READ_INPUTS = (
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


def _make_source_root(root: Path) -> Path:
    root.mkdir()
    for name in READ_INPUTS:
        path = root / name
        if name == "expression_matrix":
            path.mkdir()
            (path / "matrix.npy").write_bytes(b"matrix")
        elif name == "bmg_gene_index.csv":
            path.write_text("gene_name,indices\nMALAT1,8\nM6PR,1;7\nARF5,0\n")
        else:
            path.write_bytes(name.encode())
    (root / "last_query_result.csv").write_text("source result\n")
    return root


def test_shadow_root_links_only_read_inputs_and_isolates_loader_output(tmp_path):
    from tools.omic_tools.celltosg_runtime_adapter import writable_celltosg_root

    source_root = _make_source_root(tmp_path / "source")

    with pytest.raises(RuntimeError, match="loader failed"):
        with writable_celltosg_root(source_root) as shadow_root:
            assert {path.name for path in shadow_root.iterdir()} == set(READ_INPUTS)
            for name in READ_INPUTS:
                linked_input = shadow_root / name
                if name == "bmg_gene_index.csv":
                    assert not linked_input.is_symlink()
                    assert linked_input.read_text() == "gene_name,indices\nM6PR,1;7\nARF5,0\n"
                    continue
                assert linked_input.is_symlink()
                assert linked_input.resolve() == (source_root / name).resolve()

            shadow_result = shadow_root / "last_query_result.csv"
            assert not shadow_result.exists()
            shadow_result.write_text("new result\n")
            assert shadow_result.read_text() == "new result\n"
            assert (source_root / "last_query_result.csv").read_text() == "source result\n"
            assert "MALAT1" in (source_root / "bmg_gene_index.csv").read_text()
            raise RuntimeError("loader failed")

    assert not shadow_root.exists()


def test_each_shadow_root_is_unique(tmp_path):
    from tools.omic_tools.celltosg_runtime_adapter import writable_celltosg_root

    source_root = _make_source_root(tmp_path / "source")

    with writable_celltosg_root(source_root) as first_root:
        with writable_celltosg_root(source_root) as second_root:
            assert first_root != second_root


def test_preselected_metadata_is_written_only_inside_shadow(tmp_path):
    import pandas as pd
    from tools.omic_tools.celltosg_runtime_adapter import writable_celltosg_root

    source = _make_source_root(tmp_path / "source")
    metadata = pd.DataFrame({"donor_id": ["D1"], "matrix_row_idx": [17]})
    original = (source / "cell_metadata_with_mappings.parquet").read_bytes()
    with writable_celltosg_root(source, metadata=metadata) as shadow:
        path = shadow / "cell_metadata_with_mappings.parquet"
        assert not path.is_symlink()
        pd.testing.assert_frame_equal(pd.read_parquet(path), metadata)
    assert (source / "cell_metadata_with_mappings.parquet").read_bytes() == original


def test_shadow_root_reports_missing_required_inputs(tmp_path):
    from tools.omic_tools.celltosg_runtime_adapter import writable_celltosg_root

    source_root = _make_source_root(tmp_path / "source")
    (source_root / "bmg_gene_index.csv").unlink()
    (source_root / "x_bio_emb.npy").unlink()

    with pytest.raises(FileNotFoundError) as exc_info:
        with writable_celltosg_root(source_root):
            pass

    message = str(exc_info.value)
    assert "bmg_gene_index.csv" in message
    assert "x_bio_emb.npy" in message

"""Integration regression test for the CellTOSG writable shadow root."""

from pathlib import Path
import sys

import pandas as pd


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

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
        elif name == "cell_metadata_with_mappings.parquet":
            pd.DataFrame({
                "disease_BMG_name": ["normal", "Alzheimer's Disease", "Alzheimer's Disease"],
                "donor_id": ["D1", "D2", "unknown"],
                "tissue_general": ["brain"] * 3,
                "CMT_name": ["astrocyte"] * 3,
                "matrix_file_path": ["matrix.npy"] * 3,
                "matrix_row_idx": [0, 1, 2],
            }).to_parquet(path, index=False)
        elif name == "bmg_gene_index.csv":
            path.write_text("gene_name,indices\nMALAT1,8\nM6PR,1;7\nARF5,0\n")
        else:
            path.write_bytes(name.encode())
    return root


def test_fetch_uses_temporary_shadow_without_writing_dataset_root(
    monkeypatch, tmp_path
):
    """Removing shadow-root use would write loader state into shared input data."""
    import omic_fetch_analysis_workflow as workflow

    source_root = _make_source_root(tmp_path / "source")
    observed = {}

    class FakeQuery:
        FIELD_ALIAS = {"disease": "disease_BMG_name"}
        last_query_conditions_resolved = {}
        last_query_conditions_raw = {}

    class WritingLoader:
        LABEL_ZERO_LABELS_BY_LABEL_COLUMN = {"disease": {"normal"}}

        def __init__(self, root, output_dir, **_kwargs):
            loader_root = Path(root)
            observed["root"] = loader_root
            observed["output_dir"] = Path(output_dir)
            observed["kwargs"] = _kwargs
            (loader_root / "last_query_result.csv").write_text("selected cells\n")

            gene_index = pd.read_csv(loader_root / "bmg_gene_index.csv")
            observed["genes"] = gene_index["gene_name"].tolist()
            self.metadata = pd.read_parquet(loader_root / "cell_metadata_with_mappings.parquet")
            self.data = pd.DataFrame(
                [[float(i + 1)] * len(gene_index) for i in range(len(self.metadata))],
                columns=observed["genes"],
            )
            self.labels = self.metadata
            self.query = FakeQuery()

    original_get_path = workflow.get_path

    def get_path_for_test(key, *args, **kwargs):
        if key == "external.omnicell_data_root":
            return str(source_root)
        return original_get_path(key, *args, **kwargs)

    monkeypatch.setattr(workflow, "get_path", get_path_for_test)
    monkeypatch.setattr(workflow, "CellTOSGDataLoader", WritingLoader)

    result = workflow.omic_fetch_with_new_loader(
        {"disease": "Alzheimer's Disease", "organ": "brain"},
        str(tmp_path / "session"),
    )

    assert result[4] is True
    assert observed["genes"] == ["M6PR", "ARF5"]
    assert result[0].columns.tolist() == ["M6PR", "ARF5"]
    assert len(result[0]) == 2
    assert set(result[2].donor_id) == {"D1", "D2"}
    assert observed["kwargs"]["sample_size"] is None
    assert observed["kwargs"]["stratified_balancing"] is False
    assert "disease" not in observed["kwargs"]["conditions"]
    assert observed["root"] != source_root
    assert not observed["root"].exists()
    assert observed["output_dir"] == tmp_path / "session"
    assert not (source_root / "last_query_result.csv").exists()
    assert "MALAT1" in (source_root / "bmg_gene_index.csv").read_text()
    diseases = (tmp_path / "session/available_diseases.txt").read_text().splitlines()
    assert diseases[0] == 'Matched disease in list: "Alzheimer\'s Disease"'
    assert diseases[1:] == ["Alzheimer's Disease", "normal"]
    assert (tmp_path / "session/available_cell_types.txt").read_text().splitlines() == [
        "Matched cell type in list: NONE (no filter requested)", "astrocyte",
    ]


def test_failed_exact_query_still_saves_full_value_lists(monkeypatch, tmp_path):
    import omic_fetch_analysis_workflow as workflow

    source_root = _make_source_root(tmp_path / "source")
    original_get_path = workflow.get_path
    monkeypatch.setattr(workflow, "get_path", lambda key, *args, **kwargs:
                        str(source_root) if key == "external.omnicell_data_root"
                        else original_get_path(key, *args, **kwargs))
    result = workflow.omic_fetch_with_new_loader(
        {"disease": "Alzheimer disease", "cell type": "astrocyte"},
        str(tmp_path / "session"),
    )
    assert result[4] is False
    diseases = (tmp_path / "session/available_diseases.txt").read_text().splitlines()
    assert diseases[0] == 'Matched disease in list: NONE (requested: "Alzheimer disease")'
    assert diseases[1:] == ["Alzheimer's Disease", "normal"]
    assert (tmp_path / "session/available_cell_types.txt").read_text().splitlines()[0] == (
        'Matched cell type in list: "astrocyte"'
    )

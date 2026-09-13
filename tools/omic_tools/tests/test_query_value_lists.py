"""Session catalogs expose the exact CellTOSG query vocabulary."""

import importlib

import pandas as pd


def _save(metadata, output_dir, **kwargs):
    module = importlib.import_module("tools.omic_tools.query_value_lists")
    return module.save_query_value_lists(metadata, output_dir, **kwargs)


def _metadata():
    return pd.DataFrame({
        "disease_BMG_name": ["normal", "Alzheimer's Disease", "Glioma", "normal", None, ""],
        "CMT_name": ["astrocyte", "astrocyte", "T cell", "microglial cell", None, " "],
        "disease": ["wrong raw disease"] * 6,
        "cell_type": ["wrong raw cell type"] * 6,
    })


def test_saves_two_unique_full_vocabulary_lists_with_matches_on_first_line(tmp_path):
    _save(_metadata(), tmp_path, disease="alzheimer's disease", cell_type="ASTROCYTE")

    diseases = (tmp_path / "available_diseases.txt").read_text().splitlines()
    cells = (tmp_path / "available_cell_types.txt").read_text().splitlines()
    assert diseases[0] == 'Matched disease in list: "Alzheimer\'s Disease"'
    assert cells[0] == 'Matched cell type in list: "astrocyte"'
    assert diseases[1:] == ["Alzheimer's Disease", "Glioma", "normal"]
    assert cells[1:] == ["astrocyte", "microglial cell", "T cell"]


def test_marks_exact_miss_and_unspecified_filter_and_overwrites_old_header(tmp_path):
    _save(_metadata(), tmp_path, disease="Glioma", cell_type="T cell")
    _save(_metadata(), tmp_path, disease="Alzheimer disease")

    diseases = (tmp_path / "available_diseases.txt").read_text().splitlines()
    cells = (tmp_path / "available_cell_types.txt").read_text().splitlines()
    assert diseases[0] == 'Matched disease in list: NONE (requested: "Alzheimer disease")'
    assert cells[0] == 'Matched cell type in list: NONE (no filter requested)'
    assert len(diseases) == 4
    assert len(cells) == 4


def test_headers_remain_one_line_for_unmatched_input_with_newline(tmp_path):
    _save(_metadata(), tmp_path, disease="not\na disease")
    lines = (tmp_path / "available_diseases.txt").read_text().splitlines()
    assert len(lines) == 4
    assert "not\\na disease" in lines[0]

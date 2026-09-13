import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def test_celltosg_collapses_once_across_chunks_and_keeps_original_indices(tmp_path):
    module = importlib.import_module("tools.omic_tools.pseudobulk_pipeline.retrieval")
    celltosg = Path('/storage3/fs1/fuhai.li/Active/di.huang/Research/LLM/OmniCellTOSG')
    if not (celltosg / 'CellTOSG_Loader').exists():
        pytest.skip('CellTOSG is not installed')
    root = tmp_path / "dataset"
    (root / "expression_matrix").mkdir(parents=True)
    matrix = np.array([[0, 2, 1, 99, 0, 0], [0, 2, 2, 99, 0, 0],
                       [0, 0, 3, 99, 9, 0], [0, 0, 4, 99, 9, 0]], dtype=np.float32)
    np.save(root / 'expression_matrix/test.npy', matrix)
    (root / 'bmg_gene_index.csv').write_text('gene_name,indices\nA,1;4\nB,2\nNCRNA,3\n')
    reference = tmp_path / 'hgnc.tsv'
    reference.write_text('symbol\tstatus\tlocus_group\nA\tApproved\tprotein-coding gene\nB\tApproved\tprotein-coding gene\nNCRNA\tApproved\tnon-coding RNA\n')
    meta = pd.DataFrame({'matrix_file_path': ['test.npy'] * 4, 'matrix_row_idx': [3, 0, 2, 1]})
    counts, genes, choices, audit = module.load_raw_gene_counts(
        meta, root, celltosg, reference, tmp_path / 'retrieval', chunk_size=1
    )
    assert genes == ['A', 'B']
    np.testing.assert_array_equal(counts, matrix[[3, 0, 2, 1]][:, [4, 2]])
    assert choices.chosen_bmg_index.tolist() == [4, 2]
    assert audit['collapse_scope'] == 'entire_selected_cohort'
    assert audit['protein_coding_genes'] == 2


def test_hgnc_mapping_requires_approved_protein_coding_symbols(tmp_path):
    module = importlib.import_module("tools.omic_tools.pseudobulk_pipeline.retrieval")
    reference = tmp_path / 'hgnc.tsv'
    reference.write_text('symbol\tstatus\tlocus_group\nA\tApproved\tprotein-coding gene\nB\tEntry Withdrawn\tprotein-coding gene\n')
    mapping = tmp_path / 'mapping.csv'
    mapping.write_text('gene_name,indices\nB,4\nA,2;3\nC,8\n')
    rows, audit = module.filter_gene_mapping(mapping, reference)
    assert rows == [('A', [2, 3])]
    assert audit['excluded_bmg_genes'] == 2

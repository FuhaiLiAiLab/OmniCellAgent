import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def test_existing_output_directory_is_never_reused(tmp_path):
    module = importlib.import_module("tools.omic_tools.pseudobulk_pipeline.pipeline_stages")
    existing = tmp_path / "old_results"
    existing.mkdir()
    marker = existing / "previous.txt"
    marker.write_text("unchanged")
    with pytest.raises(FileExistsError):
        module.run_pipeline(output_dir=existing, data_root=tmp_path, celltosg_root=tmp_path,
                            hgnc_reference=tmp_path / "reference", prepare_only=True)
    assert marker.read_text() == "unchanged"


def test_preparation_aggregates_raw_counts_and_exports_donor_units(tmp_path, monkeypatch):
    module = importlib.import_module("tools.omic_tools.pseudobulk_pipeline.pipeline_stages")
    root = tmp_path / "data"
    root.mkdir()
    metadata = pd.DataFrame({
        "source": ["atlas"] * 3, "dataset_id": ["study"] * 3, "donor_id": ["A", "A", "B"],
        "disease_BMG_name": ["Alzheimer's Disease", "Alzheimer's Disease", "normal"],
        "CMT_name": ["astrocyte"] * 3, "tissue_general": ["brain"] * 3,
        "sex_normalized": ["male", "female", "female"],
        "development_stage_numeric_age": [70, 70, 65],
        "matrix_file_path": ["test.npy"] * 3, "matrix_row_idx": [0, 1, 2],
    })
    metadata.to_parquet(root / "cell_metadata_with_mappings.parquet", index=False)
    reference = tmp_path / "hgnc.tsv"
    reference.write_text("symbol\tstatus\tlocus_group\nG1\tApproved\tprotein-coding gene\n")
    (root / "bmg_gene_index.csv").write_text("gene_name,indices\nG1,0\n")
    values = np.array([[10], [20], [100]], dtype=np.int64)
    monkeypatch.setattr(module, "load_raw_gene_counts", lambda *args, **kwargs:
                        (values, ["G1"], pd.DataFrame(), {"normalization": "none"}))
    out = tmp_path / "run"
    result = module.run_pipeline(output_dir=out, data_root=root, celltosg_root=tmp_path,
                                 hgnc_reference=reference, prepare_only=True)
    counts = pd.read_csv(out / "inputs/pseudobulk_counts.csv", index_col=0)
    donors = pd.read_csv(out / "inputs/donor_metadata.csv")
    assert counts.to_numpy().sum() == 130
    assert sorted(counts.iloc[0].tolist()) == [30, 100]
    assert counts.columns.tolist() == donors.sample_id.tolist()
    assert donors.loc[donors.donor_id.eq("A"), "sex"].isna().all()
    assert result["status"] == "prepared"
    assert result["comparison_names"] == ["ad_vs_control", "ad_male_vs_female", "control_male_vs_female"]
    assert (out / "inputs/donor_metadata_issues.csv").exists()
    assert not (out / "comparisons").exists()


def test_r_process_uses_fresh_run_directory_and_ignores_user_startup_files(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    module = importlib.import_module("tools.omic_tools.pseudobulk_pipeline.pipeline_stages")
    observed = {}
    def fake_run(command, **kwargs):
        observed.update(command=command, **kwargs)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(module.subprocess, "run", fake_run)
    module._run_deseq2(sys.executable, tmp_path / "counts.csv", tmp_path / "metadata.csv",
                       tmp_path / "comparisons", tmp_path / "fit.log", 0.05, 10, 3, 3)
    assert observed["command"][1] == "--vanilla"
    assert observed["cwd"] == tmp_path

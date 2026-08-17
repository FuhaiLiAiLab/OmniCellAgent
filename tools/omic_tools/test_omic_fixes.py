"""Tests for the omic workflow correctness and validity fixes.

Run from the repo root:
    /storage3/fs1/fuhai.li/Active/di.huang/cache/miniconda/omnicellagent/bin/python \
        -m pytest tools/omic_tools/test_omic_fixes.py -v
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, REPO_ROOT)

FIXTURE_DIR = os.path.join(
    REPO_ROOT, "webapp", "sessions", "test_suite", "breast_cancer"
)

requires_fixture = pytest.mark.skipif(
    not os.path.exists(os.path.join(FIXTURE_DIR, "expression_gene.csv")),
    reason="breast_cancer test-suite fixture not present",
)


@pytest.fixture(scope="module")
def cohort():
    """Real 134-cell breast_cancer cohort: (X DataFrame, Y array, metadata)."""
    X = pd.read_csv(os.path.join(FIXTURE_DIR, "expression_gene.csv"), index_col=0)
    meta = pd.read_csv(os.path.join(FIXTURE_DIR, "labels.csv"))
    Y = (meta["disease_BMG_name"] != "normal").astype(int).values
    return X, Y, meta


@requires_fixture
def test_omic_analysis_accepts_gene_names_and_completes(cohort, tmp_path):
    from omic_analysis_components import omic_analysis

    X, Y, _ = cohort
    gene_names = list(X.columns)
    data_dict = {
        "normal_omic_feature": X[Y == 0].values,
        "disease_omic_feature": X[Y == 1].values,
        "omic_label": Y,
    }
    result = omic_analysis(
        "test_contrast",
        data_dict,
        enable_plotting=False,
        session_dir=str(tmp_path),
        gene_names=gene_names,
    )
    assert isinstance(result, dict)
    de_dir = result["differential_expression_dir"]
    table = pd.read_csv(
        os.path.join(de_dir, "unpaired_differential_expression_results.csv")
    )
    assert len(table) == 41149
    assert table["Name"].iloc[0] == "ARF5"
    assert table["Name"].notna().all()


DATASET_ROOT = "/storage3/fs1/fuhai.li/Active/Shared/dataset/OmniCellTOSG_dataset"

requires_dataset = pytest.mark.skipif(
    not os.path.exists(os.path.join(DATASET_ROOT, "cell_metadata_with_mappings.parquet")),
    reason="OmniCellTOSG dataset not present",
)


@requires_dataset
def test_suggestions_come_from_the_queried_column():
    """Suggestions must be values that a query can actually match.

    'Alzheimer disease' lives in the raw `disease` column; queries filter
    `disease_BMG_name`, which holds only "Alzheimer's Disease".
    """
    from omic_fetch_analysis_workflow import get_suggestions

    meta = pd.read_parquet(
        os.path.join(DATASET_ROOT, "cell_metadata_with_mappings.parquet"),
        columns=["disease_BMG_name"],
    )
    queryable = set(meta["disease_BMG_name"].dropna().astype(str))

    message = get_suggestions({"disease": "Alzheimer disease"}, n_matches=5)

    assert "Alzheimer's Disease" in message
    # Every suggested value must exist in the column that is actually filtered.
    import ast
    suggested = message.split("-> try ", 1)[1].strip()
    for value in ast.literal_eval(suggested):
        assert value in queryable, f"suggested {value!r} is not in disease_BMG_name"


def test_normalize_cp10k_removes_pure_depth_artifact():
    """A group that differs only by 2x depth must show log2FC 0 after CP10K."""
    from omic_fetch_analysis_workflow import normalize_cp10k

    rng = np.random.default_rng(0)
    base = rng.poisson(5, size=(50, 200)).astype(float)
    X = np.vstack([base, base * 2.0])
    Y = np.array([0] * 50 + [1] * 50)
    eps = 1e-8

    def median_lfc(matrix):
        d = matrix[Y == 1].mean(axis=0)
        c = matrix[Y == 0].mean(axis=0)
        return float(np.median(np.log2((d + eps) / (c + eps))))

    assert median_lfc(X) == pytest.approx(1.0, abs=1e-6)
    normalized = normalize_cp10k(pd.DataFrame(X)).values
    assert median_lfc(normalized) == pytest.approx(0.0, abs=1e-6)


def test_normalize_cp10k_preserves_dataframe_columns():
    """Gene symbols must survive normalization; Task 1 depends on X.columns."""
    from omic_fetch_analysis_workflow import normalize_cp10k

    X = pd.DataFrame([[1.0, 3.0], [2.0, 2.0]], columns=["ARF5", "M6PR"])
    out = normalize_cp10k(X)
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == ["ARF5", "M6PR"]
    assert out.values.sum(axis=1) == pytest.approx([1e4, 1e4])


def test_normalize_cp10k_handles_all_zero_cell():
    """An all-zero cell must not produce inf or nan."""
    from omic_fetch_analysis_workflow import normalize_cp10k

    X = pd.DataFrame([[0.0, 0.0], [1.0, 1.0]])
    out = normalize_cp10k(X).values
    assert np.isfinite(out).all()
    assert out[0].sum() == pytest.approx(0.0)

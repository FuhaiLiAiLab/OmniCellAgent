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

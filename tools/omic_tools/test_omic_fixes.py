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


def test_priority_labels_collapse_to_class_zero():
    """Match the loader: every label-zero value becomes class 0, not 0/1/2."""
    from omic_fetch_analysis_workflow import _build_labels_from_metadata

    meta = pd.DataFrame({
        "disease_BMG_name": (
            ["normal"] * 5 + ["unknown"] * 2 + ["Unclassified"] * 2
            + ["Alzheimer's Disease"] * 4 + ["Glioma"] * 3
        )
    })
    priority = {"normal", "unclassified", "unknown"}
    Y, mapping, counts, valid = _build_labels_from_metadata(
        meta, "disease", {"disease": "disease_BMG_name"}, priority
    )
    assert mapping["normal"] == 0
    assert mapping["unknown"] == 0
    assert mapping["Unclassified"] == 0
    assert mapping["Alzheimer's Disease"] != 0
    assert mapping["Glioma"] != 0
    assert mapping["Alzheimer's Disease"] != mapping["Glioma"]
    assert counts[0] == 9


def test_select_contrast_picks_largest_non_reference_class():
    """microglia_brain shape: normal vs the dominant disease, not vs the rarest."""
    from omic_fetch_analysis_workflow import select_contrast

    Y = np.array([0] * 1000 + [1] * 10 + [2] * 393 + [3] * 25)
    mapping = {"normal": 0, "Unclassified": 1, "Alzheimer's Disease": 2, "ALS": 3}
    contrast = select_contrast(Y, mapping)
    assert contrast["ref_class"] == 0
    assert contrast["alt_class"] == 2
    assert contrast["alt_name"] == "Alzheimer's Disease"
    assert sorted(contrast["excluded"]) == [1, 3]


def test_select_contrast_returns_none_with_one_class():
    from omic_fetch_analysis_workflow import select_contrast

    assert select_contrast(np.zeros(10, dtype=int), {"normal": 0}) is None


def test_all_return_paths_have_matching_arity():
    """Line 336 returned 5 values while the caller unpacks 8."""
    import ast
    import inspect
    from omic_fetch_analysis_workflow import omic_fetch_with_new_loader

    source = inspect.getsource(omic_fetch_with_new_loader)
    tree = ast.parse(source)
    arities = {
        len(node.value.elts)
        for node in ast.walk(tree)
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Tuple)
    }
    assert arities == {8}, f"inconsistent return arities: {sorted(arities)}"


def test_success_return_dict_exposes_contrast():
    """The contrast identity must be machine-readable, not just printed.

    Without this, a no-disease query (e.g. "microglia in brain") runs DE under
    comparison_name="comparison" and the caller has no way to learn which two
    classes were actually compared.
    """
    import ast
    import inspect
    from omic_fetch_analysis_workflow import omic_fetch_analysis_workflow

    source = inspect.getsource(omic_fetch_analysis_workflow)
    tree = ast.parse(source)
    success_dicts = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Return)
        and isinstance(node.value, ast.Dict)
        and any(
            isinstance(k, ast.Constant) and k.value == "comparison_name"
            for k in node.value.keys
        )
    ]
    assert len(success_dicts) == 1, "expected exactly one success return dict"
    keys = {k.value for k in success_dicts[0].keys if isinstance(k, ast.Constant)}
    assert "contrast" in keys


@requires_fixture
def test_breast_cancer_cohort_is_unreliable(cohort):
    """4 checks fail: protocol, dataset overlap, donor count, donor dominance."""
    from cohort_diagnostics import compute_cohort_diagnostics

    X, Y, meta = cohort
    lib = X.values.sum(axis=1)
    diag = compute_cohort_diagnostics(meta, Y == 0, Y == 1, lib_sizes=lib)

    assert diag["verdict"] == "unreliable"
    names = {c["check"] for c in diag["failed_checks"]}
    assert "protocol_balance" in names
    assert "dataset_overlap" in names
    assert "donor_count" in names
    assert "donor_dominance" in names


def test_balanced_cohort_is_ok():
    from cohort_diagnostics import compute_cohort_diagnostics

    n = 400
    meta = pd.DataFrame({
        "suspension_type": ["cell"] * n,
        "dataset_id": [f"ds{i % 12}" for i in range(n)],
        "donor_id": [f"donor{i % 40}" for i in range(n)],
    })
    # Block split, not row-interleaved: with a period-2 interleave, the even
    # moduli used for dataset_id (%12) and donor_id (%40) alias perfectly
    # with group parity (gcd(2, 12) = gcd(2, 40) = 2), so each group would
    # only ever see half the datasets/donors -- e.g. 0 of 12 datasets shared,
    # a spurious FAIL in a fixture meant to be genuinely comparable.
    is_ref = np.array([True] * (n // 2) + [False] * (n // 2))
    diag = compute_cohort_diagnostics(meta, is_ref, ~is_ref,
                                      lib_sizes=np.full(n, 1000.0))
    assert diag["verdict"] == "ok"
    assert diag["failed_checks"] == []


def test_unknown_donors_are_not_treated_as_one_donor():
    """35% of microglia_brain cells carry donor_id 'unknown'."""
    from cohort_diagnostics import compute_cohort_diagnostics

    n = 100
    meta = pd.DataFrame({
        "suspension_type": ["cell"] * n,
        "dataset_id": [f"ds{i % 6}" for i in range(n)],
        "donor_id": ["unknown"] * 40 + [f"donor{i % 30}" for i in range(60)],
    })
    is_ref = np.array([True, False] * 50)
    diag = compute_cohort_diagnostics(meta, is_ref, ~is_ref)
    names = {c["check"] for c in diag["failed_checks"]}
    assert "donor_usability" in names
    assert diag["checks"]["donor_usability"]["value"] == pytest.approx(0.40)


def test_missing_columns_do_not_raise():
    from cohort_diagnostics import compute_cohort_diagnostics

    meta = pd.DataFrame({"irrelevant": range(10)})
    is_ref = np.array([True] * 5 + [False] * 5)
    diag = compute_cohort_diagnostics(meta, is_ref, ~is_ref)
    assert diag["verdict"] in {"ok", "caution", "unreliable"}
    assert isinstance(diag["failed_checks"], list)


def test_success_return_dict_exposes_cohort_diagnostics():
    """Task 6: top_genes_by_fdr reaches PubMed search and the PDF with no
    verdict attached unless cohort_diagnostics/cohort_verdict ride along in
    the same success return dict."""
    import ast
    import inspect
    from omic_fetch_analysis_workflow import omic_fetch_analysis_workflow

    source = inspect.getsource(omic_fetch_analysis_workflow)
    tree = ast.parse(source)
    success_dicts = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Return)
        and isinstance(node.value, ast.Dict)
        and any(
            isinstance(k, ast.Constant) and k.value == "comparison_name"
            for k in node.value.keys
        )
    ]
    assert len(success_dicts) == 1, "expected exactly one success return dict"
    keys = {k.value for k in success_dicts[0].keys if isinstance(k, ast.Constant)}
    assert "cohort_diagnostics" in keys
    assert "cohort_verdict" in keys


def test_format_diagnostics_text_lists_failures():
    from cohort_diagnostics import format_diagnostics_text

    diag = {
        "verdict": "unreliable",
        "groups": {
            "reference": {"name": "normal", "n_cells": 67},
            "alternate": {"name": "breast cancer", "n_cells": 67},
        },
        "failed_checks": [{"check": "donor_count", "detail": "donors per group: normal=45, breast cancer=5"}],
        "caution_checks": [],
    }
    text = format_diagnostics_text(diag)
    assert "UNRELIABLE" in text
    assert "donor_count" in text
    assert "normal (n=67)" in text
    assert "anti-conservative" in text


def test_save_dataframe_stamps_diagnostics_header_under_joblib(tmp_path):
    """save_dataframe runs inside joblib.Parallel(n_jobs=-1); a closure over
    diagnostics_text must still pickle to worker processes, and the CSVs it
    writes must be re-readable with pd.read_csv(path, comment="#")."""
    from omic_analysis_components import perform_unpaired_differential_expression

    rng = np.random.default_rng(0)
    n_genes, n_disease, n_control = 40, 6, 6
    disease_df = pd.DataFrame(
        rng.poisson(5, size=(n_genes, n_disease)).astype(float),
        columns=[f"ds_sample_{i}" for i in range(n_disease)],
    )
    disease_df.insert(0, "Name", [f"GENE{i}" for i in range(n_genes)])
    control_df = pd.DataFrame(
        rng.poisson(5, size=(n_genes, n_control)).astype(float),
        columns=[f"ns_sample_{i}" for i in range(n_control)],
    )
    control_df.insert(0, "Name", [f"GENE{i}" for i in range(n_genes)])

    diagnostics_text = (
        "COHORT DIAGNOSTICS: UNRELIABLE\n"
        "  contrast: normal (n=6) vs breast cancer (n=6)\n"
        "  FAIL    donor_count: donors per group: normal=6, breast cancer=1"
    )

    perform_unpaired_differential_expression(
        disease_df=disease_df,
        normal_df=control_df,
        p_value_threshold=0.5,
        log2fc_threshold=0.0,
        sig_top_n=n_genes,
        n_jobs=-1,
        disease="test",
        de_output_dir=str(tmp_path),
        diagnostics_text=diagnostics_text,
    )

    out_path = os.path.join(str(tmp_path), "unpaired_differential_expression_results.csv")
    with open(out_path) as handle:
        header_lines = [next(handle) for _ in range(3)]
    assert header_lines[0] == "# COHORT DIAGNOSTICS: UNRELIABLE\n"
    assert header_lines[1].startswith("#   contrast:")
    assert header_lines[2].startswith("#   FAIL")

    reloaded = pd.read_csv(out_path, comment="#")
    assert len(reloaded) == n_genes
    assert "Name" in reloaded.columns
    assert "FDR" in reloaded.columns


def test_diagnostics_failure_does_not_block_de():
    """Fix-round 1, Finding 1: diagnostics are advisory only. A failure inside
    compute_cohort_diagnostics or the cohort_diagnostics.json sidecar write
    must not prevent DE (the omic_analysis call) from running. They must live
    in their own try/except, separate from the try/except guarding
    omic_analysis, with a non-fatal fallback to cohort_diagnostics=None /
    diagnostics_text=""."""
    import ast
    import inspect
    from omic_fetch_analysis_workflow import omic_fetch_analysis_workflow

    source = inspect.getsource(omic_fetch_analysis_workflow)
    tree = ast.parse(source)

    def calls_name(node, name):
        return any(
            isinstance(n, ast.Call)
            and (
                (isinstance(n.func, ast.Name) and n.func.id == name)
                or (isinstance(n.func, ast.Attribute) and n.func.attr == name)
            )
            for n in ast.walk(node)
        )

    diagnostics_trys = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Try) and calls_name(node, "compute_cohort_diagnostics")
    ]
    assert len(diagnostics_trys) == 1, (
        "compute_cohort_diagnostics must be wrapped in exactly one try block"
    )
    diag_try = diagnostics_trys[0]

    assert not calls_name(diag_try, "omic_analysis"), (
        "the diagnostics try/except must be separate from the try/except guarding omic_analysis"
    )

    assert len(diag_try.handlers) == 1, "expected exactly one except handler"
    handler = diag_try.handlers[0]
    assert isinstance(handler.type, ast.Name) and handler.type.id == "Exception", (
        "must catch broadly with except Exception -- this is advisory code"
    )

    def assigned_names(stmts):
        names = set()
        for stmt in stmts:
            for node in ast.walk(stmt):
                if isinstance(node, ast.Assign):
                    names |= {t.id for t in node.targets if isinstance(t, ast.Name)}
        return names

    fallback_names = assigned_names(handler.body)
    assert "cohort_diagnostics" in fallback_names, "handler must reset cohort_diagnostics"
    assert "diagnostics_text" in fallback_names, "handler must reset diagnostics_text"


def test_pdf_renders_cohort_caveat():
    """Fix-round 1, Finding 2: the PDF's Shared Data Summary (A4) must render
    a cohort-diagnostics caveat before the top-genes table whenever
    cohort_verdict is not "ok". Storing the verdict in shared_data (Task 6
    Step 6) is not enough if the PDF section never reads it back.

    Reads agent/langgraph_agent.py's source text directly instead of
    importing the module, since the module has heavy side-effecting imports
    (langgraph/langchain, API clients) unrelated to this check.
    """
    import ast

    agent_path = os.path.join(REPO_ROOT, "agent", "langgraph_agent.py")
    with open(agent_path) as handle:
        source = handle.read()
    tree = ast.parse(source)

    appendix_fn = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_generate_appendix"
    )
    fn_source = ast.get_source_segment(source, appendix_fn)
    assert fn_source is not None

    verdict_idx = fn_source.find("cohort_verdict")
    table_idx = fn_source.find("| # | Gene")
    assert verdict_idx != -1, "PDF appendix must read shared_data['cohort_verdict']"
    assert table_idx != -1, "top-genes table markup not found (fixture out of date?)"
    assert verdict_idx < table_idx, "cohort caveat must render BEFORE the top-genes table"

    assert '!= "ok"' in fn_source or "!= 'ok'" in fn_source, (
        "caveat must be gated on the verdict not being 'ok'"
    )
    assert "cohort_diagnostics" in fn_source, (
        "caveat must read shared_data['cohort_diagnostics'] to surface check details"
    )
    assert "failed_checks" in fn_source and "caution_checks" in fn_source, (
        "caveat must surface both failed_checks and caution_checks"
    )

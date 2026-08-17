# Omic Workflow Correctness and Biological Validity Fixes — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair the three defects that produced zero differential-expression output across all five test cohorts, and correct the biological validity problems the crash was masking.

**Architecture:** All changes are confined to `tools/omic_tools/`. Gene identity is taken from the loader's own DataFrame columns rather than re-derived from BioMedGraphica CSVs. Expression is CP10K-normalized before any statistic is computed. A new standalone module computes cohort comparability diagnostics that warn but never withhold results.

**Tech Stack:** Python 3.11, pandas 2.3.3, numpy 2.2.6, scipy 1.15.3, statsmodels 0.14.5, pytest 9.0.1.

## Global Constraints

- **`CellTOSG_Loader` must not be modified.** It lives at `/storage3/fs1/fuhai.li/Active/di.huang/Research/LLM/OmniCellTOSG/CellTOSG_Loader/`. Read it freely; never edit it.
- All edits are limited to `tools/omic_tools/`, with exactly one exception: Task 6 Step 6 adds ~5 lines to `agent/langgraph_agent.py` to surface the cohort verdict into `shared_data`. No other task may touch anything outside `tools/omic_tools/`.
- Python interpreter for all commands: `/storage3/fs1/fuhai.li/Active/di.huang/cache/miniconda/omnicellagent/bin/python`
- The cohort validity gate **warns; it never refuses**. A DE table is always produced regardless of verdict.
- Do **not** apply `log1p` anywhere in the DE path. Fold change must stay on the linear CP10K scale.
- `min_expression_threshold = 0.1` in `perform_unpaired_differential_expression` stays at `0.1`. On CP10K the mean per gene is `10000/41149 = 0.243`, which makes `0.1` a meaningful filter.
- Modules in `tools/omic_tools/` use flat imports (`from omic_analysis_components import omic_analysis`). Tests must insert `tools/omic_tools` into `sys.path`.
- Never assert a fixed median log2FC on real cohort data. Use the synthetic 2x-depth fixture for that property.

## File Structure

| File | Responsibility |
| --- | --- |
| `tools/omic_tools/cohort_diagnostics.py` | **New.** Pure functions computing cohort comparability metrics and verdict. No I/O, no loader import. |
| `tools/omic_tools/test_omic_fixes.py` | **New.** Pytest suite for all tasks. Colocated per repo convention (`mcp_tools/test_mcp.py`, `utils/test_neo4j.py`). |
| `tools/omic_tools/omic_analysis_components.py` | **Modify.** Accept `gene_names`; delete the obsolete BioMedGraphica mapping block. |
| `tools/omic_tools/omic_fetch_analysis_workflow.py` | **Modify.** CP10K normalization, gene-name capture, contrast guard, diagnostics wiring, `suspension_type` parameter, STEP 3 symbols. |

`omic_fetch_analysis_workflow.py` is 1,229 lines and `omic_analysis_components.py` is 1,306. Diagnostics go in their own module rather than growing either further.

## Test Fixture

The saved test-suite session is the fixture. It already exists on disk:

```
webapp/sessions/test_suite/breast_cancer/
    expression_gene.csv       134 rows x 41,149 gene columns (23 MB)
    labels.csv                134 rows of cell metadata
    bmg_to_gene_choice.csv    41,149 rows, column `gene_name`
```

Tests that need it skip cleanly when it is absent.

---

### Task 1: Gene names from the loader's DataFrame

Removes the crash: `ValueError: Length of values (533458) does not match length of index (41149)`.

**Files:**
- Create: `tools/omic_tools/test_omic_fixes.py`
- Modify: `tools/omic_tools/omic_analysis_components.py:46` (signature), `:60-118` (delete block)
- Modify: `tools/omic_tools/omic_fetch_analysis_workflow.py` (capture and pass `gene_names`)

**Interfaces:**
- Produces: `omic_analysis(disease_name, data_dict, enable_plotting=True, session_dir=None, gene_names=None) -> dict`
- Produces: `_resolve_gene_names(gene_names, n_features, session_dir) -> list[str]` in `omic_analysis_components.py`
- Produces: module-level `gene_names` variable in the workflow, captured immediately after fetch

- [ ] **Step 1: Write the failing test**

Create `tools/omic_tools/test_omic_fixes.py`:

```python
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
```

The five DE output filenames are fixed at `omic_analysis_components.py:401-406`:
`unpaired_differential_expression_results.csv`, `significant_genes_by_fdr.csv`,
`significant_genes_by_fc.csv`, `significant_upregulated_genes.csv`,
`significant_downregulated_genes.csv`.

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
cd /storage3/fs1/fuhai.li/Active/di.huang/Research/LLM/OmniCellAgent && \
/storage3/fs1/fuhai.li/Active/di.huang/cache/miniconda/omnicellagent/bin/python \
    -m pytest tools/omic_tools/test_omic_fixes.py::test_omic_analysis_accepts_gene_names_and_completes -v
```
Expected: FAIL with `TypeError: omic_analysis() got an unexpected keyword argument 'gene_names'`

If the DE results filename differs from `unpaired_de_results.csv`, read the tail of `perform_unpaired_differential_expression` (around `omic_analysis_components.py:404`) and use the real name in the assertion.

- [ ] **Step 3: Add the gene-name resolver**

Insert into `tools/omic_tools/omic_analysis_components.py` immediately above `def omic_analysis` (line 46):

```python
def _resolve_gene_names(gene_names, n_features: int, session_dir: str = None) -> list:
    """Resolve gene symbols for the expression feature axis.

    The loader returns a DataFrame whose columns are HGNC symbols, so the caller
    should pass them in. Falls back to the loader's own sidecar file, then fails
    loudly rather than guessing.
    """
    if gene_names is not None:
        names = list(gene_names)
        if len(names) != n_features:
            raise ValueError(
                f"gene_names has {len(names)} entries but the matrix has "
                f"{n_features} features."
            )
        return names

    choice_path = os.path.join(session_dir or "", "bmg_to_gene_choice.csv")
    if os.path.exists(choice_path):
        names = pd.read_csv(choice_path)["gene_name"].tolist()
        if len(names) == n_features:
            return names
        raise ValueError(
            f"{choice_path} has {len(names)} genes but the matrix has "
            f"{n_features} features."
        )

    raise ValueError(
        "Cannot determine gene names: none were passed and "
        f"{choice_path} is missing. The loader emits gene symbols as the "
        "columns of `dataset.data`; capture them before np.nan_to_num()."
    )
```

- [ ] **Step 4: Change the signature and replace the obsolete block**

In `tools/omic_tools/omic_analysis_components.py`, change line 46 to:

```python
def omic_analysis(disease_name: str, data_dict: dict, enable_plotting: bool = True, session_dir: str = None, gene_names: list = None) -> dict:
```

Then delete everything from the `# Load transcriptomics and proteomics data` comment through the two `groupby('Name').mean().reset_index()` lines (approximately lines 68–118) and replace with:

```python
    n_features = combined_disease_matrix.shape[1]
    gene_names = _resolve_gene_names(gene_names, n_features, session_dir)

    # The loader already collapsed transcripts to one representative per gene
    # (CellTOSG_Loader/data_loader.py: bmg_matrix_to_gene_matrix), so the feature
    # axis is 41,149 HGNC symbols. Do not re-derive it from the BioMedGraphica
    # entity tables — those describe a 533,458-row axis this data no longer uses.
    print("Creating disease DataFrame...")
    combined_disease_df = pd.DataFrame(
        combined_disease_matrix.T,
        columns=[f'ds_sample_{i}' for i in range(combined_disease_matrix.shape[0])],
    )
    combined_disease_df.insert(0, 'Name', gene_names)
    del combined_disease_matrix
    gc.collect()

    print("Creating normal DataFrame...")
    combined_normal_df = pd.DataFrame(
        combined_normal_matrix.T,
        columns=[f'ns_sample_{i}' for i in range(combined_normal_matrix.shape[0])],
    )
    combined_normal_df.insert(0, 'Name', gene_names)
    del combined_normal_matrix
    gc.collect()
```

Keep the two `np.nan_to_num` lines above this block. Keep everything from `# Print the shapes of the aggregated DataFrames` onward.

- [ ] **Step 5: Run test to verify it passes**

Run:
```bash
cd /storage3/fs1/fuhai.li/Active/di.huang/Research/LLM/OmniCellAgent && \
/storage3/fs1/fuhai.li/Active/di.huang/cache/miniconda/omnicellagent/bin/python \
    -m pytest tools/omic_tools/test_omic_fixes.py::test_omic_analysis_accepts_gene_names_and_completes -v
```
Expected: PASS

- [ ] **Step 6: Capture and pass gene_names in the workflow**

In `tools/omic_tools/omic_fetch_analysis_workflow.py`, find the fetch call in `omic_fetch_analysis_workflow` (around line 624):

```python
    (X, Y, metadata, similar_terms, retrieval_success,
     actual_label, label_fallback_message) = omic_fetch_with_new_loader(...)
```

Immediately after it, add:

```python
    # The loader returns a DataFrame whose columns are HGNC gene symbols.
    # Capture them now: np.nan_to_num() downstream returns a bare ndarray and
    # destroys column labels.
    gene_names = list(X.columns) if hasattr(X, "columns") else None
```

Then in STEP 4, change the `omic_analysis(...)` call (around line 759) to pass them:

```python
                    data_and_analysis_dict = omic_analysis(
                        comparison_name,
                        data_dict,
                        enable_plotting=enable_plotting,
                        session_dir=session_dir,
                        gene_names=gene_names,
                    )
```

- [ ] **Step 7: Commit**

```bash
git add tools/omic_tools/test_omic_fixes.py tools/omic_tools/omic_analysis_components.py tools/omic_tools/omic_fetch_analysis_workflow.py
git commit -m "fix(omic): source gene names from loader columns instead of BioMedGraphica axis

The loader collapses transcripts to 41,149 gene symbols upstream. omic_analysis
still assumed the old 533,458-row entity axis and crashed on every DE run."
```

---

### Task 2: Suggestion vocabulary resolved through FIELD_ALIAS

The workflow currently suggests disease names drawn from a column it does not query, including the exact string that just failed.

**Files:**
- Modify: `tools/omic_tools/omic_fetch_analysis_workflow.py:60-87` (`get_suggestions`)
- Test: `tools/omic_tools/test_omic_fixes.py`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: `get_suggestions(conditions: dict, n_matches: int = 5) -> str` — signature unchanged, values now drawn from queried columns.

- [ ] **Step 1: Write the failing test**

Append to `tools/omic_tools/test_omic_fixes.py`:

```python
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
```

`get_suggestions` formats each line as `  <field>: '<query>' -> try [<values>]`,
so the text after `-> try ` is a Python list literal. Parse it with
`ast.literal_eval`, never `eval`.

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
cd /storage3/fs1/fuhai.li/Active/di.huang/Research/LLM/OmniCellAgent && \
/storage3/fs1/fuhai.li/Active/di.huang/cache/miniconda/omnicellagent/bin/python \
    -m pytest tools/omic_tools/test_omic_fixes.py::test_suggestions_come_from_the_queried_column -v
```
Expected: FAIL — `suggested 'Alzheimer disease' is not in disease_BMG_name`

- [ ] **Step 3: Resolve field names through the loader's alias map**

In `tools/omic_tools/omic_fetch_analysis_workflow.py`, inside `get_suggestions`, replace:

```python
    query_builder = _get_query_builder()
    field_map = {"disease": "disease", "cell_type": "cell_type", "tissue_general": "tissue_general"}
    suggestions = {}
```

with:

```python
    query_builder = _get_query_builder()
    # available_conditions() filters include_fields directly against dataframe
    # columns and does NOT apply FIELD_ALIAS (subset_builder.py:67). Resolve here
    # or suggestions come from a column the query never touches.
    field_map = dict(getattr(query_builder, "FIELD_ALIAS", {}))
    field_map.setdefault("tissue_general", "tissue_general")
    suggestions = {}
```

`CellTOSGSubsetBuilder.FIELD_ALIAS` maps `cell_type -> CMT_name`, `disease -> disease_BMG_name`, `development_stage -> development_stage_category`, `sex -> sex_normalized`. `tissue_general` has no alias and resolves to itself.

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
cd /storage3/fs1/fuhai.li/Active/di.huang/Research/LLM/OmniCellAgent && \
/storage3/fs1/fuhai.li/Active/di.huang/cache/miniconda/omnicellagent/bin/python \
    -m pytest tools/omic_tools/test_omic_fixes.py::test_suggestions_come_from_the_queried_column -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tools/omic_tools/omic_fetch_analysis_workflow.py tools/omic_tools/test_omic_fixes.py
git commit -m "fix(omic): draw query suggestions from the columns actually filtered

get_suggestions read raw `disease`/`cell_type` while queries filter
`disease_BMG_name`/`CMT_name`, so it suggested values that can never match."
```

---

### Task 3: CP10K normalization

Sequencing depth is confounded with disease status (median library size 681,715 normal vs 379,731 disease).

**Files:**
- Modify: `tools/omic_tools/omic_fetch_analysis_workflow.py` (add `normalize_cp10k`, apply after fetch)
- Test: `tools/omic_tools/test_omic_fixes.py`

**Interfaces:**
- Consumes: `gene_names` capture from Task 1 (normalization must preserve it).
- Produces: `normalize_cp10k(X, target_sum: float = 1e4)` returning the same type as `X` (DataFrame in, DataFrame out).
- Produces: module-level `lib_sizes` (numpy array of pre-normalization per-cell totals), consumed by Task 5.

- [ ] **Step 1: Write the failing test**

Append to `tools/omic_tools/test_omic_fixes.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
cd /storage3/fs1/fuhai.li/Active/di.huang/Research/LLM/OmniCellAgent && \
/storage3/fs1/fuhai.li/Active/di.huang/cache/miniconda/omnicellagent/bin/python \
    -m pytest tools/omic_tools/test_omic_fixes.py -k normalize_cp10k -v
```
Expected: FAIL with `ImportError: cannot import name 'normalize_cp10k'`

- [ ] **Step 3: Implement normalize_cp10k**

In `tools/omic_tools/omic_fetch_analysis_workflow.py`, add immediately above `def compute_top_genes` (line 529):

```python
def normalize_cp10k(X, target_sum: float = 1e4):
    """Scale each cell to `target_sum` total counts (counts per 10K).

    Depth is confounded with group membership in these cohorts, so every
    statistic downstream must run on normalized values. `log1p` is deliberately
    NOT applied: it does not change Mann-Whitney ranks, and feeding log-scale
    values into log2(mean_a / mean_b) silently turns the fold change into a
    ratio of log means. Staying linear keeps that formula correct.

    Returns the same container type it was given, so gene-symbol columns survive.
    """
    if X is None:
        return None
    is_frame = hasattr(X, "columns")
    values = (X.values if is_frame else np.asarray(X)).astype(np.float64)
    sums = values.sum(axis=1, keepdims=True)
    scaled = values * (target_sum / np.maximum(sums, 1e-12))
    if is_frame:
        return pd.DataFrame(scaled, index=X.index, columns=X.columns)
    return scaled
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
cd /storage3/fs1/fuhai.li/Active/di.huang/Research/LLM/OmniCellAgent && \
/storage3/fs1/fuhai.li/Active/di.huang/cache/miniconda/omnicellagent/bin/python \
    -m pytest tools/omic_tools/test_omic_fixes.py -k normalize_cp10k -v
```
Expected: PASS (3 tests)

- [ ] **Step 5: Apply normalization in the workflow**

In `omic_fetch_analysis_workflow`, directly after the `gene_names` capture added in Task 1, add:

```python
    # Record pre-normalization depth for the cohort diagnostics, then normalize.
    lib_sizes = None
    if X is not None:
        raw_values = X.values if hasattr(X, "values") else np.asarray(X)
        lib_sizes = np.asarray(raw_values).sum(axis=1).astype(np.float64)
        X = normalize_cp10k(X)
        print(f"[Normalize] CP10K applied; median library size before = {np.median(lib_sizes):,.0f}")
```

- [ ] **Step 6: Run the full suite**

Run:
```bash
cd /storage3/fs1/fuhai.li/Active/di.huang/Research/LLM/OmniCellAgent && \
/storage3/fs1/fuhai.li/Active/di.huang/cache/miniconda/omnicellagent/bin/python \
    -m pytest tools/omic_tools/test_omic_fixes.py -v
```
Expected: PASS (all tests from Tasks 1-3)

- [ ] **Step 7: Commit**

```bash
git add tools/omic_tools/omic_fetch_analysis_workflow.py tools/omic_tools/test_omic_fixes.py
git commit -m "feat(omic): CP10K-normalize expression before all statistics

Depth is confounded with disease status. log1p is deliberately omitted so the
existing log2(mean/mean) fold change stays a genuine fold change."
```

---

### Task 4: Contrast guard with loader-consistent label collapsing

`_build_labels_from_metadata` gave each label-zero value its own class index; the loader collapses them all to 0. microglia_brain produced 28 classes and would have contrasted normal (1000 cells) against `Unclassified` (10).

**Files:**
- Modify: `tools/omic_tools/omic_fetch_analysis_workflow.py:240-281` (`_build_labels_from_metadata`)
- Modify: `tools/omic_tools/omic_fetch_analysis_workflow.py:722-724` (`de_gated`) and the STEP 4 group split
- Test: `tools/omic_tools/test_omic_fixes.py`

**Interfaces:**
- Consumes: nothing from Tasks 1-3.
- Produces: `select_contrast(Y, mapping) -> dict | None` with keys `ref_class`, `alt_class`, `ref_name`, `alt_name`, `excluded`, `counts`.
- Produces: `_build_labels_from_metadata` unchanged in signature; mapping semantics now collapse priority labels to 0.

- [ ] **Step 1: Write the failing test**

Append to `tools/omic_tools/test_omic_fixes.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
cd /storage3/fs1/fuhai.li/Active/di.huang/Research/LLM/OmniCellAgent && \
/storage3/fs1/fuhai.li/Active/di.huang/cache/miniconda/omnicellagent/bin/python \
    -m pytest tools/omic_tools/test_omic_fixes.py -k "priority_labels or select_contrast" -v
```
Expected: FAIL — `mapping['unknown'] == 0` assertion fails, and `ImportError` for `select_contrast`

- [ ] **Step 3: Collapse priority labels in the mapping**

In `_build_labels_from_metadata`, replace:

```python
    mapping = {label_name: idx for idx, label_name in enumerate(sorted_labels)}
```

with:

```python
    # Match CellTOSG_Loader.build_split_labels: ALL label-zero values collapse
    # into class 0. Giving each its own index produces an arbitrary contrast
    # downstream (e.g. normal vs "Unclassified" with 10 cells).
    has_priority = any(str(v).lower() in priority_lower for v in unique_vals)
    if has_priority:
        mapping = {}
        next_idx = 1
        for label_name in sorted_labels:
            if str(label_name).lower() in priority_lower:
                mapping[label_name] = 0
            else:
                mapping[label_name] = next_idx
                next_idx += 1
    else:
        mapping = {label_name: idx for idx, label_name in enumerate(sorted_labels)}
```

Note the existing `counts` comprehension iterates `mapping.values()`, which now contains duplicate zeros. Replace it with:

```python
    counts = {
        int(idx): int(np.sum((Y_mapped == idx) & valid_mask))
        for idx in sorted(set(mapping.values()))
    }
```

- [ ] **Step 4: Add select_contrast**

Add immediately after `_build_labels_from_metadata` in the same file:

```python
def select_contrast(Y, mapping: dict):
    """Choose the two-group contrast for differential expression.

    Reference is class 0 (the collapsed label-zero group, e.g. "normal").
    The alternate is the LARGEST non-reference class, so a cohort carrying many
    rare labels yields the dominant comparison rather than an arbitrary one.

    Returns None when fewer than two non-empty classes exist.
    """
    Y = np.asarray(Y)
    counts = {int(c): int(np.sum(Y == c)) for c in np.unique(Y) if int(c) >= 0}
    if counts.get(0, 0) == 0:
        return None
    non_ref = {c: n for c, n in counts.items() if c != 0 and n > 0}
    if not non_ref:
        return None

    alt_class = max(non_ref, key=lambda c: (non_ref[c], -c))
    reverse = {}
    for name, idx in mapping.items():
        reverse.setdefault(int(idx), []).append(str(name))

    return {
        "ref_class": 0,
        "alt_class": int(alt_class),
        "ref_name": "/".join(sorted(reverse.get(0, ["class_0"]))),
        "alt_name": "/".join(sorted(reverse.get(int(alt_class), [f"class_{alt_class}"]))),
        "excluded": sorted(c for c in non_ref if c != alt_class),
        "counts": counts,
    }
```

- [ ] **Step 5: Run test to verify it passes**

Run:
```bash
cd /storage3/fs1/fuhai.li/Active/di.huang/Research/LLM/OmniCellAgent && \
/storage3/fs1/fuhai.li/Active/di.huang/cache/miniconda/omnicellagent/bin/python \
    -m pytest tools/omic_tools/test_omic_fixes.py -k "priority_labels or select_contrast" -v
```
Expected: PASS (3 tests)

- [ ] **Step 6: Use the contrast in STEP 4**

The label mapping is built inside `omic_fetch_with_new_loader`, so expose it. In that function, initialize `label_mapping = None` alongside `fallback_message`, set `label_mapping = enc_mapping` where the encoded branch assigns `Y`, and set `label_mapping = _mapping` in the fallback branch.

**There are FIVE return statements in this function, not two.** Verified line numbers:

| Line | Current | Must become |
| --- | --- | --- |
| 336 | `return None, None, None, {}, False` — **only 5 values** | `return None, None, None, {}, False, label, None, None` |
| 397 | `return None, None, None, {}, False, label, None` | add `, None` |
| 516 | `return X, Y, metadata, similar_terms, True, actual_label, fallback_message` | add `, label_mapping` |
| 520 | `return None, None, None, {}, False, label, None` | add `, None` |
| 526 | `return None, None, None, {}, False, label, None` | add `, None` |

Line 336 is a **pre-existing latent crash**: it returns 5 values while the caller
at line 625 unpacks 7, so an empty-conditions query raises
`ValueError: not enough values to unpack (expected 7, got 5)`. Fixing it is
required here because this task changes the arity to 8 regardless.

Then update the unpack site in `omic_fetch_analysis_workflow` (line 625) to receive eight values:

```python
    (X, Y, metadata, similar_terms, retrieval_success,
     actual_label, label_fallback_message, label_mapping) = omic_fetch_with_new_loader(
        fetch_dict, session_dir, label=label)
```

Add a regression test for the arity bug:

```python
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
```

Then replace the `de_gated` block (around line 722):

```python
    # DE only runs when we actually have group labels. Disease-mode also
    # requires a disease query (otherwise the loader returns a single class).
    de_gated = enable_differential_expression and Y is not None and (
        actual_label != "disease" or disease_name is not None
    )
```

with:

```python
    # DE runs whenever two non-empty classes exist. The old disease-specific
    # gate is superseded: label collapsing now guarantees a meaningful class 0,
    # and select_contrast names exactly what is being compared.
    contrast = None
    if enable_differential_expression and Y is not None:
        contrast = select_contrast(Y, label_mapping or {})
    de_gated = contrast is not None
```

Inside `if de_gated:`, replace the group split:

```python
                normal_omic_feature = X[Y == 0]
                disease_omic_feature = X[Y == 1]
```

with:

```python
                normal_omic_feature = X[Y == contrast["ref_class"]]
                disease_omic_feature = X[Y == contrast["alt_class"]]
```

and replace the two `group0_count` / `group1_count` assignments with:

```python
            group0_count = int(np.sum(Y == contrast["ref_class"]))
            group1_count = int(np.sum(Y == contrast["alt_class"]))
            print(f"[DE] Contrast: '{contrast['ref_name']}' (n={group0_count}) "
                  f"vs '{contrast['alt_name']}' (n={group1_count})")
            if contrast["excluded"]:
                print(f"[DE] Excluded {len(contrast['excluded'])} other class(es) "
                      f"from this contrast: {contrast['excluded']}")
```

In the `else:` branch that prints `STEP 4 ... SKIPPED`, replace the `elif actual_label == "disease" and not disease_name:` clause with:

```python
        elif Y is not None and contrast is None:
            print("  Reason: fewer than two non-empty classes after label collapsing")
```

- [ ] **Step 7: Run the full suite**

Run:
```bash
cd /storage3/fs1/fuhai.li/Active/di.huang/Research/LLM/OmniCellAgent && \
/storage3/fs1/fuhai.li/Active/di.huang/cache/miniconda/omnicellagent/bin/python \
    -m pytest tools/omic_tools/test_omic_fixes.py -v
```
Expected: PASS (all tests from Tasks 1-4)

- [ ] **Step 8: Commit**

```bash
git add tools/omic_tools/omic_fetch_analysis_workflow.py tools/omic_tools/test_omic_fixes.py
git commit -m "fix(omic): collapse label-zero values and name the DE contrast explicitly

Matches CellTOSG_Loader.build_split_labels. microglia_brain produced 28 classes
and would have compared normal (1000 cells) against Unclassified (10)."
```

---

### Task 5: Cohort diagnostics module

**Files:**
- Create: `tools/omic_tools/cohort_diagnostics.py`
- Test: `tools/omic_tools/test_omic_fixes.py`

**Interfaces:**
- Consumes: `lib_sizes` from Task 3, `contrast` from Task 4.
- Produces: `compute_cohort_diagnostics(metadata, is_ref, is_alt, lib_sizes=None, ref_name="reference", alt_name="alternate") -> dict` with keys `verdict`, `failed_checks`, `caution_checks`, `checks`, `groups`.
- Produces: `format_diagnostics_text(diagnostics) -> str` for stdout and CSV headers.

- [ ] **Step 1: Write the failing test**

Append to `tools/omic_tools/test_omic_fixes.py`:

```python
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
    is_ref = np.array([True, False] * (n // 2))
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
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
cd /storage3/fs1/fuhai.li/Active/di.huang/Research/LLM/OmniCellAgent && \
/storage3/fs1/fuhai.li/Active/di.huang/cache/miniconda/omnicellagent/bin/python \
    -m pytest tools/omic_tools/test_omic_fixes.py -k "cohort_is or donors_are or missing_columns" -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'cohort_diagnostics'`

- [ ] **Step 3: Create the module**

Create `tools/omic_tools/cohort_diagnostics.py`:

```python
"""Cohort comparability diagnostics for the omic DE workflow.

Pure functions over cell metadata. No I/O, no loader import.

The gate WARNS; it never withholds results. Callers always produce a DE table
and attach these diagnostics to it.
"""
import numpy as np
import pandas as pd

UNUSABLE_DONOR_VALUES = {"", "nan", "none", "unknown", "na"}

PROTOCOL_IMBALANCE_FAIL = 0.30
DATASET_OVERLAP_CAUTION = 0.20
MIN_DONORS_FAIL = 10
DONOR_DOMINANCE_FAIL = 0.30
DONOR_UNUSABLE_FAIL = 0.20
DEPTH_RATIO_CAUTION_LOW = 0.67
DEPTH_RATIO_CAUTION_HIGH = 1.50


def _usable_donor_mask(series: pd.Series) -> np.ndarray:
    normalized = series.astype(str).str.strip().str.lower()
    return (~normalized.isin(UNUSABLE_DONOR_VALUES) & series.notna()).values


def compute_cohort_diagnostics(metadata, is_ref, is_alt, lib_sizes=None,
                               ref_name: str = "reference",
                               alt_name: str = "alternate") -> dict:
    """Assess whether two cell groups are comparable for differential expression.

    Args:
        metadata: per-cell DataFrame; may be missing any of the columns used.
        is_ref, is_alt: boolean arrays selecting the two groups.
        lib_sizes: per-cell totals BEFORE normalization. Optional.

    Returns a dict with `verdict` ("ok" | "caution" | "unreliable"),
    `failed_checks`, `caution_checks`, `checks`, and `groups`.
    """
    is_ref = np.asarray(is_ref, dtype=bool)
    is_alt = np.asarray(is_alt, dtype=bool)
    checks, failed, caution = {}, [], []

    def record(name, value, detail, status):
        checks[name] = {"check": name, "value": value, "detail": detail, "status": status}
        if status == "FAIL":
            failed.append(checks[name])
        elif status == "CAUTION":
            caution.append(checks[name])

    # --- protocol balance -------------------------------------------------
    if metadata is not None and "suspension_type" in metadata.columns:
        col = metadata["suspension_type"].astype(str)
        f_ref = float((col[is_ref] == "cell").mean()) if is_ref.any() else 0.0
        f_alt = float((col[is_alt] == "cell").mean()) if is_alt.any() else 0.0
        gap = abs(f_alt - f_ref)
        record(
            "protocol_balance", gap,
            f"whole-cell fraction {ref_name}={f_ref:.2f} vs {alt_name}={f_alt:.2f}; "
            "single-nucleus and whole-cell profiles are not interchangeable",
            "FAIL" if gap > PROTOCOL_IMBALANCE_FAIL else "OK",
        )

    # --- dataset overlap --------------------------------------------------
    if metadata is not None and "dataset_id" in metadata.columns:
        ref_sets = set(metadata.loc[is_ref, "dataset_id"].dropna())
        alt_sets = set(metadata.loc[is_alt, "dataset_id"].dropna())
        total = len(ref_sets | alt_sets)
        shared = len(ref_sets & alt_sets)
        overlap = (shared / total) if total else 0.0
        if shared == 0:
            status = "FAIL"
        elif overlap < DATASET_OVERLAP_CAUTION:
            status = "CAUTION"
        else:
            status = "OK"
        record(
            "dataset_overlap", overlap,
            f"{shared} of {total} datasets appear in both groups; "
            "study batch cannot be separated from biology without overlap",
            status,
        )

    # --- donor structure --------------------------------------------------
    if metadata is not None and "donor_id" in metadata.columns:
        usable = _usable_donor_mask(metadata["donor_id"])
        unusable_frac = float(1.0 - usable.mean()) if len(usable) else 0.0
        record(
            "donor_usability", unusable_frac,
            f"{unusable_frac:.1%} of cells have an unusable donor_id; "
            "these cannot be grouped and must not be treated as one donor",
            "FAIL" if unusable_frac > DONOR_UNUSABLE_FAIL else "OK",
        )

        counts, dominance = {}, 0.0
        for name, mask in ((ref_name, is_ref), (alt_name, is_alt)):
            sel = mask & usable
            donors = metadata.loc[sel, "donor_id"]
            counts[name] = int(donors.nunique())
            if len(donors):
                dominance = max(dominance, float(donors.value_counts().iloc[0] / len(donors)))

        min_donors = min(counts.values()) if counts else 0
        record(
            "donor_count", min_donors,
            f"donors per group: " + ", ".join(f"{k}={v}" for k, v in counts.items())
            + "; cells are tested as independent replicates",
            "FAIL" if min_donors < MIN_DONORS_FAIL else "OK",
        )
        record(
            "donor_dominance", dominance,
            f"largest single donor contributes {dominance:.1%} of its group's cells",
            "FAIL" if dominance > DONOR_DOMINANCE_FAIL else "OK",
        )

    # --- sequencing depth (pre-normalization, informational after CP10K) --
    if lib_sizes is not None and is_ref.any() and is_alt.any():
        lib_sizes = np.asarray(lib_sizes, dtype=float)
        med_ref = float(np.median(lib_sizes[is_ref]))
        med_alt = float(np.median(lib_sizes[is_alt]))
        ratio = med_alt / med_ref if med_ref > 0 else float("nan")
        outside = np.isfinite(ratio) and not (
            DEPTH_RATIO_CAUTION_LOW <= ratio <= DEPTH_RATIO_CAUTION_HIGH
        )
        record(
            "depth_ratio", ratio,
            f"median library size {ref_name}={med_ref:,.0f} vs {alt_name}={med_alt:,.0f} "
            "before normalization; CP10K corrects this, the size of the correction is noted",
            "CAUTION" if outside else "OK",
        )

    verdict = "unreliable" if failed else ("caution" if caution else "ok")
    return {
        "verdict": verdict,
        "failed_checks": failed,
        "caution_checks": caution,
        "checks": checks,
        "groups": {
            "reference": {"name": ref_name, "n_cells": int(is_ref.sum())},
            "alternate": {"name": alt_name, "n_cells": int(is_alt.sum())},
        },
    }


def format_diagnostics_text(diagnostics: dict) -> str:
    """Render diagnostics as plain text for stdout, CSV headers, and plot captions."""
    if not diagnostics:
        return "COHORT DIAGNOSTICS: unavailable"
    verdict = diagnostics.get("verdict", "unknown").upper()
    groups = diagnostics.get("groups", {})
    ref = groups.get("reference", {})
    alt = groups.get("alternate", {})
    lines = [
        f"COHORT DIAGNOSTICS: {verdict}",
        f"  contrast: {ref.get('name')} (n={ref.get('n_cells')}) "
        f"vs {alt.get('name')} (n={alt.get('n_cells')})",
    ]
    for item in diagnostics.get("failed_checks", []):
        lines.append(f"  FAIL    {item['check']}: {item['detail']}")
    for item in diagnostics.get("caution_checks", []):
        lines.append(f"  CAUTION {item['check']}: {item['detail']}")
    if verdict != "OK":
        lines.append(
            "  Cells are treated as independent replicates; FDR is anti-conservative."
        )
    return "\n".join(lines)
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
cd /storage3/fs1/fuhai.li/Active/di.huang/Research/LLM/OmniCellAgent && \
/storage3/fs1/fuhai.li/Active/di.huang/cache/miniconda/omnicellagent/bin/python \
    -m pytest tools/omic_tools/test_omic_fixes.py -k "cohort_is or donors_are or missing_columns" -v
```
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add tools/omic_tools/cohort_diagnostics.py tools/omic_tools/test_omic_fixes.py
git commit -m "feat(omic): add cohort comparability diagnostics

Reports protocol imbalance, dataset overlap, donor structure, and depth ratio.
Warns; never withholds results."
```

---

### Task 6: Propagate diagnostics through all six channels

`top_genes_by_fdr` already flows to `shared_data["top_genes"]` (`agent/langgraph_agent.py:1984-1986`) and from there into literature search (`:1912`) and the PDF (`:2517`). Genes from an `unreliable` cohort must not travel without their caveat.

**Files:**
- Modify: `tools/omic_tools/omic_fetch_analysis_workflow.py` (compute, print, sidecar, return dict)
- Modify: `tools/omic_tools/omic_analysis_components.py` (CSV header, volcano caption)
- Test: `tools/omic_tools/test_omic_fixes.py`

**Interfaces:**
- Consumes: `compute_cohort_diagnostics`, `format_diagnostics_text` (Task 5); `contrast` (Task 4); `lib_sizes` (Task 3).
- Produces: return-dict key `cohort_diagnostics`; file `cohort_diagnostics.json` in `session_dir`; `omic_analysis(..., diagnostics_text=None)`.

- [ ] **Step 1: Write the failing test**

Append to `tools/omic_tools/test_omic_fixes.py`:

```python
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
```

- [ ] **Step 2: Run the test to pin the contract**

This test guards `format_diagnostics_text`, which Task 5 already built, so it
passes immediately. It is here to lock the output shape that Steps 3-6 embed into
five other channels — if any later edit changes the text format, this fails first.

Run:
```bash
cd /storage3/fs1/fuhai.li/Active/di.huang/Research/LLM/OmniCellAgent && \
/storage3/fs1/fuhai.li/Active/di.huang/cache/miniconda/omnicellagent/bin/python \
    -m pytest tools/omic_tools/test_omic_fixes.py::test_format_diagnostics_text_lists_failures -v
```
Expected: PASS. If it fails, fix `format_diagnostics_text` from Task 5 before continuing.

- [ ] **Step 3: Compute and emit diagnostics in the workflow**

Add the import near the top of `omic_fetch_analysis_workflow.py`, beside `from omic_analysis_components import omic_analysis`:

```python
from cohort_diagnostics import compute_cohort_diagnostics, format_diagnostics_text
```

Inside `if de_gated:`, immediately before the `try:` that calls `omic_analysis`, add:

```python
                cohort_diagnostics = compute_cohort_diagnostics(
                    metadata,
                    Y == contrast["ref_class"],
                    Y == contrast["alt_class"],
                    lib_sizes=lib_sizes,
                    ref_name=contrast["ref_name"],
                    alt_name=contrast["alt_name"],
                )
                diagnostics_text = format_diagnostics_text(cohort_diagnostics)
                print(f"\n{diagnostics_text}\n")

                with open(os.path.join(session_dir, "cohort_diagnostics.json"), "w") as handle:
                    json.dump(cohort_diagnostics, handle, indent=2, default=str)
```

Initialize `cohort_diagnostics = None` and `diagnostics_text = ""` near `analysis_success = False` so both exist on every path. Add `import json` to the module imports if absent.

Pass the text through to the analysis:

```python
                    data_and_analysis_dict = omic_analysis(
                        comparison_name,
                        data_dict,
                        enable_plotting=enable_plotting,
                        session_dir=session_dir,
                        gene_names=gene_names,
                        diagnostics_text=diagnostics_text,
                    )
```

Add to the success return dict, next to `"analysis_success": analysis_success,`:

```python
        "cohort_diagnostics": cohort_diagnostics,
        "cohort_verdict": (cohort_diagnostics or {}).get("verdict"),
```

- [ ] **Step 4: Stamp the DE outputs**

In `omic_analysis_components.py`, extend the signature:

```python
def omic_analysis(disease_name: str, data_dict: dict, enable_plotting: bool = True, session_dir: str = None, gene_names: list = None, diagnostics_text: str = "") -> dict:
```

Immediately after `de_output_dir` is created, write the sidecar and stamp each CSV:

```python
    if diagnostics_text:
        with open(os.path.join(de_output_dir, "COHORT_DIAGNOSTICS.txt"), "w") as handle:
            handle.write(diagnostics_text + "\n")
```

Then in `perform_unpaired_differential_expression`, add the parameter `diagnostics_text: str = ""` and, in `save_dataframe`, prepend the comment header:

```python
    def save_dataframe(df, filepath):
        with open(filepath, "w") as handle:
            for line in (diagnostics_text or "").splitlines():
                handle.write(f"# {line}\n")
            df.to_csv(handle, index=False)
```

Pass `diagnostics_text=diagnostics_text` in the `perform_unpaired_differential_expression(...)` call inside `omic_analysis`.

Readers of these CSVs must use `pd.read_csv(path, comment="#")`. There are
**three** such readers in the repo — all must be updated or they break silently:

| File | Line | Read |
| --- | --- | --- |
| `tools/omic_tools/omic_fetch_analysis_workflow.py` | ~774 | `significant_genes_by_fdr.csv` |
| `tools/omic_tools/omic_fetch_analysis_workflow_microservice.py` | 161 | `significant_genes_by_fdr.csv` |
| `tools/omic_tools/omic_fetch_analysis_workflow_microservice_simple.py` | 177 | `significant_genes_by_fdr.csv` |

The two microservice files are not on the active code path, but leaving them
reading a format the writer no longer produces is a silent breakage. Add
`comment="#"` to all three. `enrichment/fc.R` also reads this filename but from
a hardcoded absolute path on a different machine — it is stale and out of scope.

- [ ] **Step 5: Caption the volcano plot**

In `create_volcano_plot`, add parameter `diagnostics_text: str = ""` and, before the figure is saved, add:

```python
    if diagnostics_text:
        verdict_line = diagnostics_text.splitlines()[0]
        plt.figtext(0.5, 0.005, verdict_line, ha="center", fontsize=8, color="firebrick")
```

Pass `diagnostics_text` from `omic_analysis` at each `create_volcano_plot(...)` call site.

- [ ] **Step 6: Surface the verdict to the agent**

In `agent/langgraph_agent.py`, in the block that stores `top_genes` (around line 1984), add immediately after:

```python
                verdict = result_content.get("cohort_verdict")
                if verdict:
                    shared_data["cohort_verdict"] = verdict
                    shared_data["cohort_diagnostics"] = result_content.get("cohort_diagnostics")
                    print(f"[SharedData] Cohort verdict: {verdict}")
```

- [ ] **Step 7: Run the full suite**

Run:
```bash
cd /storage3/fs1/fuhai.li/Active/di.huang/Research/LLM/OmniCellAgent && \
/storage3/fs1/fuhai.li/Active/di.huang/cache/miniconda/omnicellagent/bin/python \
    -m pytest tools/omic_tools/test_omic_fixes.py -v
```
Expected: PASS (all tests from Tasks 1-6)

- [ ] **Step 8: Commit**

```bash
git add tools/omic_tools/ agent/langgraph_agent.py
git commit -m "feat(omic): propagate cohort diagnostics to stdout, JSON, CSVs, plots and agent

Genes from an unreliable cohort otherwise reach PubMed search and the PDF with
no attached caveat."
```

---

### Task 7: STEP 3 gene symbols and QC reframe

`top_genes_by_expression.csv` currently emits a bare `gene_index`. Decoded, the top genes are MALAT1, three mitochondrial pseudogenes, and housekeeping genes — a data-quality signal, not disease biology.

**Files:**
- Modify: `tools/omic_tools/omic_fetch_analysis_workflow.py:677-699` (STEP 3 block)
- Test: `tools/omic_tools/test_omic_fixes.py`

**Interfaces:**
- Consumes: `gene_names` (Task 1), normalized `X` (Task 3).
- Produces: `top_genes_by_expression.csv` with columns `rank, gene_index, gene_name, mean_expression_cp10k`; return-dict key `top_gene_names`.

- [ ] **Step 1: Write the failing test**

Append to `tools/omic_tools/test_omic_fixes.py`:

```python
@requires_fixture
def test_top_genes_are_reported_as_symbols(cohort):
    """A bare integer index is not an interpretable result."""
    from omic_fetch_analysis_workflow import compute_top_genes, normalize_cp10k

    X, _, _ = cohort
    names = list(X.columns)
    idx, _values = compute_top_genes(normalize_cp10k(X).values, 20)
    symbols = [names[i] for i in idx]
    assert len(symbols) == 20
    assert all(isinstance(s, str) and s for s in symbols)
    # MALAT1 dominates both cohorts on this data.
    assert "MALAT1" in symbols
```

- [ ] **Step 2: Run test to verify it fails or passes**

Run:
```bash
cd /storage3/fs1/fuhai.li/Active/di.huang/Research/LLM/OmniCellAgent && \
/storage3/fs1/fuhai.li/Active/di.huang/cache/miniconda/omnicellagent/bin/python \
    -m pytest tools/omic_tools/test_omic_fixes.py::test_top_genes_are_reported_as_symbols -v
```
Expected: PASS — this test guards the helper contract used by Step 3. Proceed regardless.

- [ ] **Step 3: Emit symbols and relabel the step**

Replace the STEP 3 block (lines 677-699) with:

```python
    # STEP 3: Abundance QC — most-expressed genes
    # ===========================================================================
    # This is a data-quality check, NOT a discovery step. The top of this list is
    # dominated by MALAT1, mitochondrial pseudogenes and housekeeping genes.
    # Tissue markers appearing here (SFTPB/SFTPC in lung, SPP1/CD74 in microglia)
    # confirm the intended cell population was retrieved.
    print(f"\n{'='*70}")
    print(f"STEP 3: Abundance QC — Top {TOP_K_GENES} Genes by Mean Expression (CP10K)")
    print(f"{'='*70}")

    top_gene_indices, top_gene_values = compute_top_genes(X, TOP_K_GENES)
    top_gene_names = (
        [gene_names[i] for i in top_gene_indices] if gene_names else []
    )
    print(f"[QC] Top {len(top_gene_indices)} genes computed")
    if top_gene_names:
        print(f"  Top 5 genes: {top_gene_names[:5]}")
        print(f"  Top 5 mean values (CP10K): {[f'{v:.4f}' for v in top_gene_values[:5]]}")
        print("  NOTE: abundance reflects data characteristics, not disease signal.")

    times['gene_end'] = time.time()

    if len(top_gene_indices) > 0:
        top_genes_df = pd.DataFrame({
            "rank": range(1, len(top_gene_indices) + 1),
            "gene_index": top_gene_indices,
            "gene_name": top_gene_names or [""] * len(top_gene_indices),
            "mean_expression_cp10k": top_gene_values,
        })
        top_genes_path = os.path.join(session_dir, "top_genes_by_expression.csv")
        top_genes_df.to_csv(top_genes_path, index=False)
        print(f"[QC] Saved to: {top_genes_path}")
```

`compute_top_genes` must also be hardened. With a DataFrame it currently relies on
deprecated positional indexing of a Series and emits
`FutureWarning: Series.__getitem__ treating keys as positions is deprecated`.
It works on pandas 2.3.3 and breaks on a future release. Replace its body's first
statement so it always operates on an ndarray:

```python
def compute_top_genes(X, top_k=100):
    """Compute top K genes by mean expression. Returns (indices, values)."""
    if X is None or X.shape[0] == 0:
        return [], []

    # Always work on a bare ndarray: np.mean() on a DataFrame returns a Series,
    # and indexing that Series with integer positions is deprecated in pandas.
    values = X.values if hasattr(X, "values") else np.asarray(X)
    mean_expr = np.mean(values, axis=0)
    top_k_indices = np.argsort(mean_expr)[-top_k:][::-1]
    top_k_values = mean_expr[top_k_indices]

    return top_k_indices.tolist(), top_k_values.tolist()
```

Add to the success return dict, beside `"top_gene_indices": top_gene_indices,`:

```python
        "top_gene_names": top_gene_names,
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
cd /storage3/fs1/fuhai.li/Active/di.huang/Research/LLM/OmniCellAgent && \
/storage3/fs1/fuhai.li/Active/di.huang/cache/miniconda/omnicellagent/bin/python \
    -m pytest tools/omic_tools/test_omic_fixes.py -v
```
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
git add tools/omic_tools/omic_fetch_analysis_workflow.py tools/omic_tools/test_omic_fixes.py
git commit -m "feat(omic): report STEP 3 top genes as symbols and label it abundance QC

A bare gene_index is uninterpretable, and abundance rank is not disease signal."
```

---

### Task 8: `suspension_type` parameter and `num_features` correction

**Files:**
- Modify: `tools/omic_tools/omic_fetch_analysis_workflow.py:282` (`omic_fetch_with_new_loader`), `:544` (workflow signature), return dict
- Test: `tools/omic_tools/test_omic_fixes.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `omic_fetch_analysis_workflow(..., suspension_type=None)`; `omic_fetch_with_new_loader(fetch_dict, output_dir, label="disease", suspension_type=None)`.

- [ ] **Step 1: Write the failing test**

Append to `tools/omic_tools/test_omic_fixes.py`:

```python
def test_suspension_type_is_an_optional_parameter():
    """Default None must preserve current fetch behaviour exactly."""
    import inspect
    from omic_fetch_analysis_workflow import (
        omic_fetch_analysis_workflow,
        omic_fetch_with_new_loader,
    )

    outer = inspect.signature(omic_fetch_analysis_workflow).parameters
    inner = inspect.signature(omic_fetch_with_new_loader).parameters
    assert "suspension_type" in outer
    assert outer["suspension_type"].default is None
    assert "suspension_type" in inner
    assert inner["suspension_type"].default is None
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
cd /storage3/fs1/fuhai.li/Active/di.huang/Research/LLM/OmniCellAgent && \
/storage3/fs1/fuhai.li/Active/di.huang/cache/miniconda/omnicellagent/bin/python \
    -m pytest tools/omic_tools/test_omic_fixes.py::test_suspension_type_is_an_optional_parameter -v
```
Expected: FAIL — `assert 'suspension_type' in outer`

- [ ] **Step 3: Thread the parameter through**

Change the `omic_fetch_with_new_loader` signature (line 282) to:

```python
def omic_fetch_with_new_loader(fetch_dict: dict, output_dir: str, label: str = "disease",
                               suspension_type: str = None):
```

Where `conditions` is assembled (around line 331), add after the existing entries:

```python
    if suspension_type:
        # Whole-cell and single-nucleus profiles are not interchangeable. Setting
        # this constrains BOTH arms, because the loader applies query conditions
        # to the control group as well.
        conditions["suspension_type"] = suspension_type
```

Change the workflow signature (line 544) to:

```python
def omic_fetch_analysis_workflow(text=None, disease=None, cell_type=None,
                                 organ=None, tissue=None, gender=None, session_dir=None,
                                 enable_differential_expression=True, enable_plotting=True,
                                 label="disease", suspension_type=None):
```

Add to its docstring Args block:

```
        suspension_type: Optional 'cell' or 'nucleus'. Default None keeps current
            behaviour. Assay protocol is confounded with disease status in several
            cohorts (breast_cancer is 73% whole-cell in normal vs 81% nucleus in
            disease); setting this constrains both arms to one protocol.
```

Pass it at the fetch call site:

```python
        ... = omic_fetch_with_new_loader(fetch_dict, session_dir, label=label,
                                         suspension_type=suspension_type)
```

- [ ] **Step 4: Correct `num_features`**

In the success return dict, replace:

```python
        "num_features": len(top_gene_indices) if top_gene_indices else 0,
```

with:

```python
        "num_features": int(X.shape[1]) if X is not None and len(X.shape) > 1 else 0,
```

- [ ] **Step 5: Run the full suite**

Run:
```bash
cd /storage3/fs1/fuhai.li/Active/di.huang/Research/LLM/OmniCellAgent && \
/storage3/fs1/fuhai.li/Active/di.huang/cache/miniconda/omnicellagent/bin/python \
    -m pytest tools/omic_tools/test_omic_fixes.py -v
```
Expected: PASS (all tests)

- [ ] **Step 6: Commit**

```bash
git add tools/omic_tools/omic_fetch_analysis_workflow.py tools/omic_tools/test_omic_fixes.py
git commit -m "feat(omic): add optional suspension_type constraint; fix num_features

num_features reported 20 (the top-gene count) instead of the true feature count."
```

---

### Task 9: End-to-end verification

**Files:**
- Modify: none expected. Fix whatever the run exposes.
- Verify: `webapp/sessions/test_suite/test_summary.csv`

**Interfaces:**
- Consumes: everything from Tasks 1-8.
- Produces: a passing `--run-tests` run with `de_success=True` for the three cohorts that return data.

- [ ] **Step 1: Run the full unit suite**

Run:
```bash
cd /storage3/fs1/fuhai.li/Active/di.huang/Research/LLM/OmniCellAgent && \
/storage3/fs1/fuhai.li/Active/di.huang/cache/miniconda/omnicellagent/bin/python \
    -m pytest tools/omic_tools/test_omic_fixes.py -v
```
Expected: all tests PASS

- [ ] **Step 2: Run the workflow test suite**

Run:
```bash
cd /storage3/fs1/fuhai.li/Active/di.huang/Research/LLM/OmniCellAgent/tools/omic_tools && \
/storage3/fs1/fuhai.li/Active/di.huang/cache/miniconda/omnicellagent/bin/python \
    omic_fetch_analysis_workflow.py --run-tests 2>&1 | tee /tmp/run_tests_after.log
```

This takes several minutes and reads multi-GB matrices.

- [ ] **Step 3: Verify the outcome**

Run:
```bash
cd /storage3/fs1/fuhai.li/Active/di.huang/Research/LLM/OmniCellAgent && \
cat webapp/sessions/test_suite/test_summary.csv
```

Expected:
- `de_success` is `True` for `lung_adenocarcinoma`, `breast_cancer`, and `microglia_brain`. microglia_brain now runs DE because the contrast guard replaced the disease-specific gate; its contrast should read normal vs Alzheimer's Disease.
- `alzheimer_disease` and `leukemia` still show `success=False` with 0 samples. **This is correct and out of scope** — the test fixtures use disease strings absent from `disease_BMG_name`. Confirm their suggestion output now names `"Alzheimer's Disease"` and specific leukemia subtypes rather than echoing the failed string.

Run:
```bash
grep -A3 "Similar terms in database" /tmp/run_tests_after.log
```
Expected: suggestions contain `"Alzheimer's Disease"`, not `'Alzheimer disease'`.

- [ ] **Step 4: Verify diagnostics landed in every channel**

Run:
```bash
cd /storage3/fs1/fuhai.li/Active/di.huang/Research/LLM/OmniCellAgent/webapp/sessions/test_suite && \
ls breast_cancer/cohort_diagnostics.json \
   breast_cancer/differential_expression/COHORT_DIAGNOSTICS.txt && \
head -8 breast_cancer/differential_expression/significant_genes_by_fdr.csv && \
head -3 breast_cancer/top_genes_by_expression.csv && \
grep -c "COHORT DIAGNOSTICS" /tmp/run_tests_after.log
```

Expected: both files exist; the CSV opens with `#`-prefixed diagnostic lines; `top_genes_by_expression.csv` has a populated `gene_name` column; the log contains at least three `COHORT DIAGNOSTICS` blocks. breast_cancer's verdict must be `unreliable`.

- [ ] **Step 5: Commit any fixes and the verification log**

```bash
git add tools/omic_tools/
git commit -m "test(omic): verify end-to-end run produces DE for all cohorts with data"
```

---

## Self-Review Notes

**Spec coverage.** Spec items 1-9 map to tasks: 1→Task 1, 2→Task 2, 3→Task 7, 4→Task 4, 5→Task 3, 6→Tasks 5+6, 7→Task 8, 8→Task 7, 9 (incidental `num_features`)→Task 8. Spec testing items 1-6 map to Tasks 1, 2, 4, 3, 5+6, 9 respectively.

**Known deviation from the spec.** The spec's testing section item 4 originally called for asserting a median log2FC shift on real data. That was measured and found wrong (real values are +0.227 raw and +0.481 CP10K, not -0.844 → 0). The spec was corrected before this plan was written, and Task 3 uses the synthetic 2x-depth fixture instead, which yields exactly +1.0 and exactly 0.0.

**Corrections made during self-review.** Three, all verified against the code
rather than assumed:

1. The DE results filename is `unpaired_differential_expression_results.csv`
   (`omic_analysis_components.py:402`), not `unpaired_de_results.csv`. Task 1's
   assertion was corrected.
2. `compute_top_genes` relies on deprecated positional indexing of a pandas
   Series when handed a DataFrame, emitting `FutureWarning: Series.__getitem__
   treating keys as positions is deprecated`. This predates these changes but
   sits directly in Task 7's blast radius, so hardening it is folded into that
   task.
3. Task 6 Step 2 was mislabelled "verify it fails". The test passes immediately
   because Task 5 built the function; it exists to pin the text format that five
   downstream channels embed. Relabelled honestly.

**Type consistency.** `gene_names` is `list[str]` throughout. `contrast` is the dict from `select_contrast` with keys `ref_class`, `alt_class`, `ref_name`, `alt_name`, `excluded`, `counts`, used identically in Tasks 4, 5, and 6. `diagnostics_text` is `str` in Tasks 5 and 6. `lib_sizes` is a float numpy array in Tasks 3 and 5.

**Behavioural changes to expect.** microglia_brain begins producing DE where it previously skipped. Numbers will not match any historical output, because both the per-gene aggregation rule and the normalization changed. DE result CSVs now carry `#` comment headers and must be read with `pd.read_csv(path, comment="#")`.

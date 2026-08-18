# Omic Workflow: Correctness and Biological Validity Fixes

Date: 2026-08-17
Status: Approved for planning

## Problem

`python omic_fetch_analysis_workflow.py --run-tests` produced **zero differential
expression results across all five test cohorts**. The harness reported 3/5
"SUCCESS" because `success` tracks data retrieval only; `de_success` is `False`
on every row of `test_summary.csv`.

Systematic debugging of `webapp/sessions/test_suite_run.log` identified three
independent failure causes, plus a set of biological validity defects that the
crash was masking.

### Failure 1 — stale feature-axis contract (lung_adenocarcinoma, breast_cancer)

`ValueError: Length of values (533458) does not match length of index (41149)`
at `tools/omic_tools/omic_analysis_components.py:97`.

The loader now collapses transcripts to genes upstream
(`CellTOSG_Loader/data_loader.py:551`, `bmg_matrix_to_gene_matrix`), selecting one
representative transcript per gene symbol and returning a `(n_cells, 41149)`
DataFrame whose columns are HGNC symbols. `omic_analysis()` still assumes the old
BioMedGraphica entity axis and performs that same collapse itself.

Verified arithmetic:

| Source | Rows |
| --- | --- |
| `BioMedGraphica_Conn_Transcript.csv` | 412,039 |
| `BioMedGraphica_Conn_Protein_Display_Name.csv` | 121,419 |
| `omics_data` concat total | **533,458** |
| Loader output feature count | **41,149** |

The collapse moved upstream; the downstream copy was never removed. Reproduced
exactly against the saved breast_cancer session.

### Failure 2 — query vocabulary mismatch (alzheimer_disease, leukemia)

Both returned `Matched 0 samples` in under 0.5s. Queries filter
`disease_BMG_name` by exact match. `'Alzheimer disease'` is absent (the column
holds only `"Alzheimer's Disease"`); `'leukemia'` is absent (only specific
subtypes exist). This is a test-fixture defect, but it exposed Failure 3.

### Failure 3 — suggestion engine reads the wrong column

`get_suggestions()` calls `available_conditions(include_fields=["disease"])`.
`available_conditions` filters `include_fields` directly against dataframe columns
with **no `FIELD_ALIAS` resolution** (`CellTOSG_Loader/subset_builder.py:67`).
Suggestions therefore come from the raw `disease` column (154 values) while
queries filter `disease_BMG_name` (144 values).

Observed consequence: for `'Alzheimer disease'` the workflow suggested
`['Alzheimer disease', 'Alzheimer’s disease', "Alzheimer's disease", ...]` —
including the exact string that had just failed. None of the three variants exist
in the queried column. An agent following these suggestions cannot converge.

The same defect affects `cell_type` (`field_map` maps it to `cell_type`, queries
filter `CMT_name`). Only `tissue_general` is correct, by accident of having no
alias.

### Failure 4 — multi-class label encoding (microglia_brain)

`_build_labels_from_metadata` gives each label-zero value its own index, while the
loader's `build_split_labels` collapses all of them to class 0. Observed:

```
[Label] Encoded 'disease': {'normal': 0, 'Unclassified': 1, ... }   # 28 classes
[Label Check] class counts: {0: 1000, 1: 10, 2: 2, 3: 1, 4: 393, ...}
```

DE was skipped only by an unrelated disease-specific gate. Had it run, it would
have contrasted `normal` (1000 cells) against `Unclassified` (10 cells).

## Biological validity defects found during investigation

These are not crashes. They were masked by Failure 1 and would have shipped
silently once it was fixed.

**Sequencing depth is confounded with disease status.** Median library size is
681,715 in normal versus 379,731 in disease — a ratio of 0.557. Genes detected
per cell differ by only 4% (13,305 vs 12,776) while total counts differ by 44%,
which is the signature of a technical depth difference rather than reduced RNA
content. Depth normalization is the correct correction.

Magnitude, stated precisely. A pure depth difference with identical composition
would impose a uniform `log2(0.557) = -0.844` shift on every gene. Composition is
*not* identical here, so the realized effect is not a uniform downshift. Measured
on breast_cancer over genes expressed in at least one group, the median log2FC is
**+0.227 on raw counts and +0.481 after CP10K** — normalization moves it by
+0.254. The distortion is real and material; it is not a constant offset, and it
cannot be corrected after the fact by subtracting a number.

**Assay protocol is confounded with disease status, more severely than depth.**

| Cohort | normal cell/nucleus | disease cell/nucleus |
| --- | --- | --- |
| breast_cancer | 49 / 18 | 13 / 54 |
| microglia_brain | 54 / 946 | 368 / 632 |
| lung_adenocarcinoma | 918 / 77 | 995 / 0 |

Single-nucleus and whole-cell RNA-seq are not interchangeable measurements;
snRNA-seq is depleted of cytoplasmic and mitochondrial transcripts and enriched
for nuclear lncRNAs. MALAT1 is already the top-expressed gene in both cohorts. In
breast_cancer the contrast is substantially protocol, not disease. The loader's
control matching uses `match_keys: ["CMT_id", "sex_normalized",
"development_stage_category"]` and does not match on protocol, dataset, or assay.

**Study batch is confounded with disease status and cannot be corrected.**
breast_cancer has 0 of 8 `dataset_id` values shared between groups;
lung_adenocarcinoma has 2 of 31. There is no overlap with which to separate batch
from biology. This is reported, not solved.

**Cells are treated as independent replicates.** `n_disease =
disease_matrix.shape[1]` counts cells. breast_cancer's disease arm is 5 donors
with 46% of cells from one person, tested as 67 independent observations.

**Pseudobulk is not viable on the data as fetched.** Cells per donor: breast_cancer
median 1 (32 of 45 normal donors contribute exactly one cell), lung 3–5,
microglia 2. `donor_id` is `'unknown'` for 35.4% of microglia_brain cells.
Applying a ≥10-cell threshold leaves breast_cancer with zero usable normal
donors. Pseudobulk would also require changing the fetch, since
`stratified_balancing` samples cells rather than donors.

**STEP 3 output is uninterpretable.** `top_genes_by_expression.csv` emits a bare
`gene_index` with no symbol. Decoded, the top genes are MALAT1, three
mitochondrial pseudogenes (MTCYBP19, MTCO3P18, MTCO2P22), and housekeeping genes
(ACTB, EEF1A1, TMSB4X, B2M, FTL, FTH1), alongside genuine tissue markers
(SFTPB/SFTPC in lung, SPP1/CD74 in microglia).

## Decisions

The workflow architecture — fetch, describe, differential expression, enrich — is
standard and appropriate. It is retained. The work is bug fixes plus targeted
biological corrections, not a redesign.

Two decisions were made explicitly by the user and constrain the design:

1. **The CellTOSG_Loader code must not change.** All changes are confined to
   `tools/omic_tools/`.
2. **The cohort validity gate warns; it does not refuse.** A DE table is always
   produced. The known risk, accepted deliberately, is that warnings may be
   ignored downstream. This is mitigated by propagating diagnostics through six
   channels (below) rather than by withholding results.

## Scope

### 1. Gene names from the loader's own output

Capture `gene_names = list(X.columns)` at STEP 4 entry, **before**
`np.nan_to_num`, which returns a bare ndarray and destroys column labels. Pass
`gene_names` into `omic_analysis()` as a parameter.

Delete the obsolete block in `omic_analysis_components.py` (both BMG CSV reads,
`biomedgraphica_ids` insert, `BMGC_TS` filter, `mapping_dict` construction,
`Name` mapping, and `groupby('Name').mean()`).

Fallback chain: `X.columns` → `bmg_to_gene_choice.csv` in `session_dir` (column
`gene_name`) → raise with an explicit message.

Verified safe: the old path yields 41,149 unique HGNC symbols and the new path
yields 41,149 gene names. Identical cardinality and identity, so the
multiple-testing burden is unchanged. Names are clean: 0 null, 0 empty, 0
duplicates, all strings. The per-gene *value* changes from "mean across
transcripts" to "representative transcript", which is the loader's upstream
decision and not recoverable in any case.

Side effect: removes 533,458 rows of CSV parsing per run.

### 2. Suggestion vocabulary

In `get_suggestions()`, resolve field names through the loader's `FIELD_ALIAS`
before calling `available_conditions`: `disease` → `disease_BMG_name`,
`cell_type` → `CMT_name`, `tissue_general` unchanged.

Suggested `CMT_name` values are long semicolon-joined strings. This is correct —
those are the values a query must supply.

### 3. STEP 3 emits gene symbols

Add a `gene_name` column to `top_genes_by_expression.csv` beside `gene_index`,
sourced from the same `gene_names` list. Compute means on normalized values.

### 4. Contrast guard

Collapse label-zero values to class 0 as the loader's `build_split_labels` does,
using `LABEL_ZERO_LABELS_BY_LABEL_COLUMN`. Then select the contrast as **class 0
versus the largest non-reference class**. Record both class names and the list of
excluded classes in the output and in stdout.

For microglia_brain this yields normal (1000) versus Alzheimer's Disease (393)
rather than normal versus `Unclassified` (10).

**Behavior change:** the existing gate
`de_gated = ... and (actual_label != "disease" or disease_name is not None)`
is replaced by the contrast guard, which requires at least two non-empty classes
after collapsing. Cohorts such as microglia_brain that previously skipped DE will
now produce it. This is consistent with the warn-don't-refuse decision.

### 5. CP10K normalization

Normalize each cell to 10,000 total counts before splitting into groups. Do not
apply `log1p`.

Rationale: `log1p` is monotonic elementwise, so it does not change Mann-Whitney
ranks or p-values at all. It affects only mean-derived quantities. Feeding
log1p-transformed data into the existing
`np.log2((disease_mean + eps) / (control_mean + eps))` would make the result a
ratio of log-scale means — not a fold change, systematically compressed toward
zero, and silently plausible. A true 2.7x change would print as 0.32. Staying on
the linear CP10K scale keeps the existing formula correct as written.

`min_expression_threshold = 0.1` is retained unchanged. On CP10K the mean per
gene is `10000 / 41149 = 0.243`, making 0.1 a meaningful filter. Against raw
counts it filters almost nothing, which is evidence the constant was originally
written for normalized input.

Cohen's d and the variance filter also operate on the CP10K scale.

### 6. Cohort validity gate (warn mode)

Compute a `cohort_diagnostics` dict before STEP 4:

| Check | Metric | FAIL | CAUTION |
| --- | --- | --- | --- |
| Protocol balance | `abs(f_disease - f_normal)` where `f` is the fraction of a group's cells with `suspension_type == 'cell'` | > 0.30 | — |
| Dataset overlap | datasets present in both groups / total | shared == 0 | overlap < 0.20 |
| Donor count | min donors across the two groups | < 10 | — |
| Donor dominance | largest single-donor share within a group | > 0.30 | — |
| Donor usability | fraction of cells whose `donor_id` is unusable | > 0.20 | — |
| Depth ratio | median depth disease / normal, computed **before** CP10K | — | outside [0.67, 1.50] |

`donor_id` is "unusable" when, after casing and whitespace normalization, it is
empty, `NaN`, `'unknown'`, or `'none'`. Donor count and donor dominance are
computed over usable donors only.

The depth ratio is computed on pre-normalization values and is informational
after CP10K corrects it. It flags how large a correction was applied.

Verdict: `unreliable` if any FAIL, `caution` if any CAUTION and no FAIL,
otherwise `ok`. `failed_checks` is a list of human-readable strings.

Expected verdicts on the current test cohorts: breast_cancer `unreliable` (4
FAIL), microglia_brain `unreliable` (3 FAIL), lung_adenocarcinoma `caution`
(dataset overlap 2/31).

DE always runs regardless of verdict. Diagnostics propagate to **six** channels:

1. stdout
2. the workflow return dict, key `cohort_diagnostics`
3. a sidecar `cohort_diagnostics.json` in `session_dir`
4. a comment header line in each DE CSV
5. the volcano plot subtitle
6. `shared_data`, so downstream agents and `_compile_pdf` receive it

> **POST-IMPLEMENTATION CORRECTION — channel 6 does not work, and the premise
> below is false.** The final whole-branch review established that the
> `shared_data` path is unreachable: `agent/langgraph_agent.py` registers the
> agent as `"OmicMiningAgent"`, not `"OmicAnalysis"`, and unknown names are
> coerced to `"GoogleSearcher"` before the extraction code runs. Independently,
> `SubAgent.execute` returns a **string**, so the `isinstance(result_content,
> dict)` guard is always False. Critically, `shared_data["top_genes"]` was
> **equally dead before this branch** — so the justification below describes a
> path that never carried anything. Repairing it requires capturing the tool's
> return dict in `SubAgent.execute` rather than parsing the LLM's prose, which
> is beyond this plan's agreed scope. **Deferred by explicit user decision.**
> Five channels work and carry the verdict; the agent and the PDF do not.

Channel 6 matters because `top_genes_by_fdr` already flows to
`shared_data["top_genes"]` (`agent/langgraph_agent.py:1984-1986`) and from there
into literature search (`:1912`) and the PDF (`:2517`). Genes from an
`unreliable` cohort otherwise become PubMed queries and report content with no
attached caveat.

### 7. `suspension_type` as an optional parameter

Add `suspension_type` to `omic_fetch_analysis_workflow()`, defaulting to `None`,
passed through to the loader query conditions when set. Default `None` preserves
current fetch behavior exactly.

The protocol imbalance is always reported by the validity gate. With the default,
the artifact is **reported but not removed**; removing it requires the caller to
set the parameter. This follows from the warn-don't-refuse decision, since
constraining the query by default would change which data is returned.

### 8. STEP 3 reframed as abundance QC

Relabel STEP 3 output and its report section as an abundance/quality check rather
than a discovery result. MALAT1 and mitochondrial pseudogenes at the top indicate
data characteristics; SFTPB/SFTPC and SPP1/CD74 confirm the intended cell
population was retrieved. Both are useful. Neither is disease biology, and the
current framing invites that misreading.

### 9. Incidental

`num_features` in the return dict is `len(top_gene_indices)`, reporting 20 rather
than the true feature count. Corrected to the actual number of genes.

## Non-goals

- **Pseudobulk aggregation.** Blocked by cell-per-donor counts, `donor_id`
  unavailability, and the need for fetch changes. Would leave breast_cancer
  unanalyzable.
- **`dataset_id` batch correction.** Impossible with zero overlapping datasets.
  Reported only.
- **Any change to `CellTOSG_Loader`.** Explicitly excluded.
- **Test fixture repair** (`'Alzheimer disease'` → `"Alzheimer's Disease"`,
  `'leukemia'` → `'acute myeloid leukemia'`) and the harness `success` criterion.
  Tracked separately; these are test-side, not pipeline-side.

## Testing

1. **Regression, Failure 1.** Reconstruct the failing call from the saved
   breast_cancer session; assert `omic_analysis` completes and returns 41,149
   rows with `Name` populated from gene symbols.
2. **Regression, Failure 3.** Assert `get_suggestions({'disease': 'Alzheimer
   disease'})` returns `"Alzheimer's Disease"` and never returns a value absent
   from `disease_BMG_name`.
3. **Regression, Failure 4.** Given the microglia_brain label vector, assert the
   selected contrast is normal versus Alzheimer's Disease and that excluded
   classes are recorded.
4. **Normalization.** Two assertions. First, every cell sums to 10,000 after
   normalization — confirms it was applied, but is tautological on its own.
   Second, the substantive check, on a **synthetic** matrix where one group is
   the other scaled by exactly 2x with identical composition: median log2FC is
   exactly +1.0 on raw values and exactly 0.0 after CP10K. This isolates the
   depth artifact and proves normalization removes it.

   Do not assert a fixed median log2FC on real cohort data. On breast_cancer the
   real values are +0.227 raw and +0.481 CP10K; both are confounded with genuine
   compositional differences and neither is a meaningful target.
5. **Validity gate.** Assert breast_cancer and microglia_brain yield
   `unreliable`, lung_adenocarcinoma yields `caution`, and that
   `cohort_diagnostics` is present in the five working channels (see the
   correction above — channel 6 is unreachable and was deferred).
6. **End to end.** Re-run `--run-tests`; assert `de_success` is `True` for
   lung_adenocarcinoma, breast_cancer, and microglia_brain.

## Risks

- **Warnings ignored downstream.** Accepted by the user. Mitigated by five-channel
  propagation, not eliminated. If report readers overlook the caveat, they will
  read `unreliable` results as findings.
- **More cohorts now produce DE.** Replacing the disease-specific gate with the
  contrast guard means queries that previously skipped STEP 4 now produce tables.
  Intended, but it increases the surface area over which the warning risk applies.
- **Representative-transcript library size.** CP10K size factors are computed over
  the 41,149 representative transcripts, not the full transcriptome, since the
  loader has already selected. This is standard practice on a gene-by-cell matrix
  but is an approximation.
- **Results will differ from any historical output.** Both the per-gene
  aggregation rule and the normalization change. Prior numbers are not
  reproducible and should not be used as a comparison baseline.

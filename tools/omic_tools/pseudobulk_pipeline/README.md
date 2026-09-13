# Donor-level CellTOSG pseudobulk pipeline

This separate workflow implements the approved AD astrocyte analysis. It does
not import or modify the old omic workflow and refuses to reuse an output
directory. Existing session results are not analysis inputs. The optional old
session audit reads its current metadata for comparison only.

## Start here

`omic_workflow.py` is the main entry point. Its `execute()` function follows
seven numbered scientific steps; its grouped CLI arguments are defined in the
same file. Read `pipeline_stages.py` for the raw-cohort and initial-model stages.

```bash
/storage3/fs1/fuhai.li/Active/di.huang/cache/miniconda/omnicellagent/bin/python \
  tools/omic_tools/pseudobulk_pipeline/omic_workflow.py \
  --disease "Alzheimer's Disease" --organ brain --cell-type astrocyte \
  --enrichement-model disease-only --session-id alzheimer_test
```

Both `--enrichement-model` and `--enrichment-model` work. Use `--help` for all
component options. All generated plots and analysis outputs live under `webapp/sessions/`.
The session must be new; previous results are never reused as output.
Omitting `--session-id` creates a timestamped session there. `--output-dir`
is accepted only when it points inside `webapp/sessions/`. `--source-run` explicitly reuses an existing completed raw run instead
of retrieving again. `--prepare-only` stops after raw donor aggregation.

| Module | Read it for |
|---|---|
| `omic_workflow.py` | CLI arguments and the complete numbered analysis sequence |
| `run_support.py` | Validation, persistent cache defaults, paths and run manifest |
| `pipeline_stages.py` | Cohort selection, raw aggregation, initial DESeq2 stage |
| `metadata.py` | Exact matching, composite donor identity, annotation conflicts |
| `retrieval.py` | HGNC selection, raw CellTOSG reads, representative BMG columns |
| `aggregation.py` | Metacells × genes → genes × donors, raw integer sums |
| `selected_model.py` / `selected_model.R` | Default: prepare the cohort and fit exactly one selected model |
| `deseq2_analysis.R` | Initial donor contrasts for the optional full audit; shared eligibility helpers |
| `enrichment.py` | Significant-gene lists, Enrichr requests and complete tables |
| `enrichment_plots.py` | Figures regenerated from saved enrichment tables |
| `plots.py` | Donor expression PCA, volcano and gene-expression figures |

By default, `--enrichment-model` selects the **only model fitted**, and its
significant genes supply the enrichment lists. There are no preliminary AD or
sex-comparison fits. `selected_model.R` reuses the original eligibility and
model-fitting functions, preparing median-of-ratios factors without a DE fit.
The fit reads the newly saved factor CSV, matching the historical normalization
precision; raw counts are never rounded or normalized before aggregation.

Add `--full-sensitivity` to restore the full audit: initial three comparisons,
then all models from the sibling `donor_sensitivity` package. With `--source-run`,
that audit requires a raw run whose initial comparisons already completed.

The selected-only path deliberately preserves the previous complete-case donor
cohort (including study, age and sex annotations even for disease-only), so the
speed improvement does not change sample eligibility. Shared-study and study
removal models still re-filter and re-normalize their subsets as before.

`--raw-plots` adds donor PCA/volcano/violin plots from the selected fit. It exports
normalized counts and VST from that same fitted object; no extra DE fit occurs.

| User model name | Saved directory | Formula / cohort |
|---|---|---|
| `disease-only` | `full_A` | `~ disease`, full complete-case cohort |
| `study-disease` | `full_B` | `~ study + disease`, same donors |
| `fully-adjusted` | `full_C` | `~ study + age + sex + disease`, same donors |
| `shared-studies-adjusted` | `shared_C` | Fully adjusted, studies with both groups |

Session layout:

```text
webapp/sessions/SESSION_ID/
  workflow_manifest.json
  raw_pipeline/inputs/                # raw donor counts and metadata
  raw_pipeline/comparisons/           # only with --full-sensitivity
  models/models/full_A/               # only the chosen model by default
  models/cohort/                      # donor eligibility and normalization audit
  models/models/OTHER_MODELS/          # additional models only with --full-sensitivity
  models/summary/                     # selected-model summary; full audit adds stability plots
```

The default selected model uses fixed-family BH after count filtering, matching
the sensitivity audit. Optional initial models use DESeq2 adaptive independent
filtering; sensitivity models use fixed-family BH after count filtering. This distinction is intentional and is
preserved by the refactor. Plot code never refits differential expression.

## Runtime and compatibility

R defaults to the persistent installation documented in `R_ENVIRONMENT.md`.
`OMIC_RSCRIPT`, `PSEUDOBULK_RSCRIPT`, or `--rscript` can override it. Default
Matplotlib, Numba, and working-directory caches are persistent under
`dataset_outputs/pseudobulk_runtime_cache/`; explicit environment overrides remain
supported. No default depends on a runtime installation or cache under `/tmp`.

The existing `omic_pseudobulk_workflow.py` CLI remains available for the initial
three-comparison pipeline; it now imports `pipeline_stages.run_pipeline`.
`workflow.py` has been renamed, so internal imports should use `pipeline_stages`.

The approved exclusion file remains
`dataset_outputs/pseudobulk_source_qc/approved_donor_exclusions.csv`. It removes
11 reviewed control donor keys containing fractional-count metacells. Exclusions
are explicit and audited; counts are never rounded to force DESeq2 eligibility.

## Retrieval and donors

- Match `disease_BMG_name`, `CMT_name`, `tissue_general` and optional `tissue`
  exactly, ignoring case. The disease pool and normal pool share the same
  tissue/cell-type constraints.
- Use **all** qualifying metacells with known `(source, dataset_id, donor_id)`.
  There is no metacell sample cap in this pipeline. Identical donor labels in
  different source/dataset pairs are independent under the confirmed rule.
- Preserve original matrix-file/row pointers and reject duplicate pointers.
- Intersect `bmg_gene_index.csv` with approved HGNC protein-coding symbols.
  Import the installed CellTOSG expression reader and representative-gene
  collapse function. Bounded raw-read chunks reduce memory usage; representative
  candidates are selected once across the complete cohort, not per chunk.
- Reject negative, fractional or nonfinite counts. Aggregate raw selected gene
  counts by donor before any normalization and verify every gene's count total.
  These checks establish conservation on the retained gene axis, not across
  discarded BMG columns.

Donor sex, numeric age and disease are resolved only when there is one observed
valid value. Conflicting values remain missing; every observed annotation is
retained in metadata, and conflicts/missing values are listed in a separate
audit. No majority, mean age or sex imputation is performed. The pipeline cannot
independently reconstruct raw-cell donor purity from the exported metacell
arrays; this relies on the input dataset's construction and confirmed identities.

## Exactly three comparisons

| Output directory | Positive log2FoldChange | Requested model |
|---|---|---|
| `ad_vs_control` | Higher in AD than control | `~ study + age + sex + disease` |
| `ad_male_vs_female` | Higher in male than female AD donors | `~ study + age + sex` |
| `control_male_vs_female` | Higher in male than female control donors | `~ study + age + sex` |

Models use complete-case donors; missing/conflicting covariates, invalid age,
wrong disease group and zero libraries are explicitly excluded. Constant
nuisance terms may be omitted with an audit record. A confounded/saturated model
or fewer than three donors per compared group produces an explicit skipped
status, rather than a silently simplified analysis.

DESeq2 applies median-of-ratios size factors (`type="ratio"`). Genes are kept for
testing when their raw count is at least 10 in at least 3 included donors
(configurable). If no genes have a positive geometric mean across included
donors, ratio normalization is unavailable and the comparison is skipped;
there is no alternative normalization fallback. Wald tests use explicit
contrasts, BH adjustment and DESeq2 independent filtering. Automatic outlier
count replacement is disabled; Cook's filtering remains active. Complete
results retain all input genes and explain untested/filtered entries.

## Outputs

`inputs/` contains raw `pseudobulk_counts.csv` (genes × donors), `donor_metadata.csv`,
`metacell_to_donor.csv`, retrieved metadata, HGNC reference snapshot, selection,
conflict, previous-session and count-conservation audits. `retrieval/` preserves
source and compact BMG mappings plus final source-column choices.

Each `comparisons/<name>/` contains:

- `status.json`, `donor_exclusions.csv`, `design_matrix.csv`, `fit.log`;
- complete `results.csv`, `size_factors.csv`, `normalized_counts.csv`;
- donor-level `vst_expression.csv`, package versions and R session information;
- `plots/`: PCA, volcano and gene-expression PNG/PDFs, plotted donor IDs,
  underlying point tables, and a plot manifest;
- `enrichment/`: all/up/down gene lists, raw API responses, complete library
  tables, background/request provenance and direction-specific KEGG/category plots.

Primary PCA uses the 500 most-variable VST genes, regardless of DE significance.
VST reuses the fitted dispersion trend (`blind=FALSE`). Violin points represent
donors, displaying `log2(DESeq2-normalized count + 1)` with saved DESeq2 results.
Plotting never reruns DESeq2, limma or Mann–Whitney tests. Normalized/VST values
are not themselves residualized for study, age or sex.

Enrichr uses the existing 14 libraries and separate all/up/down significant sets
(default `padj < 0.05`, with no top-gene cap). Custom background remains deferred.
The recorded background is the service default: official help describes a nominal
20,000 genes, but the API does not expose the actual runtime universe membership
or size. The tested gene list is never represented as a custom background used.
Only gene symbols are submitted to Enrichr; donor metadata are not uploaded.

`run_manifest.json` records settings and stage status. Exit codes are 0 for a
successful/prepared run, 2 for partial completion, and 1 for an execution error.
New files are retained on failure for review; no automatic overwrite/restart occurs.

## Validation

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
MPLCONFIGDIR=/storage3/fs1/fuhai.li/Active/di.huang/Research/LLM/OmniCellAgent/dataset_outputs/pseudobulk_runtime_cache/matplotlib NUMBA_CACHE_DIR=/storage3/fs1/fuhai.li/Active/di.huang/Research/LLM/OmniCellAgent/dataset_outputs/pseudobulk_runtime_cache/numba \
PSEUDOBULK_RSCRIPT=/storage1/fs1/fuhai.li/Active/di.huang/cache/R/omic-deseq2/bin/Rscript \
R_LIBS=/storage1/fs1/fuhai.li/Active/di.huang/cache/R/omic-deseq2/lib/R/library \
R_LIBS_USER=/storage1/fs1/fuhai.li/Active/di.huang/cache/R/omic-deseq2/lib/R/library \
/storage3/fs1/fuhai.li/Active/di.huang/cache/miniconda/omnicellagent/bin/python \
  -m pytest tools/omic_tools/pseudobulk_pipeline/tests -q
```

Methods: [DESeq2 paper](https://doi.org/10.1186/s13059-014-0550-8),
[DESeq2 documentation](https://bioconductor.org/packages/release/bioc/vignettes/DESeq2/inst/doc/DESeq2.html).

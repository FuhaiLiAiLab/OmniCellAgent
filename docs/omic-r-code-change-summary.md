# AD Astrocyte Workflow: Code Change Summary

## Scope and review status

At your request, the Python workflow now calls the existing R DE implementation,
and enrichment analysis and volcano plots consume the same R-based DE results.
Python orchestrates the workflow and online enrichment; R performs DE and drawing.
The analysis unit remains the metacell. No donor-level DE model was introduced.

The changes were integrated into the user's local `revision` branch during
development. The user has now authorized publication to the GitHub `revision`
branch for review. Integration into any other branch remains for Di to review and
perform. The current code endpoint is `c9bb6cc`; the documentation handoff follows it.

Post-verification display edits are included in `c9bb6cc`: at the user's request,
panel A labels only the top 10 Up genes by FDR and all Down genes meeting the
existing coloring thresholds (five Down genes in test5), instead of labeling the
overall FDR top 20. Count annotations now say logFC. Following a display review,
PCA points now use size=1.7, alpha=0.25 and fixed-seed (42) shuffled drawing order
to reduce group occlusion in the single overlay. Coordinates and group memberships
are unchanged; withr::with_seed restores the RNG state after ordering.
The B correlation plot selection remains unchanged. test5 was redrawn from its
saved DE state, including standalone PNG/PDF, ABCD and the A source-gene CSV.
The observer verified 10 Up / 5 Down labels, no NS labels, and the current PCA
size/alpha, all sample coordinates preserved after realignment, and changed drawing order;
nine expression/DE/state files retained their hashes and modification times.
Old figures were backed up under `webapp/sessions/test5/plot_revision_labels_alpha/previous/`.
Evidence is in that revision directory's `render.log` and `verification.json`.
The subsequent PCA overlay revision and its backup/evidence are under
`webapp/sessions/test5/plot_revision_pca_overlap/`; only PCA and ABCD PNG/PDF were
published from that second redraw, and the nine protected inputs/DE/state files
again retained their content and mtimes.
The earlier full-chain run below applies to committed `52d6955`; these subsequent
display-only edits were validated by redraw, not another DE/online-enrichment run.

## Commit map and actual changes

| Task / follow-up | Commit | Review focus |
|---|---|---|
| Tasks 1–2: R DE extraction and Python integration | `7bac9cc` | Reuse existing limma DE, preserve five Python-compatible CSV outputs, enforce reader dtypes including empty tables, propagate R process failures, validate real Enrichr responses and statuses. Task 1 is included here; it does not have a separate feature commit. |
| Task 3: plotting migration | `425257a` | Move case-study plotting into standalone `run_casestudy.R`, remove workflow Python drawing code, add manifests and result collection. **This commit also introduced the enrichment bar-chart defect described below.** |
| Task 4 intermediate implementation | `4f915d2` | Add runtime/usage documentation and A/B panel source-gene validation. Also explicitly set orientation in the newly added bar charts; that plotting implementation was subsequently removed. This was an actual commit, not merely an uncommitted draft. |
| Task 4 final enrichment plotting decision | `5ab223c` | Remove the newly added KEGG/Reactome bar-chart implementation; invoke the original `kegg_simple.R` plotting logic separately for all/up/down and update output collection/tests/documentation. |
| Final title adjustment | `52d6955` | Label PNG/HTML plot titles with Upregulated or Downregulated, inferred from the input group directory. No enrichment values or group selection changed. |
| Final case-study display adjustment | `c9bb6cc` | Restrict volcano gene labels to colored genes, change count annotations to logFC, reduce PCA point size/opacity and shuffle only drawing order with a fixed seed. Update the source-gene/coordinate observer. |

`926f6b0` records pre-existing local pipeline settings and is not an additional
DE-method change. For the net code diff, review `926f6b0..c9bb6cc` (or
`926f6b0..revision` to include the documentation handoff);
review the commits above individually for provenance. Unrelated local changes to
`casestudy.Rmd`, local AGENTS instructions and generated HTML files are not included.

Main files:

- `tools/omic_tools/omic_fetch_analysis_workflow.py`: retained public entry point,
  contrast propagation, R dispatch, stage statuses and report collection.
- `tools/omic_tools/omic_analysis_components.py`: replace Python DE with an adapter
  to the existing R implementation; retain enrichment orchestration.
- `tools/omic_tools/run_casestudy.R`: DE, compatible exports, saved state, and A–D
  plots in one executable R file. The old panels Rmd remains a reference, not a
  runtime dependency.
- `tools/omic_tools/subprocess_r.py`, `de_results_io.py`, `enrichr_client.py`,
  `r_plotting.py`: process handling, typed CSV reads, validated enrichment responses,
  manifest-based publication and collection.
- `enrichment/kegg_simple.R`: retained independent script and original plot logic.
- `tools/omic_tools/tests/`, `docs/omic-r-pipeline.md`: validation and operating guide.

## DE and output compatibility

The implementation reuses the existing R pipeline: linear CP10K inputs are
transformed to log2(CP10K + 1), followed by the existing limma model and moderation.
It does not preserve the old Python rank-test numerical results; the reference is
the original R pipeline. CSV compatibility means the established filenames,
column order and types, including explicitly typed empty-table reads.

The seven compatible columns are `Name`, `log2_fold_change`, `effect_size`,
`p_value`, `FDR`, `is_significant`, and `abs_log2_fc`. `effect_size` retains the old
linear-expression Cohen's d definition. FDR is calculated over tested genes as in R.

The existing gene-list truncation remains: all uses the lowest-FDR significant
genes, capped at 1,000; up and down each use significant genes of the corresponding
logFC sign, ranked by P value and independently capped at 1,000. All is not the
concatenation of up and down.

## Enrichment plotting defect and final scope decision

In `425257a`, the newly written R bar-chart block used numeric `score` and numeric
`row` coordinates without specifying `geom_col` orientation. Bars were drawn with
the wrong geometry relative to horizontal pathway labels. **This did not swap the
up/down gene sets**, but made the charts misleading. The defective code path was
used for KEGG/Reactome × all/up/down × PNG/PDF: all 12 of those outputs were affected.
It did not compute or alter DE results, enrichment CSV values, FDRs or hit genes.
Case-study A–D and the pre-existing KEGG dotplot/combined plot used other code paths.

The defect was discovered when the user requested the actual up/down enrichment
figures, **before Task 4 was formally started**, not during Task 4's proposed real-data
replay. Task 3 checks covered generation, manifests, file formats and dependencies,
but missed this visual error. The numerical/file tests did not assert bar geometry.

`4f915d2` initially fixed orientation. The final decision in `5ab223c`, following
the user's instruction to reuse existing code, was to **remove that added plotting
block entirely**, rather than retain it with a patch.

Current behavior calls the original two-argument `kegg_simple.R` entry point for
each group:

- all: `plots/kegg_dotplot.*` and `plots/pathway_combined_plot.*`;
- up: the same filenames under `plots/up/`;
- down: the same filenames under `plots/down/`.

Each has PNG and HTML output. KEGG dotplots select the top 20 terms; combined
bar charts select five terms per category: GO BP, CC, MF, KEGG and DisGeNET.
These are ranking rules, not a guarantee that every displayed term has FDR < 0.05.
PNG export uses a white background; the original plotting selection, ordering and
colors are retained. HTML dependency directories must accompany the HTML files.

**Scope reduction requiring review:** separate Reactome bar charts are no longer
delivered. Reactome enrichment remains calculated and available in CSV/raw results,
but the original R script does not plot Reactome. The former per-library KEGG/Reactome
PNG/PDF bar products are retired. Old generated artifacts are not treated as current
report outputs. The two legacy Python-style volcano plots were also explicitly
retired; the case-study volcano is retained.

## Validation performed and limits

All computational validation ran inside existing Slurm allocations. Later redraws
used saved matrices/DE state/enrichment CSVs and did not refit DE or re-query Enrichr.

- **Task 1 numerical parity:** compare original R and extracted R DE on the same
  real AD matrices; align gene identities and compare logFC, P.Value and adj.P.Val.
  The case study has 18,621 tested genes and 13,467 significant genes. CSV schema,
  empty outputs, contrast reversal, invalid-input exits and independent Cohen's d
  checks were covered in the earlier evidence.
- **Task 2 integration:** Python-to-R adapters and workflow failure handling were
  tested separately, including nonzero exits, timeouts, missing outputs and avoiding
  stale results. Real online enrichment was exercised for three 1,000-gene lists and
  14 libraries: 3 POST + 42 GET requests returned HTTP 200. Empty-response and network
  failure behavior were checked separately; mocks are not counted as online evidence.
- **A/B source-gene audit:** observed actual volcano annotation-layer genes and
  corrgram label callbacks, compared order/expression input/DE statistics to CSVs,
  and compared the previously delivered CSVs. No mismatch was found. Redraw tests
  verified that 94 existing inputs/result files retained their hashes and mtimes.
- **Task 4 pre-decision plotting regression:** `19 passed in 230.84s`, including the
  A/B audit and the then-current bar-chart orientation fix. This predates the final
  decision to remove the custom bars and is not mislabeled as the final-scope suite.
- **Final original-R plotting scope:** `24 passed in 32.85s`, no failures/skips or
  warning summary. Covers three-group dispatch, staged publication, failure/stale
  result handling, original plots, HTML dependencies and real existing CSV redraws.
  The 45 enrichment CSV/gene-list inputs retained their content and mtimes.
- **Title-only follow-up:** regenerated up/down PNG/HTML from existing CSVs;
  checked the titles and inspected the combined PNGs. The 24-test suite was not
  rerun after this small title-only change; do not attribute its result to `52d6955`.

**End-to-end verification completed on 2026-09-16:** after explicit authorization
to run fresh retrieval/DE/online enrichment, commit `52d6955` was exercised through
the actual Python CLI `__main__` (via runpy to also save its returned result), with
no mocks or replaced stages. Parameters: Alzheimer's Disease, astrocyte, brain,
label=disease, sample-size=1000, plotting enabled, COMPOSITE=true, PSEUDOBULK=false.
The new session is `webapp/sessions/ad_astrocyte_e2e_20260916T215810Z`.

Actual results: normal 1,000 / AD 671 metacells, 18,621 tested / 13,467 significant
genes; all/up/down each 1,000 submitted genes matching the exported DE subsets;
3 POST + 42 GET requests returned HTTP200 across 14 libraries per group. The five
DE CSV schemas/dtypes, saved response bodies, output manifests, HTML local assets
and final report paths were checked. The CLI returned success=True,
analysis_success=True, kegg_success=True, and the returned DE/enrichment/plot stage
statuses passed the verifier. Outputs: 4 case-study PNGs + 6 enrichment PNGs and
6 HTML files. Workflow timing was 365.24 seconds. This closes the previously noted
final-chain execution gap for this case study, not for every possible query.

The run was not warning-free: three Plotly messages reported
`line.width does not currently support multiple values` during enrichment plotting.
Processes completed and files were generated; full interactive rendering behavior
has not been manually verified. Existing cohort diagnostics reported `caution`
(only 4 of 26 datasets overlap between groups, and a pre-normalization depth ratio
of approximately 0.6605). Technical workflow completion does not resolve these
cohort comparability limitations. No sex-based exclusions were introduced.

Other limits: not every PDF was opened/rendered, and full browser interaction of
every HTML widget was not tested. Case-study violin has a standalone PDF, not PNG;
only A/B gene CSVs are exported. No new standalone C/D source-gene exports were added.

Runtime used: base Python 3.13.9, NumPy 2.5.3, SciPy 1.18.1; R 4.5.2, limma 3.66.0,
ragg 1.4.0. The SciPy/NumPy warning was resolved before plotting. The R library path
includes a task-local rebuilt ragg and the existing shared library; other machines
must configure equivalent dependencies. Local runtime packages are not committed.

## Sex analysis: deliberately deferred

The advisor's instruction was: “As an example, don’t worry sex factor information,
let’s just compare Ad vs normal. Then we can parallelize the revision first.”

Accordingly, this delivery remains AD versus normal. Sex-specific DE/plot integration
is deferred under that instruction, not omitted accidentally and not described as
waiting for a new metacell-versus-donor decision. No sex-based DE was run, no sex
labels were changed, and **none of the 63 conflict-associated metacells were removed**.

A separate read-only metadata audit found six donor identifiers with both female
and male labels among 63 case-study metacells. The session and shared metadata agree;
this is an upstream annotation consistency issue, not a cross-file mismatch. It has
not been resolved against original study donor records and is not part of this
AD-versus-normal code change.

## Evidence locations

Paths below are relative to
`webapp/sessions/alzheimer_test4/unified_pipeline_validation/`:

- `2026-09-15/review-evidence/`, `2026-09-15/task2-review/`: DE parity and interface evidence.
- `2026-09-15/online-enrichr-review/`: actual HTTP responses, counts and failure controls.
- `2026-09-16-panel-audit/task4-final.log`: 19-test pre-decision regression.
- `2026-09-16-panel-audit/original-kegg-final.log`: final-scope 24-test regression.
- `2026-09-16-panel-audit/original-kegg-final/test_existing_enrichment_uses_0/`: original-R three-group redraws.
- `2026-09-16-labeled-enrichment-simple/`: final Upregulated/Downregulated PNG/HTML.
- `2026-09-16-sex-metadata/`: separate read-only metadata audit and scripts.

These data/artifacts and internal planning notes are local and are not part of the
code commits. The committed operating guide is `docs/omic-r-pipeline.md`.

The final full-run evidence is outside the historical validation directory:
`webapp/sessions/ad_astrocyte_e2e_20260916T215810Z/e2e_audit/` contains
`provenance.json`, `R-sessionInfo.txt`, `workflow.log`, `workflow_result.json`, and
`verification.json`. The reusable verifier is
`webapp/sessions/alzheimer_test4/unified_pipeline_validation/2026-09-16-e2e/run_e2e.py`;
it requires an existing Slurm allocation and creates a new session per invocation.

# How this folder was extracted from the agent

This file records how the mini pipeline was cut out of the agent's code, where it still differs from the agent, and how equality was verified. For how to run it and what it does, see `README.md`.

The source is the agent's omic pipeline on branch `omic-shared-controls` at commit `bb92fab`.

The check folder `webapp/sessions/AD_20261030_refactor_check/` exists only in the cluster checkout, because `webapp/sessions/` is gitignored. It holds the comparison scripts (`compare_outputs.py`, `check_trim.py`, `check_lines.py`), both runs' outputs and `COMPARISON.md`.

## Trimming rule

- Every function the run calls is kept, with the statements that do the work, in the original order.
- Failure handling inside that code is kept: Enrichr, R and diagnostics errors, and the no-data result.
- Removed: functions never called; options this call never sets, with their parameters, the branches they select, and statements that only serve them; and the paths for a missing input or a missing session folder.
- Option values that reach an output file are kept as constants.
- Apart from that, executed lines change only for the adaptations listed below.

## Files and their origin

| File | From `tools/omic_tools/` (or `enrichment/`) at bb92fab | Change |
|---|---|---|
| `run_ad_microglia.py` | `agent/langgraph_agent.py:omic_analysis_tool` | the tool's call and result post-processing for this cohort |
| `settings.py` | `utils/path_config.py` lookups in `configs/paths.yaml` | the four paths the run reads |
| `omic_fetch_analysis_workflow.py` | same name | trimmed (1,550 → 839 lines) |
| `omic_analysis_components.py` | same name | trimmed (577 → 500) |
| `donor_sampling.py` | same name | trimmed (320 → 204) |
| `enrichr_client.py` | same name | `read_enrichment_results` removed |
| `cohort_diagnostics.py` | same name | `keep_unknown_donor` option removed |
| `celltosg_runtime_adapter.py` | same name | metadata argument required; HGNC path from settings |
| `query_value_lists.py` | same name | flat import; "no filter requested" branch removed |
| `r_plotting.py` | same name | flat import only |
| `subprocess_r.py`, `de_results_io.py` | same name | unchanged |
| `run_casestudy.R` | same name | trimmed (852 → 736) |
| `kegg_simple.R` | `enrichment/kegg_simple.R` | arguments required; legacy branch removed |

`cell_type_lookup.py` is not included. The tool runs it as a gate before the workflow. For today's v3 data, microglia passes the gate, which writes no files. See *Where the mini differs from the agent*.

## Adaptations (changes needed to run without the agent)

Original line numbers are at bb92fab, in `tools/omic_tools/` unless noted.
- **Paths:** `get_path(...)` from `utils/path_config.py` is replaced by constants in `settings.py`:
  - `omic_fetch_analysis_workflow.py:48-50`: the repository-root `sys.path` entry and `get_path` import;
  - `omic_fetch_analysis_workflow.py:53` (`external.omnicell_root`), `:411` (`external.omnicell_data_root`) and `:1190` (`enrichment.kegg_script`);
  - `omic_analysis_components.py:11-13`: `sys.path` entry and `get_path` import.
- **Unused module constants deleted:**
  - `omic_analysis_components.py:27` `BMG_DIR` and `:29` `OUTPUT_DIR` are gone, along with their fallbacks for a missing session folder.
  - On import, the original's `OUTPUT_DIR` lookup also created `<repo>/dataset_outputs` (`create=True`). The mini does not.
- **R script path:** `omic_analysis_components.py:32-36` `_analysis_r_script()` now returns this folder's `run_casestudy.R`. The reference run used `tools/omic_tools/run_casestudy.R` of the same worktree.
- **Imports:** package-relative and `tools.omic_tools.` imports are now flat imports of this folder's modules:
  - `omic_analysis_components.py:14-25`;
  - `r_plotting.py:7-10`;
  - `donor_sampling.py:8`;
  - `query_value_lists.py:8`.
- **HGNC table:** `celltosg_runtime_adapter.py:13-15` pointed at `reference/hgnc_protein_coding.tsv` next to the module; it now reads `settings.HGNC_REFERENCE`.
  - That path is the agent's copy, `tools/omic_tools/reference/hgnc_protein_coding.tsv`: 1.4 MB, git blob `fd6bd8d`, the same in bb92fab and in the current checkout.
  - It is read in place, not copied. To make this folder self-contained, copy the file to `reference/` here and point `HGNC_REFERENCE` at it.
- **Data root:** `configs/paths.yaml` is gitignored, so bb92fab does not fix the data root. The reference run set it to the v3 root, and so does `settings.OMNICELL_DATA_ROOT`.
  - The main checkout's `configs/paths.yaml` still names the older `OmniCellTOSG_dataset`.
- **Option values that reach output files are kept as constants**, so the files stay identical:
  - `donor_sampling.json`: `keep_unknown_donor`, `control_cell_type`, `disease_cell_label`, `shared_datasets_only`, `borrow_controls`, `across_datasets`;
  - the DE input: `"adjust_chrY": True`;
  - `analysis_state.rds`: `diagnostics_text`.

## Removed

A line trace of the reference run (`sys.settrace` over `tools/omic_tools/`) guided the removal. Run `check_lines.py` in the check folder to list every executed line of the original that the mini removes or changes. There are 175, and each falls into one of these groups:
- **Adaptations** (above).
- **Functions never called:** their `def` lines, and module constants, ran at import:
  - name suggestions and query-builder caching (`get_suggestions`, `_get_query_builder`);
  - `.npy` compression (`compress_*`);
  - `keep_genes_detected_in_each_group` and `CHRY_GENES` in the workflow (used only when borrowing controls);
  - `read_enrichment_results`;
  - the donor-sampling helpers for pooling across datasets, and `ADULT_STAGES`;
  - the workflow's command-line `__main__` block;
  - imports left unused (`zipfile`, `Path`, `get_close_matches`, `typing`).
- **Options this call never sets.** Their parameters are gone. Conditions that ran with the default outcome are gone (for example `if tissue:` and `if keep_unknown_donor:`), and so are statements that only fed removed code (for example `smallest` in `donor_sampling.py`, used only by `borrow_controls`). The options are:
  - `tissue`, `gender`, `suspension_type`, `control_cell_type` and `disease_cell_label` filters;
  - `keep_unknown_donor`, `shared_datasets_only=False` and `borrow_controls`, with its `sampling_record` and `control_note`;
  - `enable_differential_expression=False`, `enable_plotting=False`, `enrichment_databases` and `fast_mode`.
- **The `label` option:** other labels (`gender`, `cell_type`), label normalisation, and the label fallback.
- **Missing-input paths:**
  - empty query or empty input (`if not conditions`, `if not fetch_dict`);
  - no session folder;
  - no `gene_names` (the `bmg_to_gene_choice.csv` sidecar read);
  - no requested value in the query lists.
- **Prints of removed options:** `Tissue: None` and `Gender: None`. The agent's job log has these two lines; no output file does.
- **R (`run_casestudy.R`):**
  - `--stage all`, `PSEUDOBULK` and `run_optional_pseudobulk`;
  - `--diagnostics-file`;
  - the `log2_already` input scale;
  - input-CSV discovery and the default output folder;
  - the `COMPOSITE`, `WHICH_PC` and `PERMUTATIONS` environment overrides, now fixed at the values the agent uses (composite on, PC1, 999).
- **R (`kegg_simple.R`):** default paths when arguments are missing, and the legacy return for folders without `enrichment_status.json`. The Python writes both status files before it calls the script.

Kept although this run cannot reach them, because they are input checks or edge handling inside executed code:
- the soft-matching `similar_terms` loop (the loader gets no conditions after sampling);
- in R, the relative-path branch of `pick_csv` and the `chry = FALSE` design path.

## Where the mini differs from the agent

These are deliberate, in the approved design. They do not change today's outputs, but they would if the data or the run failed differently:
- **Cell-type gate.** If the data changed so that microglia fell below 10 disease and 10 normal metacells in shared studies, the outcomes would differ:
  - the agent would return `success: False`, `reason: "cell_type_below_minimum"` and `cell_type_options`;
  - the mini would run the sampler, which refuses, and return the generic "No samples matched" result.
- **Name suggestions.** When retrieval fails, the agent adds a `"suggestions"` key with similar database terms and prints "Generating suggestions". The mini returns the same dict without that key.
- **Label fallback.** If the disease label could not be encoded into two groups (for example a row-count mismatch between matrix and metadata), the outcomes would differ:
  - the agent would fall back to `gender` and run a female-vs-male DE;
  - the mini skips DE and returns `success: True` with no contrast.

## Behaviour carried over from the agent (not fixed here)

- **`control_source` is `None`** in the result. The agent fills it only when controls are borrowed from other studies. The tool's docstring promises `"shared_datasets"` for this case, but `donor_sampling.json` records it.
- **The PERMANOVA is unseeded** (`adonis2`, 999 permutations). Its p-value is at the floor (`p <= 0.001`) and has matched across runs, but a rerun could differ in principle.
- **Enrichr is a live web service.** Enrichment tables reflect Enrichr's libraries on the day of the run. Per-run IDs and timings differ every time: `userListId`, `shortId`, `run_id`, `started_at` and `elapsed_seconds`.

## Verification

On 2026-10-30, the mini run (Slurm 4247353, started from `/`) and the agent tool at bb92fab (Slurm 4246694) were compared:
- They wrote the same 234 files.
- 214 files are byte-identical.
- The other 20 differ only in:
  - PDF timestamps;
  - Enrichr run IDs, upload IDs, start times and request durations;
  - random HTML widget IDs;
  - absolute output paths.
- The returned dicts are equal apart from `timing` and paths.
- No module from the agent or `tools/omic_tools/` was loaded.

A failure-path run was also compared (Slurm 4247802), with the sampler forced to raise on both sides. Both return the same "No samples matched" dict and the same two query-list files. The agent's dict also has the `suggestions` key, which the mini removes by design.

Details: `webapp/sessions/AD_20261030_refactor_check/COMPARISON.md`.

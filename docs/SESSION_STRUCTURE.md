# OmniCellAgent — Session Folder Structure

This document describes everything that lands in
`webapp/sessions/<session-id>/` after running the LangGraph agent
(`python -m agent.langgraph_agent --query "..." --session-id <name>`).

It uses the **`AD-test-2`** session (Alzheimer's Disease case study) as the
worked example. Every other case-study session (`PDAC-test-2`,
`LungCancer-test-2`, …) has an identical layout, only the comparison name
and disease cohort change.

## 1. Pipeline at a glance

The agent is a small LangGraph state machine (`agent/langgraph_agent.py`):

```
planner → executor (5 sub-agents, in plan order) → replanner → reporter
                                                                 │
                                                                 └─→ revise_report_from_feedback
                                                                     (only when called via
                                                                      benchmark/improve_and_review.py)
```

Sub-agents and the files they touch:

| #  | Agent                      | Tool / script                                                                                                                                       | Writes                                                                                                                                                                 |
| -- | -------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1  | **OmicMiningAgent**  | `tools/omic_tools/omic_fetch_analysis_workflow.py` → `omic_analysis_components.py` → `subprocess_r.py` (KEGG R script)                      | `expression_matrix.npy.lrz`, `labels*`, `top_genes_by_expression.csv`, `differential_expression/*`, `volcano_plots/*`, `enrichment_results/*`, `plots/*` |
| 2  | **BioMarkerKGAgent** | `tools/gretriever_tools/gretriever_client.py` (calls port 8001)                                                                                   | No on-disk output — answer is folded into the agent state and ultimately the report                                                                                   |
| 3  | **PubMedResearcher** | `tools/pubmed_tools/*` (PMC downloads + abstract fetches)                                                                                         | No on-disk output in the session dir (papers cached centrally)                                                                                                         |
| 4  | **GoogleSearcher**   | `tools/google_search_tools/google_search_w3m.py`                                                                                                  | No on-disk output                                                                                                                                                      |
| 5  | **ScientistsAgent**  | `tools/scientist_rag_tools/scientist_tool.py` (port 8000) — Genomics / Neuroscience / Longevity-Biostats expert KBs                              | No on-disk output                                                                                                                                                      |
| — | **Reporter**         | `agent/langgraph_agent.py::_reporting_node` + pandoc + xelatex (or wkhtmltopdf fallback)                                                          | `report_<timestamp>.md` + `report_<timestamp>.pdf`                                                                                                                 |
| — | **Revision pass**    | `benchmark/improve_and_review.py` aggregates `benchmark/ai_review_results*/`, then calls `LangGraphOmniCellAgent.revise_report_from_feedback` | `report-revised_<timestamp>.md` + `report-revised_<timestamp>.pdf`                                                                                                 |

## 2. AD-test-2 folder tree

```text
AD-test-2/
├── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──  STEP 2  (data fetch — CellTOSGDataLoader)
├── expression_matrix.npy.lrz                  scRNA-seq counts/cells × genes matrix, ZPAQ-compressed
├── labels.csv                                 per-cell metadata (the cohort that was actually loaded)
├── labels_full_disease.csv                    same + `label_index` column (0 = normal, 1 = AD)
├── label_mapping_disease.csv                  text-→-integer label key (one row per class)
├── labels.npy.lrz                             raw label vector matching the rows of expression_matrix
│
├── top_genes_by_expression.csv                STEP 3 — top-K (K=20) genes ranked by raw mean expression
│
├── differential_expression/                   STEP 4a — unpaired DE between the two label classes
│   ├── unpaired_differential_expression_results.csv   all genes
│   ├── significant_genes_by_fdr.csv                   |FDR| < threshold, sorted by FDR
│   ├── significant_genes_by_fc.csv                    |log2FC| > threshold, sorted by |log2FC|
│   ├── significant_upregulated_genes.csv              log2FC > 0 ∧ significant
│   └── significant_downregulated_genes.csv            log2FC < 0 ∧ significant
│
├── volcano_plots/                             STEP 4b — DE visualisation (Plotly)
│   ├── volcano_plot.png                       strict FDR + log2FC thresholds
│   ├── volcano_plot.html
│   ├── volcano_plot_permissive.png            looser thresholds for showing more candidates
│   └── volcano_plot_permissive.html
│
├── enrichment_results/                        STEP 4c — Enrichr API queries on the DE hit lists
│   ├── Alzheimer's_disease_all_regulated/     gene set = up ∪ down (any direction)
│   │   ├── KEGG_2021_Human_results.csv        pathway DB results
│   │   ├── Reactome_2022_results.csv
│   │   ├── WikiPathways_2019_Human_results.csv
│   │   ├── MSigDB_Hallmark_2020_results.csv
│   │   ├── GO_Biological_Process_2021_results.csv
│   │   ├── GO_Cellular_Component_2021_results.csv
│   │   ├── GO_Molecular_Function_2021_results.csv
│   │   ├── DisGeNET_results.csv
│   │   ├── Jensen_DISEASES_results.csv
│   │   ├── OMIM_Disease_results.csv
│   │   ├── OMIM_Expanded_results.csv
│   │   ├── Human_Phenotype_Ontology_results.csv
│   │   ├── GTEx_Tissue_Expression_Up_results.csv
│   │   ├── GTEx_Tissue_Expression_Down_results.csv
│   │   ├── all_enrichment_results.json        raw Enrichr payload across all DBs
│   │   └── summary.txt                        top hits per DB, human-readable
│   ├── Alzheimer's_disease_up_regulated/      gene set = up-regulated only (same DB list as above)
│   ├── Alzheimer's_disease_down_regulated/    gene set = down-regulated only
│   └── enrichment_plots/                      matplotlib bar plots from the Enrichr CSVs
│       ├── KEGG_2021_Human_all_regulated.{png,pdf}
│       ├── KEGG_2021_Human_up_regulated.{png,pdf}
│       ├── KEGG_2021_Human_down_regulated.{png,pdf}
│       ├── Reactome_2022_all_regulated.{png,pdf}
│       ├── Reactome_2022_up_regulated.{png,pdf}
│       └── Reactome_2022_down_regulated.{png,pdf}
│
├── plots/                                     STEP 5 — clusterProfiler (R) on the *_all_regulated set
│   ├── kegg_dotplot.png                       dot-plot of enriched KEGG pathways (size = gene count, colour = padj)
│   ├── kegg_dotplot.html                      interactive Plotly version of the dotplot
│   ├── kegg_dotplot_files/                      ↳ JS/CSS assets the HTML loads
│   ├── pathway_combined_plot.png              multi-panel: dotplot + lollipop + bar
│   ├── pathway_combined_plot.html             interactive version
│   └── pathway_combined_plot_files/
│
├── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──  STEP 6  (Reporter)
├── report_20260511_203218.md                  agent-authored markdown report (first run)
├── report_20260511_203218.pdf                 pandoc → xelatex (fallback: wkhtmltopdf) rendering of the .md
│
├── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──  POST-RUN (only after improve_and_review.py)
├── report-revised_20260512_140146.md          revised markdown after aggregating AI peer reviews
└── report-revised_20260512_140146.pdf         PDF render of the revised report
```

## 3. File-by-file reference

### 3.1 Raw fetched data — STEP 2

These are written by [`CellTOSGDataLoader`](../tools/omic_tools/omic_fetch_analysis_workflow.py)
(`from CellTOSG_Loader_new import CellTOSGDataLoader`), which queries
OmniCellTOSG and saves the per-cell subset that will be analysed.

| File                          | What it is                                                                                                                               | Biomedical meaning                                                                                                                                                                                                                                                                                                                                                                                                              |
| ----------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `expression_matrix.npy.lrz` | `(N_cells, N_genes)` float32 NumPy array, compressed with **lrzip ZPAQ** (≈15× ratio); often hundreds of MB before compression | Raw single-cell RNA-seq expression counts,**scRNA-seq cohort** — rows are individual cells, NOT bulk samples. For AD-test-2: 1998 cells × 533,458 gene/feature columns covering coding genes + non-coding RNAs                                                                                                                                                                                                          |
| `labels.npy.lrz`            | `(N_cells,)` int array, compressed                                                                                                     | Encoded class label per cell (0 = reference / "normal", 1 = disease, etc.)                                                                                                                                                                                                                                                                                                                                                      |
| `label_mapping_disease.csv` | 2 cols:`label_name`, `label_index`                                                                                                   | Decodes the integer Y back to disease names. For AD:`normal → 0`, `Alzheimer's Disease → 1`. Priority labels (`normal`, `unclassified`, `unknown`) sort first by `CellTOSGDataLoader.PRIORITY_LABELS_BY_TASK` so that the "control" class is always 0                                                                                                                                                             |
| `labels.csv`                | One row per cell, full metadata                                                                                                          | Per-cell ontology:`source` (Brain Cell Atlas / CellxGene / GEO / …), `dataset_id`, `tissue_general`, `tissue`, `cell_type` (raw + harmonised CMT_id/CMT_name), `disease` (raw + BMG-normalised), `sex_normalized`, `development_stage_category`, `birth_phase`, `matrix_file_path` + `matrix_row_idx` (provenance back to the underlying npy shard), `sample_index` (row in `expression_matrix.npy`) |
| `labels_full_disease.csv`   | `labels.csv` + an extra `label_index` column                                                                                         | Same metadata table with the encoded label appended — convenience join                                                                                                                                                                                                                                                                                                                                                         |

The data file naming follows
`CellTOSGDataLoader.label_column` (here `"disease"`); a re-run with
`label="gender"` would produce `label_mapping_gender.csv` and
`labels_full_gender.csv` instead.

### 3.2 Top-by-expression genes — STEP 3

Written by `compute_top_genes` in `omic_fetch_analysis_workflow.py`.

| File                            | Columns                               | Meaning                                                                                                                                                                                                                     |
| ------------------------------- | ------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `top_genes_by_expression.csv` | `rank, gene_index, gene_name, mean_expression_cp10k` | Top-20 genes ranked by**mean expression across all cells in the cohort** (NOT differential). Used as a sanity check / "housekeeping" view; high-mean genes are often ribosomal (`RPL*`, `RPS*`) or mitochondrial. |

### 3.3 Differential expression — STEP 4a

Written by `perform_unpaired_differential_expression` in
`omic_analysis_components.py`. Compares the two class arms identified by
`label_column`:

- **AD case**: group 0 = normal cells, group 1 = AD cells (stratified
  balancing pulls matched normals from the same tissue context).
- **Gender label run**: group 0 = female, group 1 = male.

The test used is a vectorised Mann-Whitney U (`fast_mannwhitney_vectorized`
/ `parallel_mannwhitney_optimized`); p-values are corrected via
Benjamini–Hochberg into FDR. log2FC is computed on the raw counts.

All five CSVs below open with `#`-prefixed comment lines carrying the cohort
validity diagnostics (verdict, contrast, FAIL/CAUTION checks) ahead of the
real header row. Read them with `pd.read_csv(path, comment="#")` — a plain
`pd.read_csv(path)` raises `ParserError: Expected 1 fields in line N, saw M`.

| File                                             | Columns                                                                            | Meaning                                                                                                                                               |
| ------------------------------------------------ | ---------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| `unpaired_differential_expression_results.csv` | `Name, log2_fold_change, effect_size, p_value, FDR, is_significant, abs_log2_fc` | **All genes** with their DE stats. `Name` is the gene symbol. `effect_size` is a Mann-Whitney effect size estimator (1 − 2 · U/(n1·n2)). |
| `significant_genes_by_fdr.csv`                 | same columns                                                                       | Subset sorted by ascending FDR — the canonical "top DEGs" list                                                                                       |
| `significant_genes_by_fc.csv`                  | same columns                                                                       | Subset sorted by `abs_log2_fc` — captures large-magnitude changes even if FDR isn't the lowest                                                     |
| `significant_upregulated_genes.csv`            | same columns                                                                       | `log2_fold_change > 0 ∧ is_significant` — over-expressed in the disease arm                                                                       |
| `significant_downregulated_genes.csv`          | same columns                                                                       | `log2_fold_change < 0 ∧ is_significant` — depleted in the disease arm                                                                             |

For AD-test-2 the top hits (`LINC02241`, `FRG1-DT`, `LINC00486`, `GLIDR`,
`ALG11`) are predominantly long non-coding RNAs and a glycosylation gene
(`ALG11`) — consistent with reported lncRNA dysregulation in AD brain and
N-glycosylation defects in tau pathology.

### 3.4 Volcano plots — STEP 4b

Written by `create_volcano_plot` in `omic_analysis_components.py` (Plotly).

| File                                        | What it shows                                                 |
| ------------------------------------------- | ------------------------------------------------------------- |
| `volcano_plot.png` / `.html`            | x = log2FC, y = −log10(FDR), strict thresholds (FDR < 0.025, |
| `volcano_plot_permissive.png` / `.html` | Same axes, looser thresholds so more candidates show up       |

Biological meaning: a classic DE visualisation. Genes in the upper-right
are strongly **up-regulated AND statistically significant** in disease;
upper-left = down-regulated. The `*_permissive` variant is useful when
the cohort is small and few genes clear strict thresholds.

### 3.5 Enrichr enrichment results — STEP 4c

Written by `perform_enrichment_analysis` → `enrichr_analysis` in
`omic_analysis_components.py`. For each of three gene sets
(`all_regulated`, `up_regulated`, `down_regulated`) the script POSTs to
the Enrichr REST API and queries 14 reference libraries.

Each per-DB CSV is the top-50 enriched terms with columns:
`Term, Overlap, P-value, Adjusted P-value, Old P-value, Old Adjusted P-value, Odds Ratio, Combined Score, Genes`. The `Genes` column is the actual
hit list, useful for cross-checking which DEGs are driving the term.

| Sub-folder                              | Source gene list | Biological angle                                                            |
| --------------------------------------- | ---------------- | --------------------------------------------------------------------------- |
| `Alzheimer's_disease_all_regulated/`  | up ∪ down DEGs  | "Which biology is*disturbed at all* in AD?" — broad pathway footprint    |
| `Alzheimer's_disease_up_regulated/`   | up DEGs only     | Processes activated / over-expressed in AD cells                            |
| `Alzheimer's_disease_down_regulated/` | down DEGs only   | Processes depleted / lost in AD cells (often metabolism, synaptic function) |

The 14 reference libraries fall into themes:

| Library                                     | Domain                 | Interpretation                                                                                     |
| ------------------------------------------- | ---------------------- | -------------------------------------------------------------------------------------------------- |
| `KEGG_2021_Human_results.csv`             | Curated pathways       | Canonical signalling/metabolic pathways (e.g. "Alzheimer disease", "OXPHOS", "Insulin signalling") |
| `Reactome_2022_results.csv`               | Curated pathways       | Finer-grained than KEGG; useful for sub-pathway resolution                                         |
| `WikiPathways_2019_Human_results.csv`     | Community pathways     | Often disease-specific (e.g. tau, amyloid pathways)                                                |
| `MSigDB_Hallmark_2020_results.csv`        | Broad hallmark sets    | 50 hallmark gene sets — "myc targets", "inflammatory response", "OXPHOS"                          |
| `GO_Biological_Process_2021_results.csv`  | Function ontology      | What the genes*do* (e.g. "synaptic transmission", "translation")                                 |
| `GO_Cellular_Component_2021_results.csv`  | Location ontology      | Where they act (e.g. "synapse", "mitochondrion")                                                   |
| `GO_Molecular_Function_2021_results.csv`  | Activity ontology      | Biochemical role (kinase, transporter, etc.)                                                       |
| `DisGeNET_results.csv`                    | Gene-disease           | Which other diseases share these genes (comorbidity / mechanism overlap)                           |
| `Jensen_DISEASES_results.csv`             | Gene-disease           | Alternative gene-disease evidence aggregator                                                       |
| `OMIM_Disease_results.csv`                | Mendelian              | Strong-effect monogenic disease overlap                                                            |
| `OMIM_Expanded_results.csv`               | Mendelian + neighbours | Extended OMIM hits                                                                                 |
| `Human_Phenotype_Ontology_results.csv`    | Clinical phenotype     | Patient-observed phenotypes (e.g. "dementia", "memory impairment")                                 |
| `GTEx_Tissue_Expression_Up_results.csv`   | Tissue specificity     | Which tissues highly express these genes — sanity check for cell-type / tissue context            |
| `GTEx_Tissue_Expression_Down_results.csv` | Tissue specificity     | Tissues where these genes are low — complementary view                                            |

Two cross-DB artefacts per gene set:

- `all_enrichment_results.json` — full Enrichr payload across all libraries (large)
- `summary.txt` — flattened human-readable summary, top hits per library

### 3.6 Enrichment bar plots — STEP 4c (visual)

Written by the same Enrichr block. matplotlib bar plots of the top-N
terms ranked by `Combined Score` (= log(p) × OR).

`enrichment_plots/<LIBRARY>_<direction>.{png,pdf}` — currently only
`KEGG_2021_Human` and `Reactome_2022` are plotted by default; the rest
remain CSV-only.

### 3.7 R clusterProfiler plots — STEP 5

Written by an R script invoked via `subprocess_r.run_r_script` (the path
is set in `configs/paths.yaml::enrichment.kegg_script`). The R script
reads `enrichment_results/<comparison>_all_regulated/` and produces:

| File                                       | What it shows                                                                                                                                                                     |
| ------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `plots/kegg_dotplot.{png,html}`          | KEGG dot-plot — y = pathway, x = GeneRatio, dot size = gene count, dot colour = adjusted p-value. Same data as the matplotlib bars but in the clusterProfiler conventional view. |
| `plots/pathway_combined_plot.{png,html}` | Multi-panel: dot-plot + lollipop + horizontal bar. Used as the headline pathway figure in the report.                                                                             |
| `plots/*_files/`                         | Plotly JS/CSS assets the `.html` versions load. Safe to delete if you only need PNGs.                                                                                           |

Biologically the dot-plot is the most-cited single panel — it captures
**which canonical pathways are over-represented** in the DE hit list and
how confident we are (colour scale).

### 3.8 Final report — STEP 6 (Reporter)

Written by `_reporting_node` in `agent/langgraph_agent.py` after the
five sub-agents finish. Filenames follow `report_<UTC timestamp>.md` /
`.pdf`.

The markdown follows the structure prescribed in the reporter prompt:

1. **scRNA-seq Cohort Analysis Summary** — sample sizes per group,
   DEG counts, top-DEG table, volcano embed.
2. **Knowledge Graph Analysis** — first-neighbour nodes of top DEGs
   from PrimeKG (drugs / pathways / GO terms / co-morbid diseases).
3. **Literature-Validated Targets** — intersection of DEGs and
   PubMed-supported targets, with citations.
4. **Pathway Enrichment Analysis** — KEGG/Reactome/GO summary with
   embedded pathway plots.
5. **Gene-Anchored Mechanistic Hypotheses** — 3–5 ranked hypotheses,
   each scored 0–100 across {fit-to-evidence, mechanistic plausibility,
   testability, novelty}.
6. **Minimal Validation Experiments** — concrete bench experiments
   (qPCR / IHC / knockdown / functional assays).

The PDF is generated by:

1. Try `pandoc → xelatex` with `LATEX_HEADER` for proper fonts, table
   styling, and code wrapping.
2. Fall back to `pandoc → HTML → wkhtmltopdf` with embedded CSS if
   xelatex is missing.
3. Fall back to HTML-only if neither PDF engine is available.

### 3.9 Revised report — POST-RUN

Written by `benchmark/improve_and_review.py` → `revise_report_from_feedback`.

| File                         | What it is                                                                                                                                                                                            |
| ---------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `report-revised_<UTC>.md`  | The same case-study analysis re-authored to address aggregated reviewer feedback from `benchmark/ai_review_results*/{apr,litllm,openrev}/`. Roughly the same length as the original (≈ 37–42 kB). |
| `report-revised_<UTC>.pdf` | Same pandoc/xelatex pipeline as the first-run report.                                                                                                                                                 |

Suffix is `-revised` (with a hyphen, distinct from `_revised_` that the
agent uses internally) so `find ... 'report-revised_*'` reliably picks
out post-review outputs.

## 4. Cross-session companion artefacts

These do NOT live under a session dir; they aggregate across sessions:

| Path                                                                                     | Producer                                             | Meaning                                                                                                           |
| ---------------------------------------------------------------------------------------- | ---------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `benchmark/ai_review_results_v2/<case>/apr/{review_*.md, meta_review.md, result.json}` | `benchmark/run_ai_review.py --reviewer apr`        | Multi-model AI peer reviews (Gemini 2.5-Pro, GPT-4-o1, GPT-4-o3-mini) of the first-run report, plus a meta-review |
| `benchmark/ai_review_results_v2/<case>/litllm/related_works.md`                        | `--reviewer litllm`                                | LitLLM-style citation analysis                                                                                    |
| `benchmark/ai_review_results_v2/<case>/openrev/review.md`                              | `--reviewer openrev`                               | OpenReviewer LLaMA-3 8B review (uses HuggingFace transformers if no OpenRouter key)                               |
| `benchmark/ai_review_results_v2/<case>/<reviewer>_revised/`                            | `benchmark/improve_and_review.py` (re-review pass) | Same reviewers run a second time on the revised report                                                            |
| `benchmark/ai_review_results_v2/score_comparison.{csv,png}`                            | `improve_and_review.py` final viz                  | 1–10 score per (case × reviewer), original vs revised                                                           |
| `logs/appendix/supplementary_reports[_<suffix>].pdf`                                   | `scripts/combine_supplementary_pdfs.py`            | Cover + TOC + (3 cases × first-run + revised) bundled into one PDF (≈100 pages)                                 |

## 5. Re-running the same case

Two facts about naming that prevent accidental overwrites:

1. **Session dir name comes from `--session-id`** — passing
   `--session-id AD-test-2` keeps the original `AD-test/` intact.
2. **Reports and revisions get a UTC timestamp** — running the agent
   twice within the same session dir produces
   `report_<t1>.{md,pdf}` and `report_<t2>.{md,pdf}` side-by-side; the
   reviewer scripts then auto-pick the latest report by mtime via
   `--session-suffix` + glob (see `benchmark/run_ai_review.py::_resolve_report`).

The large `expression_matrix.npy.lrz` (≈340 MB for 1k cells × 533k
features) IS overwritten on re-fetch, so if you need to keep the prior
matrix, copy it before re-running.

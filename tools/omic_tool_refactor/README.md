# Mini omic pipeline: AD microglia case study

This folder runs one case study from OmniCellAgent's single-cell omic analysis without the LLM agent: Alzheimer's disease vs normal microglia from brain, compared as metacells from OmniCellTOSG v3.

The run has five stages:
1. selects disease and normal metacells;
2. normalises the expression;
3. runs limma differential expression (DE);
4. runs Enrichr pathway enrichment;
5. draws the case-study figures.

It writes the same files and returns the same result as the agent's `omic_analysis_tool` for this case study (verified 2026-10-30; see `EXTRACTION.md`).

## Start

```bash
OCA=/storage3/fs1/fuhai.li/Active/di.huang/Research/LLM/OmniCellAgent
PY=/storage3/fs1/fuhai.li/Active/di.huang/cache/miniconda/omnicellagent/bin/python

$OCA/scripts/r_env.sh $PY $OCA/tools/omic_tool_refactor/run_ad_microglia.py <output_root>
```

- **`scripts/r_env.sh`** loads R (module `r/4.5.2`, library in `R_LIBS`), checks the R packages, then runs the command.
- **Working directory:** the command works from any directory.
- **`<output_root>`:** use a new, empty folder. It is created if missing. In a folder used before, Enrichr moves the previous results into `history/` subfolders.
- **Exit code:** 0 if the run succeeds, 1 if it fails. Either way it prints `success=... num_samples=...`.
- **Resources:** about 7 minutes on 8 CPUs. The verified jobs requested 64 GB; Slurm's sampled peak was about 4 GB. On the cluster:

  ```bash
  srun -p general-cpu -A compute2-fuhai.li -c 8 --mem 64G \
      $OCA/scripts/r_env.sh $PY $OCA/tools/omic_tool_refactor/run_ad_microglia.py <output_root>
  ```

From Python, with this folder on `sys.path` and R on `PATH` (for example, Python started through `scripts/r_env.sh`):

```python
import run_ad_microglia
result = run_ad_microglia.main("/absolute/output_root")  # the result dict; omic_result.json is written only by the command line
```

## Input

### What you give it

| Input | Where | Value |
|---|---|---|
| output folder | command-line argument | any path |
| disease | `run_ad_microglia.py` `DISEASE` | `"Alzheimer's disease"` |
| cell type | `run_ad_microglia.py` `CELL_TYPE` | `"microglial cell"` |
| organ | `run_ad_microglia.py` `ORGAN` | `"brain"` |
| comparison | `omic_fetch_analysis_workflow.py` | disease vs normal (label `"disease"`) |

The three names are matched against the OmniCellTOSG vocabulary:
- **Disease names:** both v3 spellings, "Alzheimer's Disease" and "Alzheimer disease", count as one disease. The match ignores letter case, surrounding spaces and a possessive "'s".
- **Cell-type and organ names:** the match is exact apart from letter case.

The pipeline is verified only for this case study. Some cohorts need options the agent sets but this folder no longer has (see `EXTRACTION.md`): a separate control cell type, metacells without donor IDs, or controls borrowed from other studies.

### Data and software it reads

Paths are set in `settings.py`:

| Setting | Path | What is read |
|---|---|---|
| `OMNICELL_DATA_ROOT` | `/storage3/fs1/fuhai.li/Active/Shared/dataset/OmniCellTOSG_dataset_v3` | `cell_metadata_with_mappings.parquet` (one row per metacell), `bmg_gene_index.csv` (gene → matrix column), `expression_matrix/` (counts). Read only: a temporary folder links these files. |
| `OMNICELL_ROOT` | `/storage3/fs1/fuhai.li/Active/di.huang/Research/LLM/OmniCellTOSG` | the `CellTOSG_Loader` package (checkout at `5abaf2e` for the verified run) |
| `HGNC_REFERENCE` | `tools/omic_tools/reference/hgnc_protein_coding.tsv` | HGNC approved protein-coding symbols, which define the gene axis |
| `KEGG_SCRIPT` | `kegg_simple.R` in this folder | the enrichment plot script |

It also needs:
- **Python:** the `omnicellagent` conda environment (pandas, numpy, pyarrow, requests).
- **R:**
  - `run_casestudy.R` uses limma, statmod, data.table, ggplot2, ggrepel, patchwork, RColorBrewer, corrgram, vegan, multcompView, ggpubr, ragg, withr, ggplotify and jsonlite. AnnotationDbi and org.Hs.eg.db are optional; they fill the gene full names.
  - `kegg_simple.R` uses dplyr, stringr, forcats, plotly and htmlwidgets.
- **Network:** the Enrichr web service (`https://maayanlab.cloud/Enrichr`).

### Fixed analysis settings

| Setting | Value | Set in |
|---|---|---|
| metacells per group (cap) | 1,000 | `donor_sampling.DEFAULT_SAMPLE_SIZE` |
| minimum metacells per group | 10 | `donor_sampling.MIN_METACELLS_PER_GROUP` |
| sampling seed | 42 | `omic_fetch_analysis_workflow.py` |
| normalisation | counts per 10,000 (CP10K) per metacell; DE uses log2(CP10K + 1) | workflow, `run_casestudy.R` |
| DE significance | Benjamini–Hochberg FDR < 0.05 | `omic_analysis_components.py` → `run_casestudy.R` |
| genes per significant-gene table | at most 1,000 | same |
| figure thresholds | FDR < 0.05 and \|log2FC\| ≥ 1 (volcano colours); top 100 genes (PCA); top 20 (correlation plot); 999 permutations (PERMANOVA) | `run_casestudy.R` |
| Enrichr | 14 libraries; 50 best terms saved per library; 60 s per request | `omic_analysis_components.py`, `enrichr_client.py` |
| R time limit | 3,600 s per call | workflow |

## Pipeline

```
run_ad_microglia.py ── disease, cell type, organ
  │
  ▼  omic_fetch_analysis_workflow.py
1 Select the cohort      donor_sampling.py          disease + normal metacells, shared studies only
2 Load expression        celltosg_runtime_adapter + CellTOSG_Loader   raw counts, HGNC protein-coding genes
3 Normalise, QC          CP10K; 20 most abundant genes
4 Differential expr.     cohort_diagnostics.py; run_casestudy.R --stage de (limma)
5 Enrichment             enrichr_client.py → Enrichr: all / up / down gene lists
6 Case-study figures     run_casestudy.R --stage plots
7 Pathway plots          kegg_simple.R × 3 (all / up / down)
  │
  ▼
result dict ── omic_result.json
```

The numbers below are from the verified run on 2026-10-30.

**1. Select the cohort** (`omic_fetch_with_new_loader`, `donor_sampling.sample_known_donor_cohort`)
- **Vocabularies:** reads the metadata of all metacells and writes every disease and cell-type name, with the matched name on line one, to `available_diseases.txt` (146 names) and `available_cell_types.txt` (757).
- **Candidates:** takes the microglia metacells from brain with an AD or normal label, and drops metacells without a usable donor ID.
- **Shared studies:** keeps only studies (`dataset_id`) that contribute both AD and normal metacells; there are 10. A study that holds one group only would carry its study effect into that group alone.
- **Minimum and cap:** stops if either group has fewer than 10 metacells. Otherwise each group is capped at 1,000 metacells: one per donor first, then further rounds, with seed 42.
- **This run:**
  - AD: 1,000 of 1,189 eligible metacells, from 110 donor IDs;
  - normal: all 649 metacells, from 85 donor IDs.
- **Record:** `donor_sampling.json`.

**2. Load expression** (`celltosg_runtime_adapter.writable_celltosg_root`, `CellTOSGDataLoader`)
- **Temporary data folder:** holds the selected metadata and a gene index restricted to HGNC approved protein-coding symbols.
- **Loader:** OmniCellTOSG's loader reads the raw count rows of the 1,649 metacells for 19,109 genes. It writes `expression_gene.csv` (counts), `labels.csv` (metadata per metacell) and `bmg_to_gene_choice.csv` (which matrix column each gene uses).
- **Labels:** normal = 0, Alzheimer's disease = 1.

**3. Normalise and QC**
- **Normalisation:** scales each metacell to 10,000 counts (CP10K), without a log.
- **Abundance check:** writes the 20 genes with the highest mean CP10K to `top_genes_by_expression.csv`. This checks that the intended cell type was retrieved; it is not a disease signal.

**4. Differential expression** (`select_contrast`, `cohort_diagnostics.py`, `omic_analysis_components.py`, `run_casestudy.R --stage de`)
- **Contrast:** normal (649) vs Alzheimer's disease (1,000).
- **Cohort diagnostics:**
  - six checks: protocol balance, dataset overlap, donor usability, donor count, donor dominance and sequencing-depth ratio;
  - written to `cohort_diagnostics.json` and `differential_expression/COHORT_DIAGNOSTICS.txt`; this run's verdict is OK;
  - advisory only: they never stop the run.
- **Inputs to R:** the two matrices (genes × metacells) go to `foranalysis_combined_{normal,disease}_df_*.csv`, and each metacell's study to `de_covariates_*.csv`.
- **In R:**
  - Expression is transformed to log2(CP10K + 1).
  - Genes without variance are removed, leaving 18,241 tested genes.
  - The limma model is `~ 0 + group + chrY_score + dataset`. `chrY_score` is the mean log2 expression of seven Y-chromosome genes and adjusts for each metacell's male RNA share. `dataset` adjusts for study and is dropped if it is confounded with group.
  - The model is fitted with robust, trended `eBayes`.
- **Result:** 5,175 genes at FDR < 0.05.
- **Output:**
  - `differential_expression/`: the full table (`unpaired_differential_expression_results.csv`, with log2 fold change, Cohen's d as `effect_size`, p-value and FDR) and four significant-gene tables (by FDR, by fold change, up, down), each with at most 1,000 genes;
  - `casestudy_R/DE_results_table.csv`: limma's own table;
  - `casestudy_R/analysis_state.rds`: the saved R state used by step 6.

**5. Enrichment** (`perform_enrichment_analysis`, `enrichr_client.fetch_enrichment`)
- **Gene lists:** three, all at FDR < 0.05 with no fold-change cutoff:
  - all: the 1,000 genes with the lowest FDR;
  - up: up to 1,000 up-regulated genes, by p-value;
  - down: up to 1,000 down-regulated genes, by p-value.
- **Libraries:** each list goes to Enrichr against 14 libraries: GO Biological Process, Molecular Function and Cellular Component (2021), KEGG 2021 Human, Reactome 2022, WikiPathways 2019 Human, MSigDB Hallmark 2020, DisGeNET, OMIM Disease, OMIM Expanded, Human Phenotype Ontology, Jensen DISEASES, and GTEx tissue expression up and down.
- **Output:** `enrichment_results/Alzheimer's_disease_{all,up,down}_regulated/`, each with:
  - one `<library>_results.csv` per library (the 50 terms with the lowest adjusted p);
  - the raw Enrichr responses (`all_enrichment_results.json`, `raw/`);
  - `gene_list.txt`, `summary.txt` and `enrichment_status.json`.

  `enrichment_results/enrichment_status.json` records the three groups.

**6. Case-study figures** (`r_plotting.render_analysis_plots`, `run_casestudy.R --stage plots`)

R reloads `analysis_state.rds`, checks that it matches the input matrices (MD5), and draws into `casestudy_R/`:
- **PCA and PERMANOVA:** PCA of the 100 genes with the lowest FDR, and a PERMANOVA on Bray–Curtis distances with 999 permutations. This run: R² = 0.087, p ≤ 0.001.
- **Volcano plot:** `DE_results_volcano.{png,pdf}`. It colours genes with FDR < 0.05 and |log2FC| ≥ 1 and labels up to 15 of them.
- **Correlation plot:** the 20 genes with the lowest FDR (`DE_results_corrgram_genes`).
- **PC1 violin plot:** by group, with a Wilcoxon test (`DE_results_violin_PC1_2groups.pdf`).
- **PCA panel:** with Tukey letters (`DE_results_PCA_panel`).
- **Composite figure A–D:** `DE_results_figure_ABCD`.
- **Gene lists:** the genes behind panels A and B (`DE_results_panel_{A,B}_*.csv`), plus `plot_manifest.json`.

**7. Pathway plots** (`r_plotting.render_enrichment_plots`, `kegg_simple.R`)

For each gene list it draws, as PNG and interactive HTML:
- a KEGG dot plot of the 20 best KEGG terms;
- a combined bar plot of the 5 best terms each from GO BP, CC, MF, KEGG and DisGeNET.

Output goes to `plots/` (all), `plots/up/`, `plots/down/` and `plots/plot_manifest.json`.

**8. Result** (`run_ad_microglia.main`)

The workflow's result dict, post-processed as the agent tool does, is saved by the command line to `omic_result.json`. Main fields:
- **Run:** `success`, `num_samples` (1,649 metacells), `num_features` (19,109 genes), `contrast` (groups and counts).
- **Findings:** `top_genes_by_fdr` (20 genes with log2FC, FDR, p, effect size) and `enrichment_top_terms` (up and down: at most 5 terms with adjusted p < 0.05 from KEGG, Reactome, GO BP and MSigDB Hallmark).
- **QC:** `qc_highest_mean_expression_genes` (the cell-identity check from step 3).
- **Status:** per step (`de_success`, `enrichment_status`, `de_plots_status`, `enrichment_plot_status`, ...).
- **Files:** `analysis_paths` and `plots_for_report` (figure paths), plus `timing`.

## Output folder

```
<output_root>/
├── omic_result.json                         result dict (step 8)
└── omic_microglial_cell/
    ├── available_diseases.txt, available_cell_types.txt   vocabularies (1)
    ├── donor_sampling.json                  cohort selection (1)
    ├── expression_gene.csv, labels.csv, bmg_to_gene_choice.csv   loader output (2)
    ├── top_genes_by_expression.csv          abundance check (3)
    ├── cohort_diagnostics.json              cohort checks (4)
    ├── foranalysis_combined_*_df_*.csv, de_covariates_*.csv   DE inputs (4)
    ├── differential_expression/             DE tables, COHORT_DIAGNOSTICS.txt (4)
    ├── enrichment_results/                  Enrichr results per gene list (5)
    ├── casestudy_R/                         figures, analysis_state.rds (4, 6)
    ├── plots/                               KEGG and pathway plots (7)
    └── volcano_plots/                       empty; kept from the agent's folder layout
```

The session folder holds 234 files, about 710 MB:
- the two DE input matrices, about 400 MB;
- `expression_gene.csv`, 130 MB.

## Reading the results

- **Samples are metacells,** about 200 cells of one cell type from one source file, not donors.
  - The 1,649 metacells carry 195 donor IDs: 110 AD and 85 normal.
  - Some studies record a sample rather than a person as the donor, so there are fewer persons.
  - limma treats metacells as independent, so its FDR overstates the evidence.
  - In a 2026-10-30 check on this cohort, AD and normal labels were shuffled among persons within each study. Each fold still gave 1,469–6,502 genes at FDR < 0.05 (`webapp/sessions/AD_20261030_v3_study_split/STUDY_SPLIT_SUMMARY.md` in the cluster checkout; `webapp/sessions/` is not in git).
  - The leading genes and the MHC signal did replicate across disjoint study sets.
- **No gene reaches |log2FC| ≥ 1** on the log2(CP10K + 1) scale. So the volcano colours and labels no genes, and the panel A gene CSV has a header only. The 5,175 FDR < 0.05 genes all have smaller fold changes.
- **Enrichr is a live service.** Its libraries can change between runs.

## Files in this folder

| File | Role |
|---|---|
| `run_ad_microglia.py` | entry point: the case-study arguments, the call, result post-processing |
| `settings.py` | data, code and reference paths |
| `omic_fetch_analysis_workflow.py` | steps 1–8 in order |
| `donor_sampling.py` | cohort selection (step 1) |
| `query_value_lists.py` | vocabulary files (step 1) |
| `celltosg_runtime_adapter.py` | temporary data folder and HGNC gene filter (step 2) |
| `cohort_diagnostics.py` | cohort checks (step 4) |
| `omic_analysis_components.py` | DE input export and R call, Enrichr calls, enrichment summary (steps 4–5) |
| `de_results_io.py` | typed reading of the DE tables |
| `enrichr_client.py` | Enrichr requests, response checks, status files (step 5) |
| `r_plotting.py` | R plot calls and plot manifests (steps 6–7) |
| `subprocess_r.py` | runs `Rscript` |
| `run_casestudy.R` | limma DE (`--stage de`) and case-study figures (`--stage plots`) |
| `kegg_simple.R` | pathway plots |
| `EXTRACTION.md` | how this folder was cut from the agent's code, where it differs, and how equality was verified |

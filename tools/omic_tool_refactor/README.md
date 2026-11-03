# Mini omic pipeline: AD microglia case study

This folder runs one case study from OmniCellAgent's single-cell omic analysis without the LLM agent: Alzheimer's disease vs normal microglia from brain, compared as metacells from OmniCellTOSG v3.

The run has five stages:
1. selects disease and normal metacells;
2. normalises the expression;
3. runs limma differential expression (DE);
4. runs Enrichr pathway enrichment;
5. draws the case-study figures.

It writes the same files and returns the same result as the agent's `omic_analysis_tool` for this case study (verified 2026-10-30; see `EXTRACTION.md`), with one exception: the volcano plot was redrawn after the extraction (commit `e58bcc7`). So the volcano, panel A of the composite figure and the panel gene tables now differ from the agent's.

There are two entry points:
- **`run_ad_microglia.py`:** the microglia run described below. It is the agent tool's call.
- **`run_ad_sex_stratified.py`:** runs all three AD case-study cohorts (microglia, astrocyte, glutamatergic neuron) and draws the female-vs-male four-sector figure. See *Sex-stratified figure*.

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

`run_casestudy.R` also reads `ad_label_genes.tsv` from its own folder: the genes the volcano plot may label. It lists 375 Alzheimer's disease genes that at least two of GWAS Catalog 2023, KEGG hsa05010, DisGeNET and Jensen DISEASES name (Enrichr libraries, retrieved 2026-11-02).

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
| figure thresholds | FDR < 0.05, no fold-change cutoff (volcano colours); 10 labels per direction from `ad_label_genes.tsv` (volcano); top 100 genes (PCA); top 20 (correlation plot); 999 permutations (PERMANOVA) | `run_casestudy.R` |
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
- **Volcano plot:** `DE_results_volcano.{png,pdf}`.
  - It colours genes with FDR < 0.05 by the sign of log2FC, with no fold-change cutoff, and draws the FDR = 0.05 line.
  - It labels up to 20 genes from `ad_label_genes.tsv`: the 10 most significant up and the 10 most significant down. Another list can be passed with `--label-genes`.
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

`run_ad_sex_stratified.py` writes the same kind of cohort folder for each cell type: `omic_microglial_cell/`, `omic_astrocyte/` and `omic_glutamatergic_neuron/`. It also writes `omic_result_<cell type>.json` for each cohort it runs, and:

```
<output_root>/sex_stratified/
├── foursector_scatter_AD.png, .pdf          the figure
└── <cell type>/                             merged_de.csv, merged_de_categories.csv,
                                             sex_disease_interaction.csv, summary.json
```

The microglia session folder holds 234 files, about 710 MB:
- the two DE input matrices, about 400 MB;
- `expression_gene.csv`, 130 MB.

## Sex-stratified figure (three AD cohorts)

```bash
$OCA/scripts/r_env.sh $PY $OCA/tools/omic_tool_refactor/run_ad_sex_stratified.py <output_root>
```

This draws `foursector_scatter_AD.{png,pdf}`: for each cohort, the AD-vs-control log2 fold change in females (x) against males (y). It reproduces the figure first drawn on 2026-10-30 in `AD_20261027_v3_omic_only/sex_stratified/`.

1. **Cohorts:** the workflow above runs for "microglial cell", "astrocyte" and "glutamatergic neuron", the agent's AD cell types, into `omic_<cell type>/`.
   - The call is the one `run_ad_microglia.py` makes. On the 2026-11-02 data: 1,649, 2,000 and 1,599 metacells.
   - A cohort folder that already holds its DE inputs is reused, so a `run_ad_microglia.py` output root can be extended without re-running microglia.
   - A reused folder is not checked for which code or data produced it.
2. **Sex models:** `sex_stratified_de.R` on each cohort, all three at once. It reads four files from the cohort folder: the two DE input matrices, `de_covariates_*.csv` (the study) and `labels.csv` (sex and donor).
   - **Join:** `labels.csv` is joined to the matrix columns by position (disease block, then normal block). The join is checked against each metacell's disease status and study.
   - **Data:** the same log2(CP10K + 1) values and gene filter as the main DE. Metacells without a female or male label are dropped: 4, 15 and 0 in the three cohorts.
   - **Model:** one limma model, `~ 0 + group + dataset` with group = status × sex. The donor (`source|dataset_id|donor_id`) is a random block (`duplicateCorrelation`; within-donor correlation 0.16, 0.23 and 0.19). The fit uses robust, trended `eBayes`.
   - **Contrasts:** the disease effect in each sex (the figure's axes), the sex effect within each status, and the sex × disease interaction.
   - **Output:** `sex_stratified/<cell type>/merged_de.csv` (the figure's input), `sex_disease_interaction.csv` (every contrast, ordered by interaction p) and `summary.json` (counts).
3. **Figure:** `plot_sex_stratified_scatter.py` draws one panel per cohort.
   - **Colouring:** a gene is coloured if, in a sex, FDR < 0.05 and |log2FC| ≥ 0.5. Categories: female-specific, male-specific, both up, both down, discordant.
   - **Labels:** up to 10 genes per panel.
   - **Output:** `sex_stratified/foursector_scatter_AD.{png,pdf}`, plus `merged_de_categories.csv` per cohort.

**Runtime:** about an hour from an empty output root (58 minutes on 2026-11-02, 8 CPUs, 64 GB requested, peak about 6 GB):
- the three cohorts run one after another, about 18 minutes;
- the sex models then take as long as the slowest, about 40 minutes, almost all of it `duplicateCorrelation` on about 18,500 genes;
- the figure takes seconds.

**Verified 2026-11-02**, a single run of this command into an empty folder, against the 2026-10-30 outputs in `AD_20261027_v3_omic_only`:
- For each cohort, the four sex-model inputs are byte-identical.
- So are `merged_de.csv`, `sex_disease_interaction.csv`, `merged_de_categories.csv` and the PNG.
- `summary.json` differs only in `cohort_dir`, and the PDF only in its creation date.
- See `COMPARISON.md` in the check folder.

This check ran before the volcano change of `e58bcc7`. That commit changes only the volcano plot (plot stage) of `run_casestudy.R`, not the four files the sex models read.

**Reading it:**
- At the figure's cutoffs the panels are nearly empty: 17, 7 and 4 coloured genes.
  - Each sex has 764–1,953 genes at FDR < 0.05, but their median |log2FC| is about 0.1. Most fall inside the ±0.5 band.
- **Do not cite the interaction counts in `summary.json` and `sex_disease_interaction.csv` yet.**
  - Y-chromosome genes come out as significant sex × disease interactions (FDR < 0.05) in every cohort: RPS4Y1 in microglia; UTY, DDX3Y, USP9Y and NLGN4Y in astrocytes; DDX3Y in glutamatergic neurons. That points to leftover study × sex × status confounding.
  - No donor-level sex-label permutation has been run to show the counts exceed chance.

## Reading the results

- **Samples are metacells,** about 200 cells of one cell type from one source file, not donors.
  - The 1,649 metacells carry 195 donor IDs: 110 AD and 85 normal.
  - Some studies record a sample rather than a person as the donor, so there are fewer persons.
  - limma treats metacells as independent, so its FDR overstates the evidence.
  - In a 2026-10-30 check on this cohort, AD and normal labels were shuffled among persons within each study. Each fold still gave 1,469–6,502 genes at FDR < 0.05 (`webapp/sessions/AD_20261030_v3_study_split/STUDY_SPLIT_SUMMARY.md` in the cluster checkout; `webapp/sessions/` is not in git).
  - The leading genes and the MHC signal did replicate across disjoint study sets.
- **No gene reaches |log2FC| ≥ 1** on the log2(CP10K + 1) scale. The 5,175 FDR < 0.05 genes all have smaller fold changes.
  - With the agent's |log2FC| ≥ 1 cutoff, the volcano coloured and labelled no genes. That is why `e58bcc7` colours by FDR alone.
  - The volcano's colours therefore mark statistical significance, not large effects.
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
| `ad_label_genes.tsv` | AD genes the volcano plot may label |
| `run_ad_sex_stratified.py` | entry point for the three AD cohorts and the sex-stratified figure |
| `sex_stratified_de.R` | sex × disease limma model for one cohort folder |
| `plot_sex_stratified_scatter.py` | the four-sector figure, one panel per cohort |
| `EXTRACTION.md` | how this folder was cut from the agent's code, where it differs, and how equality was verified |

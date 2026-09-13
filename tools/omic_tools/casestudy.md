# Case-study panels — spec

Spec for `casestudy_panels.Rmd` and its runner `run_casestudy.R`.

## 1. Purpose

Takes the two expression matrices a session already exports, runs differential
expression in R, and emits the four figure panels (A–D) used in the
OmniCellAgent case studies.

It is a **downstream consumer** of a session directory, not part of the Python
pipeline. Nothing in the repo reads its output; nothing in the pipeline depends
on it running. It exists because the Python workflow writes the
`foranalysis_combined_*` matrices and then never uses them again — this is what
they are for.

It duplicates the Python DE by design: limma's moderated *t* on log2 CP10K is a
different estimator from the Mann-Whitney used in
`omic_analysis_components.py`, and the panels are built for a manuscript rather
than for the agent's PDF.

## 2. Inputs

### Required — from a session directory

| File | Shape | Source |
|---|---|---|
| `foranalysis_combined_normal_df_<Disease>.csv` | genes × cells | `omic_analysis_components.py:143` |
| `foranalysis_combined_disease_df_<Disease>.csv` | genes × cells | `omic_analysis_components.py:142` |

Column 1 is `Name` (HGNC symbol); every other column is **one cell**, named
`ns_sample_i` / `ds_sample_i`. The `sample` in those names is a misnomer — see
Limitations.

Contents are **linear CP10K**: `normalize_cp10k()` in
`omic_fetch_analysis_workflow.py:727` rescales each cell to sum to 10,000 and
deliberately does not apply `log1p`. Verified: every column sums to exactly
`10000.00`.

### Optional — pseudobulk only

`labels.csv`, joined to the matrices **positionally** (see Limitations).
Supplies `donor_id`.

### Environment

```
module load r/4.5.2                 # R is not on PATH by default
module load cmake/3.30.9            # only needed to (re)install ggpubr
```

Packages: `limma`, `statmod`, `corrgram`, `vegan`, `multcompView`, `ggpubr`,
`ggplot2`, `ggrepel`, `patchwork`, `RColorBrewer`, `data.table`, `pheatmap`,
`ragg`, `ggplotify`, `knitr`. Installed under
`/rdcw/fs1/fuhai.li/Active/di.huang/cache/R` (set by `~/.Rprofile`).

Two environment constraints the document works around rather than fails on:

- **No pandoc.** `rmarkdown::render()` runs every chunk then dies at HTML
  conversion. The runner uses `knitr::knit()`, which honours `eval=` (unlike
  `purl()`) and produces `.md` plus all figures.
- **No graphics devices.** `capabilities()[c("png","jpeg","cairo","X11")]` is all
  `FALSE` on this R build, so base `png()` raises *"X11 is not available"* and
  knitr's default device fails on *every* chunk, not just plotting ones. All
  raster output goes through `ragg`; `pdf()` is native and used alongside.

## 3. Workflow

```
load-data        read both CSVs, keep gene symbols as rownames
    ↓
scale-check      assert median colSum == 1e4, then log2(x + 1)      [hard stop on mismatch]
    ↓
combine          intersect genes, cbind, build `group`, drop zero-variance genes
    ↓
de               lmFit(~ 0 + group) → makeContrasts → contrasts.fit → eBayes  ⇒ res
    ↓
pca-permanova    prcomp on top-N genes, vegdist(bray), adonis2               ⇒ pcoadata
    ↓
panel-a  panel-b  panel-c  panel-d       (plotting only; none creates data another needs)
    ↓
figure-abcd      assembled composite                    [params$composite]
    ↓
pseudobulk       donor-level contrast                   [params$do_pseudobulk, default off]
    ↓
heatmap          reference only                         [eval = FALSE]
```

Calculations are deliberately separated from plotting so no plotting chunk
silently produces an object another plotting chunk consumes.

### Running

```bash
module load r/4.5.2
Rscript run_casestudy.R <session_dir> [out_dir]
```

Input matrices are discovered by pattern, so the disease name (and its
apostrophe) never has to be typed. Output defaults to
`<session_dir>/casestudy_R/`, never the session root.

Options are environment variables: `COMPOSITE=false` (~3× faster),
`PSEUDOBULK=true`, `PERMUTATIONS=999`, `WHICH_PC=PC2`.

## 4. Expected output

Written to `<out_dir>/`:

| File | Contents |
|---|---|
| `DE_results_table.csv` | cell-level DE: Gene, logFC, AveExpr, t, P.Value, adj.P.Val, B |
| `DE_results_table_pseudobulk_donor.csv` | donor-level DE, same columns (opt-in) |
| `DE_results_volcano.{png,pdf}` | panel A |
| `DE_results_corrgram_genes.{png,pdf}` | panel B |
| `DE_results_violin_PC1_3groups.pdf` | panel C |
| `DE_results_PCA_panel.{png,pdf}` | panel D |
| `DE_results_figure_ABCD.{png,pdf}` | assembled composite |
| `casestudy_panels.md` | knit log with all console diagnostics |

Reference run — `alzheimer_test_3`, 2,000 cells, **3m52s, 4.7 GB peak**:

```
Analysis matrix: 31660 genes x 2000 cells (Healthy n=1000, Diseased n=1000)
Contrast: Diseased - Healthy
Genes with FDR < 0.05: 16939   (Up 7149 | Down 9790)
PERMANOVA: R2 = 0.197, p <= 0.005
Donor-level FDR < 0.05: 3600   from 135 donors (Healthy 91, Diseased 44)
```

## 5. The panels — what each shows and why

### A — Volcano

Effect size (`logFC`) against evidence (`-log10 P`) for every gene at once, with
FDR and fold-change cutoffs drawn.

**Why:** it is the only panel that shows the *joint* distribution. Its value here
is diagnostic rather than decorative — it makes visible that most significant
genes carry `|logFC| < 0.25`, i.e. that significance is being driven by sample
size rather than effect magnitude.

**One design decision worth recording.** The y-axis is `-log10(P.Value)` while
colouring uses `adj.P.Val`. These are reconcilable because BH adjustment is
monotone in raw *p*: `{adj.P < a}` equals `{P < p*}` for
`p* = max(P.Value[adj.P.Val < a])`. The dashed line is drawn at that empirical
crossing point and its value printed in the subtitle (`raw p = 0.0267` on the
reference run — not 0.05). Plotting `-log10(adj.P.Val)` instead would also be
consistent but compresses the axis into horizontal stripes, because BH's
`cummin` ties large blocks of genes to identical adjusted values.

### B — Gene correlation (corrgram, top 20 DE genes)

Pairwise correlation among the top-ranked genes across cells.

**Why:** guards against reading a long gene list as a long list of *findings*. If
the top genes are mutually highly correlated, they are one axis of variation
observed 20 times, not 20 independent signals. Any pathway or enrichment
narrative built on the list should be discounted accordingly.

### C — PC1 violin (Healthy / Diseased / All)

Distribution of the dominant expression axis per group, with pairwise Wilcoxon
and a global Kruskal–Wallis test. The pooled "All" group is a reference
distribution, not a third condition.

**Why:** collapses the whole transcriptome to one interpretable number per cell
and asks whether the groups separate on it. Distribution shape also exposes
multimodality that a mean-based test would hide.

### D — PCA + PERMANOVA composite

PC1/PC2 scatter, marginal boxplots with compact-letter display, and a PERMANOVA
panel reporting df, R², and *p*.

**Why:** this is the only panel that reports **effect magnitude at the cohort
level**. A DE list can be enormous while global separation is weak; R² (0.197
here) says how much total variance group membership actually explains. It is the
correct check against over-reading panel A's gene count.

*p* is reported as an inequality (`p <= 0.005`) because a permutation test cannot
resolve below `1/(permutations + 1)`. Use `PERMUTATIONS=999` for publication.

## 6. Limitations

### Statistical — these determine whether the figure survives review

1. **Cells are treated as biological replicates.** limma in panel A, the Wilcoxon
   tests in panel C, and PERMANOVA in panel D all assume independence. Cells from
   one donor are correlated; the effective *n* is donors, not cells.

2. **The inflation is measured, not hypothetical.** On `alzheimer_test_3`:
   cell-level 16,939 significant genes vs donor-level 3,600. Cell-level FDRs
   reach `1e-154` across 2,000 correlated cells; donor-level bottoms out at
   `1e-9` across 135 donors. Only the second is plausible for the design.

3. **The two analyses disagree about *which* genes matter.** Top-100 overlap is
   **6/100**. 751 genes are donor-significant but *not* cell-significant — so the
   cell-level test is not merely over-permissive, it ranks a different set.
   `ABCA7` and `TARDBP` appear in the donor-level top 5 and nowhere near the
   cell-level top 12.

4. **Pseudobulk discards ~half the data.** 988 of 2,000 cells carry
   `donor_id == "unknown"`. This is a source-metadata limit, not fixable here.

5. **The donor design is imbalanced**: 91 healthy vs 44 diseased donors, far
   weaker than the 1000-vs-1000 cell counts suggest.

6. **Pseudobulk averages CP10K, not summed raw counts.** A common approximation,
   but it cannot recover depth weighting. Proper count-based pseudobulk requires
   exporting counts from the Python side.

7. **Batch and study are ignored entirely.** The reference cohort spans 62
   batches and dozens of `dataset_id`s; donors nest within studies, so study-level
   confounding survives even the donor-level contrast. A mixed model with donor
   nested in dataset is the next step.

8. **CP10K assumes each cell's total counts are comparable.** No depth covariate
   is fitted. Cohort-level depth imbalance is reported by
   `cohort_diagnostics.py`, not by this document — read
   `cohort_diagnostics.json` alongside these results.

9. **Bray–Curtis on log2 data is unconventional.** It is valid (values are
   non-negative) but is an abundance-style dissimilarity applied to a
   log-transformed matrix; Euclidean on the same matrix would be the more
   standard choice.

### Structural — these will break silently if ignored

10. **The `labels.csv` join is positional, not keyed.** The matrices carry no
    donor, batch, or dataset column. Verified for `alzheimer_test_3` that rows
    1–1000 are the disease block and 1001–2000 the normal block — but nothing in
    the code enforces this, and it breaks outright when `select_contrast()`
    populates `excluded`, since those cells appear in `labels.csv` and in neither
    matrix. The pseudobulk chunk checks the row count and block ordering and
    refuses rather than misaligning. **A cell id column in the export would
    remove this entire class of risk.**

11. **`ns_sample_i` / `ds_sample_i` names invite the error in item 1.** Any tool
    or reader that sees "sample" will assume replicate.

12. **Composite panel quality is degraded by design.** In `figure_ABCD`, panel
    D's PERMANOVA block clips and panel B's `panel.pie` circles render as
    ellipses, because everything is re-rendered into a quarter-width cell.
    The standalone files are correct — assemble publication figures from those.

13. **Panel B's capture route is fragile.** `corrgram` is base graphics laid out
    with `par(mfrow)`. `patchwork::wrap_elements()` composes it **without error
    and produces a near-empty panel**; only `ggplotify::as.ggplot()` captures it
    correctly. Do not "simplify" this.

14. **The scale assertion is a hard stop, deliberately.** If median column sums
    are not 1e4, the document errors instead of transforming. Magnitude alone
    cannot distinguish log-scale from low-valued linear data — the previous
    `q99 > 100` heuristic silently skipped the transform on this data (measured
    `q99 = 3.5`), which would have made `logFC` a difference of linear CP10K
    means mislabelled as a log2 fold change.

15. **The heatmap chunk is disabled** (`eval = FALSE`). `pheatmap` with complete
    linkage on 2,000 columns is slow, illegible at that width, and — because a
    knitr error aborts the document — would take panels A–D down with it.

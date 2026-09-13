No files were created or edited. I reviewed the declared pair—[casestudy.md](/storage3/fs1/fuhai.li/Active/di.huang/Research/LLM/OmniCellAgent/tools/omic_tools/casestudy.md:1), [casestudy_panels.Rmd](/storage3/fs1/fuhai.li/Active/di.huang/Research/LLM/OmniCellAgent/tools/omic_tools/casestudy_panels.Rmd:1), and its [runner](/storage3/fs1/fuhai.li/Active/di.huang/Research/LLM/OmniCellAgent/tools/omic_tools/run_casestudy.R:1). Their hashes were unchanged at the end of the review. I did not rerun the R Markdown because that would write outputs; instead, I inspected the existing post-source knit transcript and rendered figures.

## Overall verdict

- **Execution:** No blocker for the current Alzheimer reference dataset in its configured cluster environment. A successful render exists with all expected main outputs and no execution-error marker.
- **Specification:** **Partially conformant.** It captures the broad workflow unusually well and documents several genuine limitations, but contains materially incorrect descriptions of Panels A, C, D, and the optional donor aggregation. It is not sufficiently exact for an independent reimplementation.
- **Visualization:** Useful as **exploratory cell-level QC**, but **major revision is required** before it is defensible as a manuscript figure supporting Alzheimer’s-disease biology or population-level inference.
- **Composite:** The existing [A–D rendering](/storage3/fs1/fuhai.li/Active/di.huang/Research/LLM/OmniCellAgent/webapp/sessions/alzheimer_test_3/casestudy_R/DE_results_figure_ABCD.png) is not publication-ready and does not faithfully support all of the messages claimed in the specification.

## 1. Specification validation

| Area                                                            | Assessment                                                                                                                                                                                                                                                                                                                                                            |
| --------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Inputs, file discovery, broad execution order and output naming | Mostly accurate.                                                                                                                                                                                                                                                                                                                                                      |
| Scale validation                                                | Oversimplified. The spec says columns sum exactly to 10,000 and describes an equality assertion, but the code checks only each matrix’s**median** column sum within 1% ([Rmd](/storage3/fs1/fuhai.li/Active/di.huang/Research/LLM/OmniCellAgent/tools/omic_tools/casestudy_panels.Rmd:120)). Minority malformed, zero-sum, or partially missing cells can pass. |
| Separation of calculations from plotting                        | Overstated. Panel C computes Kruskal–Wallis and Wilcoxon tests inside its plotting chunk; Panel D computes ANOVA/Tukey compact letters; Panel B computes correlations during drawing.                                                                                                                                                                                |
| Panel A                                                         | Materially inaccurate. Colour does not simply encode FDR; both colour and shape encode the**joint** FDR-plus-effect threshold ([Rmd](/storage3/fs1/fuhai.li/Active/di.huang/Research/LLM/OmniCellAgent/tools/omic_tools/casestudy_panels.Rmd:274)).                                                                                                              |
| Panel B                                                         | Incomplete. The spec omits that this is pooled-condition Pearson correlation with angular PCA reordering. These are`corrgram` defaults, not explicit design choices in the code ([CRAN manual](https://stat.ethz.ch/CRAN/web/packages/corrgram/corrgram.pdf)).                                                                                                       |
| Panel C                                                         | Materially incorrect. PC1 is not a whole-transcriptome axis: it is PC1 of the top 100 disease-selected genes, standardized gene by gene. “All” is then tested as if it were an independent third group.                                                                                                                                                             |
| Panel D                                                         | Materially incorrect.\(R^2=0.197\) is not “cohort-level total variance explained.” It is the proportion of Bray–Curtis sums of squares in an unscaled, disease-selected 100-gene space.                                                                                                                                                                            |
| Donor “pseudobulk”                                            | Incorrectly described. The code averages`expr`, which has already been transformed to `log2(CP10K+1)`, not linear CP10K ([aggregation](/storage3/fs1/fuhai.li/Active/di.huang/Research/LLM/OmniCellAgent/tools/omic_tools/casestudy_panels.Rmd:619)).                                                                                                              |
| Reproducibility                                                 | Insufficient. The spec lacks exact package versions, RNG seed, input checksums, output dimensions, complete parameter validation, and a run manifest.                                                                                                                                                                                                                 |

The spec is correct about several important matters: contrast direction, the raw-\(p\)/FDR crossing in Panel A, cell pseudoreplication, batch confounding, positional metadata joining, Bray–Curtis being unconventional, permutation resolution, and composite distortion.

It is also more accurate than two passages in the Rmd itself: the Rmd says the composite is assembled “outside R” even though Section 9 assembles it ([Rmd](/storage3/fs1/fuhai.li/Active/di.huang/Research/LLM/OmniCellAgent/tools/omic_tools/casestudy_panels.Rmd:68)), and its final caveat points readers to Section 9 for pseudobulk although pseudobulk is Section 10.

## 2. Critical correctness issues

### Panel C tests a duplicated, non-independent “All” group

The code copies every Healthy and Diseased observation into `All`, binds those copies back to the original data, and then performs all three pairwise Wilcoxon tests plus Kruskal–Wallis ([Rmd](/storage3/fs1/fuhai.li/Active/di.huang/Research/LLM/OmniCellAgent/tools/omic_tools/casestudy_panels.Rmd:376)).

Consequences:

- Healthy-versus-All and Diseased-versus-All compare a group against a distribution containing that same group.
- The global Kruskal–Wallis test treats duplicated observations as independent.
- The displayed global \(p\) and two of the three brackets are invalid.
- The pairwise stars are also explicitly **unadjusted**, as confirmed by the current [knit transcript](/storage3/fs1/fuhai.li/Active/di.huang/Research/LLM/OmniCellAgent/webapp/sessions/alzheimer_test_3/casestudy_R/casestudy_panels.md:458) and [ggpubr documentation](https://rpkgs.datanovia.com/ggpubr/reference/stat_compare_means.html).

**Fix:** Remove `All` from all hypothesis tests—preferably remove it from the figure entirely. If a pooled density is retained as a visual reference, render it as a background distribution and explicitly state that it is a deterministic mixture, not an independent group.

### Panel A calls FDR-significant genes “NS” (Discard)

The saved DE table contains:

- 16,939 genes with FDR < 0.05;
- 605 meeting FDR < 0.05 and \(|\mathrm{logFC}|\ge0.25\);
- therefore 16,334 FDR-significant genes—96.4% of all FDR hits—are coloured grey and assigned the legends “NS” and “Not sig.”

This contradicts the intended message that the plot should reveal many statistically significant but small effects.

**Fix:** Use four truthful classes:

- FDR ≥ 0.05;
- FDR < 0.05, small effect;
- FDR < 0.05, positive effect threshold;
- FDR < 0.05, negative effect threshold.

Rename the shape legend to “Meets joint FDR + effect criterion,” rather than “Significant.”

### Panel D combines incompatible geometries

The PCA uses centered and gene-standardized expression:		

```r
prcomp(..., scale. = TRUE)
```

The adjacent PERMANOVA uses Bray–Curtis on unscaled `log2(CP10K+1)` values:

```r
vegdist(..., method = "bray")
```

Thus, the scatter and \(R^2\) do not summarize the same geometry. In addition, the genes were selected using the same disease labels later used to claim separation ([selection and testing](/storage3/fs1/fuhai.li/Active/di.huang/Research/LLM/OmniCellAgent/tools/omic_tools/casestudy_panels.Rmd:226)). That makes disease separation expected rather than independent confirmation. Reusing the same data for selection and selective analysis is the classic circular-analysis problem ([Kriegeskorte et al.](https://www.nature.com/articles/nn.2303)).

**Fix:** Either:

- use Euclidean distance on the exact centered/scaled matrix used by PCA; or
- retain Bray–Curtis and display a PCoA derived from that same Bray–Curtis matrix.

Use an unsupervised, pre-specified feature set—or label the panel plainly as “PCA of the 100 most disease-associated genes” and do not present it as validation or global transcriptomic separation.

### All inferential annotations use cells as replicates

The specification acknowledges this, but its panel interpretations still overstate the results. Cell-level limma \(p\)-values, Panel C tests, Panel D ANOVA letters, and unrestricted PERMANOVA permutations do not provide donor-level evidence.

The reference result illustrates the impact: 16,939 cell-level FDR hits versus 3,600 in the donor-mean table, with only 6 genes overlapping between the two top-100 lists. Single-cell studies that ignore biological-replicate variation are known to produce severe false-discovery inflation ([Squair et al.](https://www.nature.com/articles/s41467-021-25960-2), [Zimmerman et al.](https://www.nature.com/articles/s41467-021-21038-1)).

**Fix:** For a manuscript figure, make donors/studies the plotted and inferential units. A visualization-only repair can make the current panels honest as descriptive cell-level QC, but it cannot make their population-level \(p\)-values valid.

## 3. Scientific and biological concerns

### Panel B does not demonstrate a single shared biological program (Discard)

Pooling Healthy and Diseased cells can mix group mean shifts, group-specific covariance, donor structure and dataset effects. In a read-only recalculation of the exact top 20 genes:

- 21 of 190 gene pairs differed in correlation by at least 0.5 between Healthy and Diseased cells;
- 12 pairs had \(|r|\ge0.7\) in Healthy but \(|r|<0.3\) in Diseased.

A single pooled corrgram therefore conceals condition-specific structure and cannot justify the specification’s “one axis observed 20 times” interpretation. CP10K is also relative/compositional data, for which ordinary correlation can be misleading ([Lovell et al.](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004075)).

**Fix:** State the estimand, then calculate correlations separately by condition using donor-level or donor-residualized expression. Show a conventional diverging heatmap fixed to \([-1,1]\), with a legend and preferably uncertainty or differential-correlation results.

### The optional donor result is not a reliable correction yet

The aggregation:

- averages log-transformed expression rather than summing raw counts;
- gives each donor equal downstream weight regardless of cell count;
- ignores dataset;
- drops 49.4% of cells with unusable donor IDs;
- keys donors only as `group|donor`.

In the reference metadata, 11 such keys span multiple `dataset_id` values, involving 173 usable cells. The code therefore silently assumes those repeated identifiers represent the same person. Standard count-pseudobulk workflows aggregate raw counts by biological sample and cell type before normalization ([edgeR guide](https://www.bioconductor.org/packages/devel/bioc/vignettes/edgeR/inst/doc/edgeRUsersGuide.pdf)).

**Fix:** Export stable cell IDs, globally valid donor/study IDs, and raw counts. Aggregate using `dataset_id + donor_id + cell_type`, verify cross-study donor identity explicitly, and include study/batch covariates.

### PERMANOVA is incompletely qualified

Beyond the wrong experimental unit:

- permutations are free rather than constrained by donor/study;
- no seed is set;
- no dispersion assessment is performed;
- disease and dataset are strongly confounded.

The `vegan` documentation explicitly notes that PERMANOVA can confound location with dispersion and provides `betadisper` for checking it; it also supports constrained permutations through `strata`/permutation controls ([vegan documentation](https://vegandevs.github.io/vegan/reference/adonis.html)).

### Biological context is absent from the figure

The panels say only “Healthy” and “Diseased.” They omit Alzheimer’s disease, astrocytes, cell/nucleus status, tissue/region, numbers of cells and donors, and multi-study origin. Consequently, the figure is not self-contained and encourages readers to interpret study/batch structure as disease biology.

## 4. Visualization-design concerns

- The 22 × 5.5-inch one-row composite becomes unreadable at journal width. Panel B circles become ellipses and Panel D loses text; the spec correctly acknowledges this.
- Even standalone Panel B has clipped/overprinted gene labels, no correlation scale, and low-precision pie encoding.
- Panel C uses a red–green contrast and a palette inconsistent with Panels A and D.
- Panel D uses large opaque points for 2,000 cells, causing heavy overplotting; its legend obscures data.
- Panel A labels the 20 smallest adjusted \(p\)-values regardless of effect threshold or biological relevance, including grey “NS” genes.
- Panel C reports significance stars rather than exact valid estimates, effect sizes or uncertainty.
- Panel D’s compact letters come from cell-level ANOVA/Tukey tests, are unexplained, and are redundant for two groups.
- No panel shows donor-level sample sizes or uncertainty. Violin widths also conceal that `All` contains twice as many rows.

A defensible publication layout would be a 2×2 grid with one consistent, colourblind-safe group palette, harmonized typography, explicit biological titles, and a caption specifying the observation and inferential units.

## 5. Run blockers and code robustness

There is **no demonstrated blocker for the reference session**, but these conditional blockers remain:

- R 4.5.2 and every loaded package are hard dependencies. `pheatmap` is loaded even though its chunk is disabled, so its absence blocks all four panels.
- `rmarkdown::render()` requires Pandoc; the wrapper appropriately uses `knitr::knit()` ([runner](/storage3/fs1/fuhai.li/Active/di.huang/Research/LLM/OmniCellAgent/tools/omic_tools/run_casestudy.R:94)).
- The Rmd’s manual no-Pandoc parameter example omits `composite`, although a later chunk evaluates `params$composite`.
- Exactly one file must match each filename pattern.
- `WHICH_PC`, `PERMUTATIONS`, `n_pca` and `n_corrgram` are not validated.
- Genes may retain up to 20% non-finite values, but `prcomp()` and `vegdist()` cannot safely consume those missing values.
- The median scale check can miss zero-sum or malformed cells; negative “log2_already” inputs are not rejected before Bray–Curtis.
- Zero raw \(p\)-values would produce infinite volcano coordinates.
- No seed is set for PERMANOVA or label placement.
- No `sessionInfo()`, package lockfile, input checksum or plot-data manifest is saved.
- Optional outputs are not removed when disabled. A stale pseudobulk or composite file can therefore appear to belong to the latest knit.
- Bray–Curtis storage and permutation cost scale approximately quadratically with cell count, limiting generalization beyond the 2,000-cell example.

## Recommended disposition

For **exploratory QC**, retain the overall panel concept after correcting Panel A’s categories, removing Panel C’s invalid tests, relabeling Panel D as supervised, and making Panel B condition-specific.

For a **biologically and biostatistically defensible manuscript figure**, the minimum redesign is:

1. Base inference and primary plotting on donors/studies, not cells.
2. Repair the donor/data identity contract and use raw-count pseudobulk or an appropriate mixed model.
3. Use an unsupervised feature space—or explicitly label disease-selected views as descriptive.
4. Make PCA/PCoA and PERMANOVA use the same transformation and distance.
5. Show exact estimands, uncertainty, sample sizes, missingness and study context.
6. Assemble a 2×2 publication figure from corrected standalone panels.

As currently written, the figure can show what happened in these 2,000 selected cells, but it does not reliably demonstrate a general Alzheimer’s-disease astrocyte program.

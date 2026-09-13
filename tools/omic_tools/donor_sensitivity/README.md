# AD donor sensitivity analysis

This standalone folder audits whether the AD/control signal depends on study,
age/sex adjustment, or individual studies. It reads the completed donor-level
run; it does not change the original results or rerun enrichment.

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/storage3/fs1/fuhai.li/Active/di.huang/Research/LLM/OmniCellAgent/dataset_outputs/pseudobulk_runtime_cache/matplotlib \
/storage3/fs1/fuhai.li/Active/di.huang/cache/miniconda/omnicellagent/bin/python \
  tools/omic_tools/donor_sensitivity/run.py \
  --source-run webapp/sessions/pseudobulk_ad_astrocyte_deseq2_v2 \
  --output-dir webapp/sessions/ad_donor_sensitivity_v1 \
  --rscript /storage1/fs1/fuhai.li/Active/di.huang/cache/R/omic-deseq2/bin/Rscript
```

The CLI output directory must be under `webapp/sessions/`, must not already
exist, and must not be inside the source run. The
isolated R environment must contain DESeq2 and jsonlite; its library path is
set by the wrapper. R runs with `--vanilla` inside the new output folder.

## Models and controls

| Model | Donors | Formula |
|---|---|---|
| full_A | Fixed original AD/control complete-case cohort | `~ disease` |
| full_B | Same as A | `~ study + disease` |
| full_C | Same as A | `~ study + age + sex + disease` |
| shared_C | Studies containing both AD and controls | `~ study + age + sex + disease` |
| leave_one_out_01… | Shared cohort, one study removed in turn | Same adjusted formula |

Full A/B/C share donor order, raw counts, count-filter genes and original
validated DESeq2 median-of-ratios size factors. Dispersions are refitted for each
model. Subset models re-filter and re-estimate normalization. Complete covariates,
replication, design rank and residual degrees of freedom are checked; failures
are reported rather than silently changing the donor cohort or dropping
nonconstant covariates. Positive fold change always means higher in AD.

For controlled FDR comparisons, adaptive independent filtering is disabled.
BH uses the entire count-filter retained gene family, including p=1 placeholders
for unavailable/nonconverged tests during correction only; those genes keep
missing final adjusted P-values. This audit's significance counts can differ
from the original analysis, which used adaptive independent filtering.
The goal is effect stability, not maximizing significant genes.

## Outputs

- `manifest.json`, `cohort_support.csv`: models and study/disease support.
- `models/<name>/`: complete gene results, 95% Wald intervals, donor metadata,
  size factors, design matrices, fit logs and status.
- `summary/model_summary.csv`, `gene_effects.csv`, `effect_concordance.csv`.
- `summary/fold_change_concordance.png/pdf`: A/B, B/C and C/shared effect comparisons.
- `summary/focused_effect_forest.png/pdf`: all fitted models for five prespecified
  ribosome genes and five leading genes in the original adjusted analysis.
- `summary/within_study_effects.png/pdf`: descriptive donor mean-ratio effects;
  no per-study DE tests or P-values are invented.
- `summary/report.md`, supporting plotted-data tables and source-preservation audit.

Smaller subsets may lose significance because of reduced power. Evaluate fold
change direction, magnitude and uncertainty alongside P-values. Forest intervals
are approximate 95% Wald intervals, not multiple-testing-adjusted intervals.

Tests:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. MPLCONFIGDIR=/storage3/fs1/fuhai.li/Active/di.huang/Research/LLM/OmniCellAgent/dataset_outputs/pseudobulk_runtime_cache/matplotlib \
PSEUDOBULK_RSCRIPT=/storage1/fs1/fuhai.li/Active/di.huang/cache/R/omic-deseq2/bin/Rscript \
R_LIBS=/storage1/fs1/fuhai.li/Active/di.huang/cache/R/omic-deseq2/lib/R/library \
R_LIBS_USER=/storage1/fs1/fuhai.li/Active/di.huang/cache/R/omic-deseq2/lib/R/library \
/storage3/fs1/fuhai.li/Active/di.huang/cache/miniconda/omnicellagent/bin/python \
  -m pytest tools/omic_tools/donor_sensitivity/tests -q
```

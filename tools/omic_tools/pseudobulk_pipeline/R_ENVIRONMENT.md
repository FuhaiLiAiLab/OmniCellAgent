# Persistent DESeq2 runtime

The default R executable is:

```text
/storage1/fs1/fuhai.li/Active/di.huang/cache/R/omic-deseq2/bin/Rscript
```

The environment was cloned from the validated installation with all 138 Conda
package versions/builds preserved: R 4.5.3, DESeq2 1.50.2, jsonlite 2.0.0.
It is independent of the former temporary installation. No packages directly
under the user's other R library were modified.

Both entry points select this persistent runtime. `--rscript` and
`PSEUDOBULK_RSCRIPT` override it; the unified CLI also accepts `OMIC_RSCRIPT`.
Python wrappers use `Rscript --vanilla` and isolate R_LIBS, R_LIBS_USER and
R_LIBS_SITE to the chosen environment's `lib/R/library`.

## Persistent caches

The unified CLI creates the following defaults under the repository:

```text
dataset_outputs/pseudobulk_runtime_cache/matplotlib/  # MPLCONFIGDIR
dataset_outputs/pseudobulk_runtime_cache/numba/       # NUMBA_CACHE_DIR
dataset_outputs/pseudobulk_runtime_cache/working/     # TMPDIR
```

Explicit environment overrides are respected. These are cache/working files;
scientific outputs always live in the chosen session directory.

## Run the tests

From the repository root:

```bash
export PSEUDOBULK_RSCRIPT=/storage1/fs1/fuhai.li/Active/di.huang/cache/R/omic-deseq2/bin/Rscript
export R_LIBS=/storage1/fs1/fuhai.li/Active/di.huang/cache/R/omic-deseq2/lib/R/library
export R_LIBS_USER="$R_LIBS"
export R_LIBS_SITE="$R_LIBS"
export MPLCONFIGDIR=/storage3/fs1/fuhai.li/Active/di.huang/Research/LLM/OmniCellAgent/dataset_outputs/pseudobulk_runtime_cache/matplotlib
export NUMBA_CACHE_DIR=/storage3/fs1/fuhai.li/Active/di.huang/Research/LLM/OmniCellAgent/dataset_outputs/pseudobulk_runtime_cache/numba
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python -m pytest \
  tools/omic_tools/pseudobulk_pipeline/tests \
  tools/omic_tools/donor_sensitivity/tests -q
```

The tests use disposable fixtures; production execution has no dependency on
the previous temporary R installation or temporary cache paths.

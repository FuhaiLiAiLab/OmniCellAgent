#!/usr/bin/env Rscript
#
# Render casestudy_panels.Rmd against a session directory.
#
#   module load r/4.5.2
#   Rscript run_casestudy.R <session_dir> [out_dir]
#
# The two input matrices are discovered by pattern, so the disease name never
# has to be typed (and its apostrophes never have to be escaped):
#
#   foranalysis_combined_normal_df_*.csv   -> reference group
#   foranalysis_combined_disease_df_*.csv  -> alternate group
#
# Output defaults to <session_dir>/casestudy_R so nothing collides with the
# Python workflow's own outputs in the session root.
#
# Options are environment variables, so the common case stays a one-liner:
#   COMPOSITE=false     skip the assembled A-B-C-D figure (roughly 3x faster)
#   PSEUDOBULK=true     also run the donor-level contrast (section 10)
#   PERMUTATIONS=999    PERMANOVA permutations (default 999)
#   WHICH_PC=PC2        which component panel C plots (default PC1)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
  stop("Usage: Rscript run_casestudy.R <session_dir> [out_dir]", call. = FALSE)
}

session_dir <- normalizePath(args[1], mustWork = TRUE)
out_dir <- if (length(args) >= 2) args[2] else file.path(session_dir, "casestudy_R")

pick_one <- function(pattern, what) {
  hits <- list.files(session_dir, pattern = pattern)
  if (length(hits) == 0) {
    stop(sprintf("No %s matrix in %s (looked for %s)", what, session_dir, pattern),
         call. = FALSE)
  }
  if (length(hits) > 1) {
    stop(sprintf("%d candidate %s matrices in %s: %s\nPass the session of a single run.",
                 length(hits), what, session_dir, paste(hits, collapse = ", ")),
         call. = FALSE)
  }
  hits
}

ref_csv <- pick_one("^foranalysis_combined_normal_df_.*\\.csv$",  "reference")
alt_csv <- pick_one("^foranalysis_combined_disease_df_.*\\.csv$", "alternate")

env_flag <- function(name, default) {
  v <- Sys.getenv(name, unset = NA)
  if (is.na(v) || !nzchar(v)) return(default)
  isTRUE(tolower(v) %in% c("1", "true", "yes", "t"))
}
env_num <- function(name, default) {
  v <- suppressWarnings(as.numeric(Sys.getenv(name, unset = "")))
  if (is.na(v)) default else v
}

params <- list(
  session_dir   = session_dir,
  ref_csv       = ref_csv,
  alt_csv       = alt_csv,
  ref_label     = "Healthy",
  alt_label     = "Diseased",
  out_dir       = out_dir,
  out_prefix    = "DE_results",
  input_scale   = "linear_cp10k",
  fdr_thresh    = 0.05,
  fc_thresh     = 1,
  permutations  = env_num("PERMUTATIONS", 999),
  n_corrgram    = 20,
  n_pca         = 100,
  which_pc      = Sys.getenv("WHICH_PC", unset = "PC1"),
  composite     = env_flag("COMPOSITE", TRUE),
  do_pseudobulk = env_flag("PSEUDOBULK", FALSE),
  labels_csv    = "labels.csv"
)

dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

cat("session_dir  :", session_dir, "\n")
cat("reference    :", ref_csv, "\n")
cat("alternate    :", alt_csv, "\n")
cat("out_dir      :", out_dir, "\n")
cat("composite    :", params$composite, "\n")
cat("pseudobulk   :", params$do_pseudobulk, "\n")
cat("permutations :", params$permutations, "\n\n")

rmd <- file.path(
  dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1]))),
  "casestudy_panels.Rmd"
)
if (!file.exists(rmd)) rmd <- "casestudy_panels.Rmd"

# knit() rather than rmarkdown::render(): pandoc is not installed on this
# cluster, and knit() honours eval= chunk options while purl() does not.
knitr::opts_chunk$set(fig.path = file.path(out_dir, "fig-"))
knitr::knit(rmd, output = file.path(out_dir, "casestudy_panels.md"))

cat("\n=== DONE ===\n")
print(list.files(out_dir))

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

legacy_render <- function(args) {
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
}

# Standalone DE uses the existing panels Rmd algorithm. The positional render
# command remains available while the plotting chunks are migrated separately.
parse_de_config <- function(args) {
  allowed <- c("stage", "ref-csv", "alt-csv", "ref-label", "alt-label",
               "input-scale", "de-dir", "state-file", "diagnostics-file",
               "de-fdr", "top-n", "final-out-dir", "final-de-dir")
  options <- list()
  positional <- character()
  i <- 1L
  while (i <= length(args)) {
    if (startsWith(args[i], "--")) {
      key <- substring(args[i], 3L)
      if (!key %in% allowed || i == length(args) || startsWith(args[i + 1L], "--")) {
        stop("Unknown option or missing value: ", args[i], call. = FALSE)
      }
      if (!is.null(options[[key]])) stop("Duplicate option: --", key, call. = FALSE)
      options[[key]] <- args[i + 1L]
      i <- i + 2L
    } else {
      positional <- c(positional, args[i])
      i <- i + 1L
    }
  }
  if (length(positional) < 1L || length(positional) > 2L) {
    stop("Usage: Rscript run_casestudy.R --stage de <session_dir> [out_dir]", call. = FALSE)
  }
  get_option <- function(name, default) {
    if (is.null(options[[name]])) default else options[[name]]
  }
  session_dir <- normalizePath(positional[1], mustWork = TRUE)
  pick_csv <- function(option, pattern) {
    path <- options[[option]]
    if (is.null(path)) {
      hits <- list.files(session_dir, pattern = pattern, full.names = TRUE)
      if (length(hits) != 1L) stop("Expected one input for --", option,
                                  "; found ", length(hits), call. = FALSE)
      path <- hits
    } else if (!startsWith(path, "/")) {
      path <- file.path(session_dir, path)
    }
    normalizePath(path, mustWork = TRUE)
  }
  out_dir <- if (length(positional) == 2L) positional[2] else file.path(session_dir, "casestudy_R")
  config <- list(
    stage = get_option("stage", "de"), session_dir = session_dir,
    ref_csv = pick_csv("ref-csv", "^foranalysis_combined_normal_df_.*\\.csv$"),
    alt_csv = pick_csv("alt-csv", "^foranalysis_combined_disease_df_.*\\.csv$"),
    ref_label = get_option("ref-label", "Healthy"),
    alt_label = get_option("alt-label", "Diseased"),
    input_scale = get_option("input-scale", "linear_cp10k"),
    out_dir = out_dir,
    de_dir = get_option("de-dir", file.path(session_dir, "differential_expression")),
    final_out_dir = get_option("final-out-dir", out_dir),
    final_de_dir = get_option("final-de-dir", get_option("de-dir", file.path(session_dir, "differential_expression"))),
    state_file = get_option("state-file", file.path(out_dir, "analysis_state.rds")),
    diagnostics_file = get_option("diagnostics-file", NULL),
    de_fdr = suppressWarnings(as.numeric(get_option("de-fdr", "0.05"))),
    top_n = suppressWarnings(as.numeric(get_option("top-n", "1000")))
  )
  if (!identical(config$stage, "de")) stop("This entrypoint currently supports --stage de.", call. = FALSE)
  if (!is.finite(config$de_fdr) || config$de_fdr < 0 || config$de_fdr > 1) {
    stop("--de-fdr must be between 0 and 1.", call. = FALSE)
  }
  if (!is.finite(config$top_n) || config$top_n < 1 || config$top_n != floor(config$top_n)) {
    stop("--top-n must be a positive integer.", call. = FALSE)
  }
  labels <- c(config$ref_label, config$alt_label)
  if (any(!nzchar(trimws(labels))) || anyDuplicated(make.names(labels))) {
    stop("Reference and alternate labels must be nonempty and distinct R model names.", call. = FALSE)
  }
  if (!config$input_scale %in% c("linear_cp10k", "log2_already")) {
    stop("--input-scale must be linear_cp10k or log2_already.", call. = FALSE)
  }
  if (!is.null(config$diagnostics_file) && !file.exists(config$diagnostics_file)) {
    stop("Diagnostics file not found: ", config$diagnostics_file, call. = FALSE)
  }
  config
}

read_inputs <- function(config) {
  if (!requireNamespace("data.table", quietly = TRUE)) stop("R package data.table is required.")
  read_expr <- function(path) {
    dt <- data.table::fread(path)
    if (ncol(dt) < 3L) stop("Each group needs a gene column and at least two cells: ", path)
    if (!names(dt)[1] %in% c("Name", "Gene") && is.numeric(dt[[1]])) {
      stop("First column must contain gene identifiers: ", path)
    }
    genes <- as.character(dt[[1]])
    if (anyNA(genes) || any(!nzchar(trimws(genes))) || anyDuplicated(genes)) {
      stop("Gene identifiers must be nonempty and unique: ", path)
    }
    dt[, 1 := NULL]
    if (!all(vapply(dt, is.numeric, logical(1)))) stop("Expression columns must be numeric: ", path)
    mat <- as.matrix(dt)
    rownames(mat) <- genes
    storage.mode(mat) <- "numeric"
    if (any(is.infinite(mat)) || any(mat < 0, na.rm = TRUE)) {
      stop("Expression values must be nonnegative, without infinities: ", path)
    }
    mat
  }
  mat_ref <- read_expr(config$ref_csv)
  mat_alt <- read_expr(config$alt_csv)

  # casestudy_panels.Rmd: scale-check. Retain its median-column-sum check.
  if (identical(config$input_scale, "linear_cp10k")) {
    med_ref <- median(colSums(mat_ref, na.rm = TRUE))
    med_alt <- median(colSums(mat_alt, na.rm = TRUE))
    ok <- abs(med_ref - 1e4) / 1e4 < 0.01 && abs(med_alt - 1e4) / 1e4 < 0.01
    if (!ok) stop(sprintf("Expected linear CP10K; median column sums are %.1f / %.1f.", med_ref, med_alt))
  }

  # casestudy_panels.Rmd: combine. Reference columns precede alternate columns.
  common_genes <- intersect(rownames(mat_ref), rownames(mat_alt))
  if (length(common_genes) < 10L) stop("Too few overlapping genes between files.")
  mat_ref <- mat_ref[common_genes, , drop = FALSE]
  mat_alt <- mat_alt[common_genes, , drop = FALSE]
  expr <- cbind(mat_ref, mat_alt)
  group <- factor(c(rep(config$ref_label, ncol(mat_ref)), rep(config$alt_label, ncol(mat_alt))),
                  levels = c(config$ref_label, config$alt_label))
  colnames(expr) <- make.unique(colnames(expr), sep = "_")
  if (identical(config$input_scale, "linear_cp10k")) expr <- log2(expr + 1)
  expr <- expr[rowSums(is.finite(expr)) >= ncol(expr) * 0.8, , drop = FALSE]
  v <- apply(expr, 1, stats::var, na.rm = TRUE)
  expr <- expr[is.finite(v) & v > 0, , drop = FALSE]
  if (nrow(expr) < 2L) stop("Too few variable genes for limma.")
  if (any(rowSums(is.finite(expr[, group == config$ref_label, drop = FALSE])) < 2L) ||
      any(rowSums(is.finite(expr[, group == config$alt_label, drop = FALSE])) < 2L)) {
    stop("Retained genes need at least two finite observations in each group.")
  }
  message(sprintf("DE input: %d tested genes, %s n=%d, %s n=%d",
                  nrow(expr), config$ref_label, ncol(mat_ref), config$alt_label, ncol(mat_alt)))
  list(expr = expr, group = group, mat_ref = mat_ref, mat_alt = mat_alt,
       input_md5 = tools::md5sum(c(config$ref_csv, config$alt_csv)))
}

run_de <- function(inputs, config) {
  if (!requireNamespace("limma", quietly = TRUE)) stop("R package limma is required.")
  if (!requireNamespace("statmod", quietly = TRUE)) stop("R package statmod is required for robust limma.")
  expr <- inputs$expr
  group <- inputs$group
  # Verbatim statistical sequence from casestudy_panels.Rmd's de chunk.
  design <- model.matrix(~ 0 + group)
  colnames(design) <- make.names(levels(group))
  contrast_str <- paste(make.names(config$alt_label), "-", make.names(config$ref_label))
  cm <- limma::makeContrasts(contrasts = contrast_str, levels = design)
  fit <- limma::lmFit(expr, design)
  fit2 <- limma::contrasts.fit(fit, cm)
  fit2 <- limma::eBayes(fit2, robust = TRUE, trend = TRUE)
  res <- limma::topTable(fit2, coef = 1, number = Inf, sort.by = "P", adjust.method = "BH")
  res$Gene <- rownames(res)
  res <- res[, c("Gene", "logFC", "AveExpr", "t", "P.Value", "adj.P.Val", "B")]
  if (any(!is.finite(as.matrix(res[, c("logFC", "P.Value", "adj.P.Val")])))) {
    stop("limma returned non-finite DE statistics.")
  }
  list(schema_version = 1L, config = config, inputs = inputs, native = res,
       contrast = contrast_str, limma_version = as.character(utils::packageVersion("limma")))
}

export_results <- function(state, config) {
  res <- state$native
  # Preserve the original Python input order, restricted to R-tested genes.
  genes <- rownames(state$inputs$expr)
  res <- res[match(genes, res$Gene), , drop = FALSE]
  ref <- state$inputs$mat_ref[genes, , drop = FALSE]
  alt <- state$inputs$mat_alt[genes, , drop = FALSE]
  if (identical(config$input_scale, "log2_already")) {
    ref <- 2^ref - 1
    alt <- 2^alt - 1
    if (any(is.infinite(ref)) || any(is.infinite(alt))) stop("Cannot recover finite linear expression for effect_size.")
  }
  # Compatibility-only Cohen's d; does not affect limma, filtering, or FDR.
  n_ref <- rowSums(is.finite(ref))
  n_alt <- rowSums(is.finite(alt))
  pooled_sd <- sqrt(((n_ref - 1) * apply(ref, 1, stats::var, na.rm = TRUE) +
                    (n_alt - 1) * apply(alt, 1, stats::var, na.rm = TRUE)) /
                   (n_ref + n_alt - 2))
  cohens_d <- (rowMeans(alt, na.rm = TRUE) - rowMeans(ref, na.rm = TRUE)) / (pooled_sd + 1e-8)
  compat <- data.frame(Name = res$Gene, log2_fold_change = res$logFC,
                       effect_size = unname(cohens_d), p_value = res$P.Value,
                       FDR = res$adj.P.Val, is_significant = res$adj.P.Val < config$de_fdr,
                       abs_log2_fc = abs(res$logFC), check.names = FALSE)
  if (any(!is.finite(compat$effect_size))) stop("Cannot export non-finite effect_size.")
  significant <- compat[compat$is_significant, , drop = FALSE]
  top <- function(table, field, decreasing = FALSE) {
    head(table[order(table[[field]], decreasing = decreasing), , drop = FALSE], config$top_n)
  }
  tables <- list(
    unpaired_differential_expression_results = compat,
    significant_genes_by_fdr = top(significant, "FDR"),
    significant_genes_by_fc = top(significant, "abs_log2_fc", TRUE),
    significant_upregulated_genes = top(significant[significant$log2_fold_change > 0, , drop = FALSE], "p_value"),
    significant_downregulated_genes = top(significant[significant$log2_fold_change < 0, , drop = FALSE], "p_value")
  )
  diagnostics <- if (is.null(config$diagnostics_file)) character() else
    readLines(config$diagnostics_file, warn = FALSE, encoding = "UTF-8")
  for (directory in unique(c(config$out_dir, config$de_dir, dirname(config$state_file)))) {
    if (!dir.exists(directory) && !dir.create(directory, recursive = TRUE)) stop("Cannot create output directory: ", directory)
  }
  write_csv <- function(table, path, comments = character()) {
    temporary <- tempfile(pattern = ".de-", tmpdir = dirname(path))
    on.exit(unlink(temporary), add = TRUE)
    # Binary mode avoids a second locale-dependent conversion after UTF-8
    # encoding; in LC_CTYPE=C a text connection can corrupt diagnostic bytes.
    connection <- file(temporary, open = "wb")
    tryCatch({
      if (length(comments)) writeLines(enc2utf8(paste0("# ", comments)), connection, useBytes = TRUE)
      utils::write.table(table, connection, sep = ",", row.names = FALSE, col.names = TRUE,
                         quote = TRUE, na = "", qmethod = "double")
    }, finally = close(connection))
    if (!file.rename(temporary, path)) stop("Cannot write output: ", path)
  }
  for (name in names(tables)) write_csv(tables[[name]], file.path(config$de_dir, paste0(name, ".csv")), diagnostics)
  write_csv(state$native, file.path(config$out_dir, "DE_results_table.csv"))
  state$compat <- compat
  # Python validates files in a staging directory before moving them. Keep
  # reusable state pointed at the published paths, not deleted temp locations.
  state$config$out_dir <- config$final_out_dir
  state$config$de_dir <- config$final_de_dir
  if (!identical(config$final_out_dir, config$out_dir)) {
    state$config$state_file <- file.path(config$final_out_dir, basename(config$state_file))
  }
  state$diagnostics_text <- diagnostics
  state$config$diagnostics_file <- NULL
  temporary <- tempfile(pattern = ".state-", tmpdir = dirname(config$state_file))
  on.exit(unlink(temporary), add = TRUE)
  saveRDS(state, temporary)
  if (!file.rename(temporary, config$state_file)) stop("Cannot write analysis state.")
  message(sprintf("DE complete: %d tested genes, %d significant; contrast %s",
                  nrow(compat), nrow(significant), state$contrast))
  invisible(state)
}

main <- function(args) {
  if ("--stage" %in% args) {
    config <- parse_de_config(args)
    inputs <- read_inputs(config)
    state <- run_de(inputs, config)
    export_results(state, config)
  } else {
    legacy_render(args)
  }
}

if (sys.nframe() == 0L) main(commandArgs(trailingOnly = TRUE))

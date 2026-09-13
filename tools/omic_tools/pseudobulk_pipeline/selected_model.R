#!/usr/bin/env Rscript
# Fit exactly the selected AD-versus-control model from raw donor inputs.
# CLI: raw_pipeline_dir output_dir model alpha min_count min_gene_donors min_group_donors [export_expression]

selected_model_script_dir <- local({
  frames <- sys.frames()
  files <- lapply(frames, function(frame) frame$ofile)
  files <- Filter(function(path) !is.null(path), files)
  path <- if (length(files)) files[[length(files)]] else {
    argument <- commandArgs(trailingOnly = FALSE)
    sub("^--file=", "", argument[startsWith(argument, "--file=")][1L])
  }
  dirname(normalizePath(path, mustWork = TRUE))
})

canonical_output_path <- function(path) {
  if (!startsWith(path, "/")) path <- file.path(getwd(), path)
  suffix <- character()
  while (!file.exists(path) && !dir.exists(path)) {
    suffix <- c(basename(path), suffix)
    parent <- dirname(path)
    if (identical(path, parent)) stop("Cannot resolve output directory.")
    path <- parent
  }
  path <- normalizePath(path, mustWork = TRUE)
  for (part in suffix) {
    path <- if (part == "..") dirname(path) else if (part == ".") path else file.path(path, part)
  }
  path
}

read_selected_source <- function(raw_directory, cohort_directory, settings, input_api, model_api) {
  count_path <- file.path(raw_directory, "inputs", "pseudobulk_counts.csv")
  metadata_path <- file.path(raw_directory, "inputs", "donor_metadata.csv")
  inputs <- input_api$read_inputs(count_path, metadata_path)
  # Use the original AD-versus-control complete-case rule for every formula.
  reasons <- input_api$donor_exclusions(inputs$metadata, input_api$comparisons[[1L]])
  included <- lengths(reasons) == 0L
  audit <- inputs$metadata
  audit$included <- included
  audit$exclusion_reasons <- vapply(reasons, paste, character(1L), collapse = ";")
  model_api$write_csv(audit, file.path(cohort_directory, "donor_exclusions.csv"))
  metadata <- inputs$metadata[included, , drop = FALSE]
  metadata$age <- suppressWarnings(as.numeric(metadata$age))
  model_api$write_csv(metadata, file.path(cohort_directory, "donor_metadata.csv"))
  if (!nrow(metadata)) model_api$fail("no_complete_case_donors", "No covariate-complete, nonzero-library donors remain.")
  counts <- inputs$counts[, metadata$sample_id, drop = FALSE]
  keep <- rowSums(counts >= settings$min_count) >= settings$min_gene_donors
  if (!any(keep)) model_api$fail("no_genes_pass_count_filter", "No genes pass the full-cohort count filter.")
  retained <- counts[keep, , drop = FALSE]
  expected <- model_api$independent_ratio_factors(retained)
  # This estimates normalization alone: no preliminary DESeqDataSet fit.
  factors <- DESeq2::estimateSizeFactorsForMatrix(retained)
  if (any(!is.finite(factors)) || any(factors <= 0)) {
    model_api$fail("invalid_size_factors", "DESeq2 ratio factors must be finite and positive.")
  }
  relative_error <- abs(factors / expected$factors - 1)
  tolerance <- 1e-8
  if (any(relative_error > tolerance)) {
    model_api$fail("size_factor_validation_failed", "Full-cohort DESeq2 ratio factors differ from the independent calculation.")
  }
  factor_path <- file.path(cohort_directory, "size_factors.csv")
  model_api$write_csv(data.frame(sample_id = metadata$sample_id, size_factor = unname(factors)),
                      factor_path)
  # Historical sensitivity models read the original run's R CSV factors.
  # Preserve that serialization precision before fitting: even last-digit
  # input differences can change optimizer paths for extreme low-count genes.
  # Only normalization factors cross this boundary; raw counts stay integers.
  saved_factors <- utils::read.csv(factor_path, colClasses = "character", check.names = FALSE)
  factors <- as.numeric(saved_factors$size_factor)
  relative_error <- abs(factors / expected$factors - 1)
  if (any(!is.finite(factors)) || any(factors <= 0) || any(relative_error > tolerance)) {
    model_api$fail("size_factor_validation_failed", "Exported ratio factors differ from the independent calculation.")
  }
  validation <- list(max_relative_error = max(relative_error), tolerance = tolerance,
                     n_reference_genes = expected$n_reference_genes,
                     factor_precision = "R CSV serialization, matching historical full-cohort factor reuse")
  model_api$write_csv(data.frame(gene = rownames(counts), in_count_filter = keep),
                      file.path(cohort_directory, "gene_filter.csv"))
  model_api$write_json(validation, file.path(cohort_directory, "factor_validation.json"))
  list(counts = counts, metadata = metadata, keep = keep, original_factors = factors,
       factor_source = "estimated_full_cohort_ratio_factors", factor_validation = validation,
       original_comparison = cohort_directory, original_status = NULL,
       n_input_donors = nrow(inputs$metadata), n_excluded_donors = sum(!included),
       source_files = list(counts = count_path, metadata = metadata_path))
}

main <- function(args = commandArgs(trailingOnly = TRUE)) {
  if (!length(args) %in% c(7L, 8L)) {
    stop("Usage: selected_model.R raw_pipeline_dir output_dir model alpha min_count min_gene_donors min_group_donors [export_expression]")
  }
  if (!requireNamespace("jsonlite", quietly = TRUE) || !requireNamespace("DESeq2", quietly = TRUE)) {
    stop("R packages DESeq2 and jsonlite are required.")
  }
  numbers <- suppressWarnings(as.numeric(args[4L:7L]))
  if (any(!is.finite(numbers)) || numbers[1L] <= 0 || numbers[1L] >= 1 ||
      any(numbers[2L:4L] != floor(numbers[2L:4L])) || any(numbers[2L:3L] < 1) ||
      numbers[4L] < 3 || any(numbers[2L:4L] > .Machine$integer.max)) {
    stop("Require 0 < alpha < 1, positive integer count/gene thresholds, and min_group_donors >= 3.")
  }
  export <- if (length(args) == 8L) tolower(args[8L]) else "false"
  if (!export %in% c("true", "false")) stop("export_expression must be true or false.")
  source_path <- normalizePath(args[1L], mustWork = TRUE)
  output <- canonical_output_path(args[2L])
  if (output == source_path || startsWith(output, paste0(source_path, "/")) ||
      startsWith(source_path, paste0(output, "/"))) stop("Output must be separate from the raw pipeline directory.")
  protected <- file.path(output, c("manifest.json", "models", "cohort"))
  links <- Sys.readlink(protected)
  if (any(file.exists(protected)) || any(!is.na(links) & nzchar(links))) {
    stop("Refusing to overwrite existing selected-model outputs.")
  }
  if (!dir.exists(output) && !dir.create(output, recursive = TRUE)) stop("Cannot create selected-model output directory.")
  input_api <- new.env(parent = globalenv())
  model_api <- new.env(parent = globalenv())
  sys.source(file.path(selected_model_script_dir, "deseq2_analysis.R"), envir = input_api)
  sys.source(file.path(selected_model_script_dir, "..", "donor_sensitivity", "models.R"), envir = model_api)
  settings <- list(alpha = numbers[1L], min_count = numbers[2L],
                   min_gene_donors = numbers[3L], min_group_donors = numbers[4L],
                   independent_filtering = FALSE, p_adjust_method = "fixed-family BH",
                   complete_case_covariates = I(c("study", "age", "sex", "disease")),
                   export_expression = export == "true")
  manifest_path <- file.path(output, "manifest.json")
  manifest <- list(status = "running", execution_mode = "selected_model_only",
                   source_run = source_path, output_dir = output, settings = settings,
                   model_order = I(args[3L]), models = list(), fitted_model_count = 0L,
                   source_files = list(counts = file.path(source_path, "inputs", "pseudobulk_counts.csv"),
                                       metadata = file.path(source_path, "inputs", "donor_metadata.csv")),
                   started_at = format(Sys.time(), tz = "UTC", usetz = TRUE))
  model_api$write_json(manifest, manifest_path)
  tryCatch({
    cohort_directory <- file.path(output, "cohort")
    if (!dir.create(cohort_directory)) stop("Cannot reserve cohort directory.")
    source <- read_selected_source(source_path, cohort_directory, settings, input_api, model_api)
    definitions <- model_api$model_specs(source$metadata)
    selected <- Filter(function(spec) identical(spec$name, args[3L]), definitions$models)
    if (length(selected) != 1L) model_api$fail("unknown_model", paste("Selected model is unavailable:", args[3L]))
    model_api$write_csv(definitions$support, file.path(cohort_directory, "cohort_support.csv"))
    group_counts <- function(rows) as.list(table(factor(source$metadata$disease[rows], levels = c("control", "AD"))))
    manifest$full_cohort <- list(n_donors = nrow(source$metadata), group_counts = group_counts(rep(TRUE, nrow(source$metadata))),
                                 n_input_donors = source$n_input_donors, n_excluded_donors = source$n_excluded_donors)
    shared_rows <- source$metadata$study %in% definitions$shared_studies
    manifest$shared_cohort <- list(n_donors = sum(shared_rows), group_counts = group_counts(shared_rows),
                                   n_studies = length(definitions$shared_studies))
    manifest$shared_studies <- I(definitions$shared_studies)
    manifest$factor_validation <- source$factor_validation
    manifest$versions <- list(R = R.version.string, DESeq2 = as.character(utils::packageVersion("DESeq2")),
                              jsonlite = as.character(utils::packageVersion("jsonlite")))
    manifest$library_paths <- I(.libPaths())
    model_api$write_json(manifest, manifest_path)
    cat("[Selected model]", args[3L], "\n")
    flush.console()
    status <- model_api$fit_model(selected[[1L]], source, definitions$shared_studies, settings,
                                  file.path(output, "models", args[3L]), export_expression = settings$export_expression)
    manifest$models[[args[3L]]] <- status
    manifest$fitted_model_count <- status$fitted_model_count
    manifest$status <- if (identical(status$status, "success")) "success" else if (identical(status$status, "error")) "error" else "partial"
    manifest$completed_at <- format(Sys.time(), tz = "UTC", usetz = TRUE)
    model_api$write_json(manifest, manifest_path)
    cat("[Selected model]", args[3L], status$status, "\n")
    if (identical(status$status, "error")) 1L else 0L
  }, error = function(e) {
    manifest$status <- "error"
    manifest$reason_code <- if (!is.null(e$reason_code)) e$reason_code else "selected_model_error"
    manifest$reason <- conditionMessage(e)
    manifest$completed_at <- format(Sys.time(), tz = "UTC", usetz = TRUE)
    model_api$write_json(manifest, manifest_path)
    message(conditionMessage(e))
    1L
  })
}

if (sys.nframe() == 0L) {
  exit_code <- tryCatch(main(), error = function(e) { message(conditionMessage(e)); 1L })
  quit(save = "no", status = exit_code)
}

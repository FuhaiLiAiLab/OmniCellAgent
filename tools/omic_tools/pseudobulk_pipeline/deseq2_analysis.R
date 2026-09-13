#!/usr/bin/env Rscript
# Raw donor pseudobulk counts only. This runner never substitutes a DE method,
# resolves conflicting donor annotations, or simplifies a confounded model.
# Reference: https://bioconductor.org/packages/release/bioc/vignettes/DESeq2/inst/doc/DESeq2.html

comparisons <- list(
  list(name = "ad_vs_control", column = "disease", numerator = "AD",
       reference = "control", disease = NULL, nuisance = c("study", "age", "sex")),
  list(name = "ad_male_vs_female", column = "sex", numerator = "male",
       reference = "female", disease = "AD", nuisance = c("study", "age")),
  list(name = "control_male_vs_female", column = "sex", numerator = "male",
       reference = "female", disease = "control", nuisance = c("study", "age"))
)

abort_input <- function(code, message) {
  stop(structure(list(message = message, call = NULL, reason_code = code),
                 class = c("pseudobulk_input_error", "error", "condition")))
}

package_version_or_null <- function(name) {
  tryCatch(as.character(utils::packageVersion(name)), error = function(e) NULL)
}

write_table <- function(value, path) {
  utils::write.csv(value, path, row.names = FALSE, na = "", quote = TRUE)
}

write_matrix <- function(value, path) {
  write_table(data.frame(gene = rownames(value), value, check.names = FALSE), path)
}

write_status <- function(status, directory) {
  jsonlite::write_json(status, file.path(directory, "status.json"),
                       auto_unbox = TRUE, pretty = TRUE, null = "null", na = "null",
                       digits = NA)
}

status_template <- function(spec, settings) {
  requested <- paste("~", paste(c(spec$nuisance, spec$column), collapse = " + "))
  list(
    comparison = spec$name, status = "pending", reason_code = NULL, reason = NULL,
    alpha = settings$alpha,
    contrast = list(column = spec$column, numerator = spec$numerator,
                    reference = spec$reference),
    requested_formula = requested, formula = requested,
    included_donors = I(character()), excluded_donors = list(),
    dropped_constant_terms = I(character()), constant_term_audit = list(),
    complete_case_covariates = I(c("study", "age", "sex", "disease")),
    filter = list(min_count = settings$min_count,
                  min_gene_donors = settings$min_gene_donors,
                  rule = "raw count >= min_count in >= min_gene_donors included donors",
                  independent_filtering = TRUE),
    min_group_donors = settings$min_group_donors,
    normalization = list(method = "DESeq2 median-of-ratios", type = "ratio",
                         gene_scope = "genes passing the count filter",
                         zero_geometric_mean_gene_policy = "excluded from size factor reference",
                         fallback = "none"),
    test = list(method = "DESeq2 Wald", beta_prior = FALSE,
                automatic_outlier_replacement = FALSE,
                cooks_cutoff = "DESeq2 default: 0.99 quantile of F(p,m-p)",
                p_adjust_method = "BH", lfc_shrinkage = "none"),
    vst = list(method = "varianceStabilizingTransformation", blind = FALSE,
               gene_scope = "genes passing the count filter",
               sample_unit = "donor", used_for_differential_expression = FALSE),
    versions = list(R = R.version.string, DESeq2 = package_version_or_null("DESeq2"),
                    jsonlite = package_version_or_null("jsonlite")),
    warnings = I(character()), started_at = format(Sys.time(), tz = "UTC", usetz = TRUE)
  )
}

read_inputs <- function(counts_csv, metadata_csv) {
  raw <- utils::read.csv(counts_csv, check.names = FALSE, colClasses = "character",
                         na.strings = c("", "NA"))
  if (ncol(raw) < 2L || names(raw)[1L] != "gene" || nrow(raw) < 1L) {
    abort_input("invalid_counts", "Counts must contain a first column named gene, genes, and donor columns.")
  }
  genes <- raw[[1L]]
  samples <- names(raw)[-1L]
  if (anyNA(genes) || any(!nzchar(trimws(genes))) || anyDuplicated(genes) ||
      anyNA(samples) || any(!nzchar(trimws(samples))) || anyDuplicated(samples)) {
    abort_input("invalid_counts", "Gene and count-matrix sample identifiers must be nonempty and unique.")
  }
  values <- suppressWarnings(as.numeric(as.matrix(raw[-1L])))
  if (any(!is.finite(values)) || any(values < 0) || any(values != floor(values)) ||
      any(values > .Machine$integer.max)) {
    abort_input("invalid_counts", paste(
      "Counts must be finite, nonnegative integers no greater than",
      .Machine$integer.max, "; no rounding or normalization is performed."))
  }
  counts <- matrix(as.integer(values), nrow = nrow(raw),
                   dimnames = list(genes, samples))
  metadata <- utils::read.csv(metadata_csv, check.names = FALSE,
                              colClasses = "character", na.strings = c("", "NA"))
  required <- c("sample_id", "source", "dataset_id", "donor_id", "study", "disease",
                "sex", "age", "n_metacells", "library_size", "sex_conflict",
                "age_conflict", "disease_conflict")
  missing <- setdiff(required, names(metadata))
  if (length(missing) || anyDuplicated(names(metadata))) {
    abort_input("invalid_metadata", paste("Missing or duplicate metadata columns:", paste(missing, collapse = ", ")))
  }
  if (anyNA(metadata$sample_id) || anyDuplicated(metadata$sample_id) ||
      !setequal(metadata$sample_id, samples)) {
    abort_input("sample_mismatch", "Count columns and unique metadata sample_id values must match exactly.")
  }
  metadata <- metadata[match(samples, metadata$sample_id), , drop = FALSE]
  rownames(metadata) <- metadata$sample_id
  identity <- metadata[c("source", "dataset_id", "donor_id")]
  if (anyNA(identity) || any(!nzchar(trimws(as.matrix(identity)))) || anyDuplicated(identity)) {
    abort_input("invalid_metadata", "Each donor must have a unique, nonempty (source, dataset_id, donor_id) identity.")
  }
  for (column in c("sex_conflict", "age_conflict", "disease_conflict")) {
    flag <- tolower(trimws(metadata[[column]]))
    if (anyNA(flag) || any(!flag %in% c("true", "false", "1", "0"))) {
      abort_input("invalid_metadata", paste("Conflict flags must be explicit booleans:", column))
    }
    metadata[[column]] <- flag %in% c("true", "1")
  }
  actual_library_size <- colSums(counts)
  reported_library_size <- suppressWarnings(as.numeric(metadata$library_size))
  if (any(!is.finite(reported_library_size)) ||
      any(reported_library_size != actual_library_size)) {
    abort_input("library_size_mismatch", "Metadata library_size must equal the sum of raw input counts for each donor.")
  }
  metadata$library_size <- unname(actual_library_size)
  list(counts = counts, metadata = metadata)
}

donor_exclusions <- function(metadata, spec) {
  reasons <- lapply(seq_len(nrow(metadata)), function(i) character())
  add <- function(mask, reason) {
    for (i in which(mask)) reasons[[i]] <<- c(reasons[[i]], reason)
  }
  empty <- function(x) is.na(x) | !nzchar(trimws(x))
  for (column in c("study", "disease", "sex", "age")) {
    add(empty(metadata[[column]]), paste0("missing_", column))
  }
  for (column in c("sex", "age", "disease")) {
    add(metadata[[paste0(column, "_conflict")]], paste0(column, "_conflict"))
  }
  add(!empty(metadata$disease) & !metadata$disease %in% c("AD", "control"), "invalid_disease")
  add(!empty(metadata$sex) & !metadata$sex %in% c("male", "female"), "invalid_sex")
  age <- suppressWarnings(as.numeric(metadata$age))
  add(!empty(metadata$age) & (!is.finite(age) | age < 0), "invalid_age")
  add(metadata$library_size == 0, "zero_library")
  if (!is.null(spec$disease)) {
    add(!is.na(metadata$disease) & metadata$disease != spec$disease, "outside_comparison_disease")
  }
  reasons
}

run_comparison <- function(spec, inputs, settings, directory) {
  audit <- new.env(parent = emptyenv())
  audit$status <- status_template(spec, settings)
  finish <- function(state, code = NULL, reason = NULL) {
    audit$status$status <- state
    audit$status$reason_code <- code
    audit$status$reason <- reason
    audit$status$completed_at <- format(Sys.time(), tz = "UTC", usetz = TRUE)
    cat("\nComparison status:", state, if (!is.null(code)) code else "", "\n")
    if (!is.null(reason)) cat(reason, "\n")
    audit$status
  }
  log <- file(file.path(directory, "fit.log"), open = "wt")
  sink(log)
  sink(log, type = "message")
  on.exit({
    sink(type = "message")
    sink()
    close(log)
    write_status(audit$status, directory)
  }, add = TRUE)

  tryCatch(withCallingHandlers({
    cat("Comparison:", spec$name, "\n", R.version.string, "\n")
    # STEP 1: keep eligible donors and record every exclusion.
    metadata <- inputs$metadata
    reasons <- donor_exclusions(metadata, spec)
    include <- lengths(reasons) == 0L
    audit$status$included_donors <- I(metadata$sample_id[include])
    audit$status$excluded_donors <- lapply(which(!include), function(i) {
      list(sample_id = metadata$sample_id[i], source = metadata$source[i],
           dataset_id = metadata$dataset_id[i], donor_id = metadata$donor_id[i],
           reasons = I(reasons[[i]]))
    })
    exclusion_table <- metadata
    exclusion_table$included <- include
    exclusion_table$exclusion_reasons <- vapply(reasons, paste, character(1L), collapse = ";")
    write_table(exclusion_table, file.path(directory, "donor_exclusions.csv"))
    metadata <- metadata[include, , drop = FALSE]
    metadata$age <- suppressWarnings(as.numeric(metadata$age))
    counts <- inputs$counts[, metadata$sample_id, drop = FALSE]
    audit$status$n_input_donors <- nrow(inputs$metadata)
    audit$status$n_included_donors <- nrow(metadata)
    audit$status$n_excluded_donors <- sum(!include)
    audit$status$group_counts <- as.list(table(factor(metadata[[spec$column]],
                                                    levels = c(spec$reference, spec$numerator))))
    write_table(metadata, file.path(directory, "donor_metadata.csv"))

    if (!nrow(metadata)) {
      return(finish("skipped", "no_complete_case_donors", "No covariate-complete, nonzero-library donors remain."))
    }
    # STEP 2: construct the requested adjusted model and verify estimability.
    dropped <- spec$nuisance[vapply(spec$nuisance, function(term) {
      length(unique(metadata[[term]])) == 1L
    }, logical(1L))]
    audit$status$dropped_constant_terms <- I(dropped)
    audit$status$constant_term_audit <- lapply(dropped, function(term) {
      list(term = term, value = as.character(metadata[[term]][1L]),
           reason = "constant among covariate-complete included donors")
    })
    retained_terms <- c(setdiff(spec$nuisance, dropped), spec$column)
    formula <- stats::reformulate(retained_terms)
    audit$status$formula <- paste(deparse(formula), collapse = " ")
    cat("Requested formula:", audit$status$requested_formula, "\n")
    cat("Fitted formula:", audit$status$formula, "\n")
    cat("Dropped constant nuisance terms:", paste(dropped, collapse = ", "), "\n")
    if (any(unlist(audit$status$group_counts) < settings$min_group_donors)) {
      return(finish("skipped", "insufficient_group_donors",
                    paste("Each compared level requires at least", settings$min_group_donors, "eligible donors.")))
    }
    metadata$study <- factor(metadata$study, levels = sort(unique(metadata$study)))
    metadata$sex <- factor(metadata$sex, levels = c("female", "male"))
    metadata$disease <- factor(metadata$disease, levels = c("control", "AD"))
    metadata[[spec$column]] <- factor(metadata[[spec$column]],
                                    levels = c(spec$reference, spec$numerator))
    model <- stats::model.matrix(formula, data = metadata)
    qr_model <- qr(model)
    rank <- qr_model$rank
    aliased <- if (rank < ncol(model)) colnames(model)[qr_model$pivot[seq.int(rank + 1L, ncol(model))]] else character()
    audit$status$design <- list(n_rows = nrow(model), n_columns = ncol(model), rank = rank,
                                residual_df = nrow(model) - rank,
                                columns = I(colnames(model)), aliased_columns = I(aliased),
                                reference_level = levels(metadata[[spec$column]])[1L])
    write_table(data.frame(sample_id = rownames(model), model, check.names = FALSE),
                file.path(directory, "design_matrix.csv"))
    if (rank < ncol(model)) {
      return(finish("skipped", "design_not_full_rank",
                    "Requested adjusted design is confounded; no nonconstant covariate was removed."))
    }
    if (nrow(model) <= rank) {
      return(finish("skipped", "no_residual_degrees_of_freedom",
                    "The adjusted model has no residual degrees of freedom for dispersion estimation."))
    }
    target_coefficient <- paste0(spec$column, spec$numerator)
    if (!target_coefficient %in% colnames(model) ||
        !all(model[, target_coefficient] == as.integer(metadata[[spec$column]] == spec$numerator))) {
      return(finish("error", "contrast_direction_invalid", "Model coding does not match the explicit numerator/reference contrast."))
    }
    # STEP 3: filter low-count genes on the raw donor count matrix.
    keep <- rowSums(counts >= settings$min_count) >= settings$min_gene_donors
    audit$status$filter$n_input_genes <- nrow(counts)
    audit$status$filter$n_retained_genes <- sum(keep)
    audit$status$filter$n_filtered_genes <- sum(!keep)
    if (!any(keep)) {
      return(finish("skipped", "no_genes_pass_count_filter", "No input genes pass the raw-count filter in this comparison."))
    }
    ratio_reference <- rowSums(counts[keep, , drop = FALSE] > 0L) == ncol(counts)
    audit$status$normalization$n_reference_genes <- sum(ratio_reference)
    if (!any(ratio_reference)) {
      return(finish("skipped", "ratio_normalization_unavailable",
                    "Every retained gene has a zero in at least one donor; strict ratio size factors are undefined."))
    }
    if (!requireNamespace("DESeq2", quietly = TRUE)) {
      return(finish("error", "missing_deseq2", "DESeq2 is unavailable in this R library path."))
    }
    # STEP 4: fit raw genes × donors counts; DESeq2 estimates ratio size factors.
    dds <- DESeq2::DESeqDataSetFromMatrix(countData = counts[keep, , drop = FALSE],
                                          colData = metadata, design = formula)
    dds <- DESeq2::estimateSizeFactors(dds, type = "ratio")
    size_factors <- DESeq2::sizeFactors(dds)
    if (any(!is.finite(size_factors)) || any(size_factors <= 0)) {
      return(finish("error", "invalid_size_factors", "DESeq2 returned nonpositive or nonfinite ratio size factors."))
    }
    audit$status$test$requested_dispersion_fit <- "parametric"
    # Disable automatic count replacement; Cook's filtering remains at its default.
    dds <- DESeq2::DESeq(dds, test = "Wald", fitType = "parametric",
                         sfType = "ratio", betaPrior = FALSE,
                         minReplicatesForReplace = Inf, parallel = FALSE, quiet = FALSE)
    coefficient <- paste0(spec$column, "_", spec$numerator, "_vs_", spec$reference)
    if (!coefficient %in% DESeq2::resultsNames(dds)) {
      return(finish("error", "contrast_direction_invalid", "DESeq2 did not produce the requested numerator/reference coefficient."))
    }
    audit$status$contrast$coefficient <- coefficient
    audit$status$test$actual_dispersion_fit <- attr(DESeq2::dispersionFunction(dds), "fitType")
    audit$status$test$coefficients <- I(DESeq2::resultsNames(dds))
    result <- DESeq2::results(dds, contrast = c(spec$column, spec$numerator, spec$reference),
                              alpha = settings$alpha, independentFiltering = TRUE,
                              pAdjustMethod = "BH")
    normalized <- sweep(counts, 2L, size_factors, "/")
    all_results <- data.frame(gene = rownames(counts), baseMean = rowMeans(normalized),
                              log2FoldChange = NA_real_, lfcSE = NA_real_, stat = NA_real_,
                              pvalue = NA_real_, padj = NA_real_, tested = FALSE,
                              filter_reason = ifelse(rowSums(counts) == 0, "all_zero", "low_count"),
                              check.names = FALSE)
    result_frame <- as.data.frame(result)
    all_results[keep, names(result_frame)] <- result_frame
    all_results$tested[keep] <- is.finite(result_frame$pvalue)
    reasons <- rep("", nrow(result_frame))
    reasons[is.na(result_frame$padj) & is.finite(result_frame$pvalue)] <- "independent_filter"
    reasons[is.na(result_frame$pvalue)] <- "pvalue_unavailable"
    fit_metadata <- S4Vectors::mcols(dds)
    if (!is.null(fit_metadata$maxCooks)) {
      cutoff <- stats::qf(0.99, ncol(model), nrow(model) - ncol(model))
      cooks <- !is.na(fit_metadata$maxCooks) & fit_metadata$maxCooks > cutoff
      reasons[is.na(result_frame$pvalue) & cooks] <- "cooks_outlier"
    }
    if (!is.null(fit_metadata$betaConv)) {
      reasons[!is.na(fit_metadata$betaConv) & !fit_metadata$betaConv] <- "nonconverged"
    }
    all_results$filter_reason[keep] <- reasons
    audit$status$filter$n_finite_pvalues <- sum(is.finite(all_results$pvalue))
    audit$status$filter$n_finite_padj <- sum(is.finite(all_results$padj))
    audit$status$filter$result_filter_reasons <- as.list(table(all_results$filter_reason))
    result_metadata <- S4Vectors::metadata(result)
    audit$status$filter$independent_filter_threshold <- unname(result_metadata$filterThreshold)
    audit$status$filter$independent_filter_theta <- unname(result_metadata$filterTheta)
    # Full VST supports small retained gene sets; vst()'s nsub approximation does not.
    transformed <- DESeq2::varianceStabilizingTransformation(dds, blind = FALSE)
    expression <- SummarizedExperiment::assay(transformed)
    if (any(!is.finite(expression))) {
      return(finish("error", "invalid_vst", "VST produced nonfinite donor expression values."))
    }
    write_table(all_results, file.path(directory, "results.csv"))
    write_table(data.frame(sample_id = colnames(counts), size_factor = unname(size_factors)),
                file.path(directory, "size_factors.csv"))
    write_matrix(normalized, file.path(directory, "normalized_counts.csv"))
    write_matrix(expression, file.path(directory, "vst_expression.csv"))
    audit$status$vst$n_genes <- nrow(expression)
    audit$status$vst$n_donors <- ncol(expression)
    audit$status$normalization$matrix_gene_scope <- "all input genes"
    packages <- sort(loadedNamespaces())
    write_table(data.frame(package = packages, version = vapply(packages, function(p) {
      as.character(utils::packageVersion(p))
    }, character(1L))), file.path(directory, "package_versions.csv"))
    writeLines(capture.output(utils::sessionInfo()), file.path(directory, "session_info.txt"))
    audit$status$library_paths <- I(.libPaths())
    finish("success")
  }, warning = function(w) {
    audit$status$warnings <- I(c(audit$status$warnings, conditionMessage(w)))
    cat("WARNING:", conditionMessage(w), "\n")
    invokeRestart("muffleWarning")
  }), error = function(e) {
    finish("error", if (!is.null(e$reason_code)) e$reason_code else "deseq2_error", conditionMessage(e))
  })
}

main <- function(args = commandArgs(trailingOnly = TRUE)) {
  if (length(args) != 7L) {
    stop("Usage: deseq2_analysis.R counts_csv metadata_csv output_dir alpha min_count min_gene_donors min_group_donors")
  }
  if (!requireNamespace("jsonlite", quietly = TRUE)) stop("R package jsonlite is required.")
  numbers <- suppressWarnings(as.numeric(args[4L:7L]))
  if (any(!is.finite(numbers)) || numbers[1L] <= 0 || numbers[1L] >= 1 ||
      any(numbers[2L:4L] != floor(numbers[2L:4L])) || any(numbers[2L:3L] < 1) ||
      numbers[4L] < 3 || any(numbers[2L:4L] > .Machine$integer.max)) {
    stop("Require 0 < alpha < 1, positive integer count/gene thresholds, and min_group_donors >= 3.")
  }
  settings <- list(alpha = numbers[1L], min_count = numbers[2L],
                   min_gene_donors = numbers[3L], min_group_donors = numbers[4L])
  output <- args[3L]
  targets <- file.path(output, vapply(comparisons, `[[`, character(1L), "name"))
  if (any(file.exists(targets))) stop("Refusing to overwrite an existing comparison directory.")
  if (!dir.exists(output) && !dir.create(output, recursive = TRUE)) stop("Cannot create output directory.")
  for (directory in targets) if (!dir.create(directory)) stop("Cannot create comparison directory.")
  inputs <- tryCatch(read_inputs(args[1L], args[2L]), error = function(e) e)
  if (inherits(inputs, "error")) {
    for (i in seq_along(comparisons)) {
      status <- status_template(comparisons[[i]], settings)
      status$status <- "error"
      status$reason_code <- if (!is.null(inputs$reason_code)) inputs$reason_code else "invalid_input"
      status$reason <- conditionMessage(inputs)
      write_status(status, targets[i])
      writeLines(status$reason, file.path(targets[i], "fit.log"))
    }
    message(conditionMessage(inputs))
    return(1L)
  }
  statuses <- lapply(seq_along(comparisons), function(i) {
    run_comparison(comparisons[[i]], inputs, settings, targets[i])
  })
  for (status in statuses) cat(status$comparison, ": ", status$status, "\n", sep = "")
  if (any(vapply(statuses, function(s) s$status == "error", logical(1L)))) 1L else 0L
}

if (sys.nframe() == 0L) {
  exit_code <- tryCatch(main(), error = function(e) {
    message(conditionMessage(e))
    1L
  })
  quit(save = "no", status = exit_code)
}
